# -*- coding: utf-8 -*-
# @Time    : 2018/1/20 下午5:01
# @Author  : Zhixin Piao
# @Email   : piaozhx@shanghaitech.edu.cn

# Base
import numpy as np
import time
import random
import os

# Pytorch
from base.base_network import *
import visdom

# tools
from tools.helper import AverageMeter, adjust_learning_rate
import tools.beautiful_output


class Model:
    def __init__(self):
        self.pedestrian_num = 20
        self.hidden_size = 128

        # input
        self.input_frame = 5
        self.input_size = 2  # * self.input_frame
        self.n_layers = 2

        # target
        self.target_frame = 5
        self.target_size = 2
        self.window_size = 100

        # learn
        self.lr = 2e-3
        self.weight_decay = 5e-3
        self.batch_size = 256
        self.n_epochs = 10000
        self.test_interval = 5

        # show
        self.vis = visdom.Visdom('http://admin', port=31070, env='GC256')
        self.train_loss_list = []
        self.test_loss_list = []
        self.time_window = 300

        # data
        self.data_path = 'data/GC.npz'
        self.load_data()

    def load_data(self):
        # load data
        data = np.load(self.data_path)
        train_X, train_Y = data['train_X'], data['train_Y']
        test_X, test_Y = data['test_X'], data['test_Y']

        if self.batch_size <= 0:
            self.batch_size = train_X.shape[0]

        self.test_input_traces = torch.FloatTensor(test_X).cuda()
        self.test_target_traces = torch.FloatTensor(test_Y).cuda()

        # (B, pedestrian_num, frame_size, 2)
        train_input_traces = torch.FloatTensor(train_X)
        # (B, pedestrian_num, frame_size, 2)
        train_target_traces = torch.FloatTensor(train_Y)

        self.train_input_traces = train_input_traces.cuda()
        self.train_target_traces = train_target_traces.cuda()

        # data loader
        train = torch.utils.data.TensorDataset(train_input_traces, train_target_traces)
        self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def init_net(self):
        self.encoder_net = EncoderNetWithLSTM(self.pedestrian_num, self.input_size, self.hidden_size, n_layers=self.n_layers).cuda()
        self.decoder_net = DecoderNet(self.pedestrian_num, self.target_size, self.hidden_size, self.window_size).cuda()
        self.regression_net = RegressionNet(self.pedestrian_num, self.target_size, self.hidden_size).cuda()
        self.attn = Attention()

        self.encoder_optimizer = optim.Adam(self.encoder_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.decoder_optimizer = optim.Adam(self.decoder_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.regression_optimizer = optim.Adam(self.regression_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def main_compute_step(self, batch_input_traces, batch_target_traces):
        batch_size = batch_input_traces.size(0)

        target_traces = batch_input_traces[:, :, self.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in range(self.input_frame - 1):
            input_hidden_traces, encoder_hidden = self.encoder_net(batch_input_traces[:, :, i], encoder_hidden)

        regression_list = []

        for i in range(self.target_frame):
            # encode LSTM
            input_hidden_traces, encoder_hidden = self.encoder_net(target_traces, encoder_hidden)

            # NN with Attention
            target_hidden_traces = self.decoder_net(target_traces)
            Attn_nn = self.attn(target_hidden_traces, target_hidden_traces)
            c_traces = torch.bmm(Attn_nn, input_hidden_traces)

            # predict next frame traces
            regression_traces = self.regression_net(c_traces, target_hidden_traces, target_traces)

            # decoder --> location
            target_traces = regression_traces

            regression_list.append(regression_traces)

        regression_traces = torch.stack(regression_list, 2)

        # compute loss
        L2_square_loss = ((batch_target_traces - regression_traces) ** 2).sum() / self.pedestrian_num
        MSE_loss = ((batch_target_traces - regression_traces) ** 2).sum(3).sqrt().mean()

        self.loss = L2_square_loss

        return L2_square_loss.item(), MSE_loss.item(), regression_traces

    def train(self, epoch):
        MSE_loss_meter = AverageMeter()
        L2_square_loss_meter = AverageMeter()
        adjust_learning_rate([self.encoder_optimizer, self.decoder_optimizer, self.regression_optimizer], self.lr, epoch)

        for i, (train_input_traces, train_target_traces) in enumerate(self.train_loader):
            train_input_traces = train_input_traces.cuda()
            train_target_traces = train_target_traces.cuda()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.regression_optimizer.zero_grad()

            L2_square_loss, MSE_loss, _ = self.main_compute_step(train_input_traces, train_target_traces)
            MSE_loss_meter.update(MSE_loss)
            L2_square_loss_meter.update(L2_square_loss)

            # Update parameters with optimizers
            self.loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.regression_optimizer.step()

        return MSE_loss_meter.avg, L2_square_loss_meter.avg

    def test(self):
        with torch.no_grad():
            L2_square_loss, MSE_loss, _ = self.main_compute_step(self.test_input_traces, self.test_target_traces)
            return MSE_loss

    def run(self):
        self.init_net()

        for epoch in range(1, self.n_epochs + 1):
            MSE_loss, L2_square_loss = self.train(epoch)
            print('Epoch: [%d/%d], L2_suqare_loss: %.9f, MSE_loss: %.9f' % (epoch, self.n_epochs, L2_square_loss, MSE_loss))

            self.train_loss_list.append(MSE_loss)
            self.vis.line(np.array(self.train_loss_list[-self.window_size:]), win='train', opts={'title': 'train loss'})

            if epoch % self.test_interval == 0:
                test_loss = self.test()
                self.test_loss_list.append(test_loss)
                self.vis.line(np.array(self.test_loss_list[-self.window_size:]), win='test', opts={'title': 'test loss'})
                print('----TEST----\n' + 'MSE Loss:%s' % test_loss)


def set_random_seed(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

def main():
    set_random_seed()
    model = Model()
    model.run()


if __name__ == '__main__':
    main()
