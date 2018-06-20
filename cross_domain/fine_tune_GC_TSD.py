# -*- coding: utf-8 -*-
# @Time    : 2017/11/10 下午6:34
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import sys
import visdom

sys.path.append("./")
from base.base_network import *


class Trace2Trace_LN:
    def __init__(self, ):
        self.pedestrian_num = 20
        self.hidden_size = 40

        # input
        self.input_frame = 5
        self.input_size = 2  # * self.input_frame

        # target
        self.target_frame = 5
        self.target_size = 2
        self.window_size = 1

        # learn
        self.lr = 1e-2
        self.weight_decay = 5e-3
        self.batch_size = 2000
        self.n_epochs = 100000

        # data
        self.train_loss_list = []
        self.test_loss_list = []

    def load_data(self, data_path):
        # load data
        data = np.load(data_path)
        train_X, train_Y = data['train_X'] + 0.5, data['train_Y'] + 0.5
        test_X, test_Y = data['test_X'] + 0.5, data['test_Y'] + 0.5

        if self.batch_size <= 0:
            self.batch_size = train_X.shape[0]

        self.test_input_traces = Variable(torch.from_numpy(test_X).cuda())
        self.test_target_traces = Variable(torch.from_numpy(test_Y).cuda())

        # (B, pedestrian_num, frame_size, 2)
        train_input_traces = torch.from_numpy(train_X)
        # (B, pedestrian_num, frame_size, 2)
        train_target_traces = torch.from_numpy(train_Y)

        self.train_input_traces = Variable(train_input_traces.cuda())
        self.train_target_traces = Variable(train_target_traces.cuda())

        # data loader
        train = torch.utils.data.TensorDataset(train_input_traces, train_target_traces)
        self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def main_compute_step(self, batch_input_traces, batch_target_traces):
        batch_size = batch_input_traces.size(0)

        target_traces = batch_input_traces[:, :, self.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in xrange(self.input_frame - 1):
            input_hidden_traces, encoder_hidden = self.encoder_net(batch_input_traces[:, :, i], encoder_hidden)

        regression_list = []

        for i in xrange(self.target_frame):
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

        return L2_square_loss.data[0], MSE_loss.data[0], regression_traces

    def train(self, train_input_traces, train_target_traces):

        # Zero gradients of both optimizers
        self.encoder_net.zero_grad()
        self.decoder_net.zero_grad()
        self.regression_net.zero_grad()

        L2_square_loss, MSE_loss, _ = self.main_compute_step(train_input_traces, train_target_traces)
        self.loss.backward()
        # MSE_loss.backward()

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.regression_optimizer.step()

        return L2_square_loss, MSE_loss

    def test(self):
        L2_square_loss, MSE_loss, _ = self.main_compute_step(self.test_input_traces, self.test_target_traces)
        return MSE_loss

    def save_model(self, model_name, epoch):
        dir_path = '%s/lr_%s_iter_%d' % (self.model_path, self.lr, epoch)
        if not os.path.exists(dir_path):
            os.system('mkdir -p  %s' % dir_path)

        torch.save(self.encoder_net.state_dict(), '%s/encoder_net.pkl' % dir_path)
        torch.save(self.decoder_net.state_dict(), '%s/decoder_net.pkl' % dir_path)
        torch.save(self.regression_net.state_dict(), '%s/regression_net.pkl' % dir_path)

    def init_net(self, pre_train):
        self.encoder_net = EncoderNetWithLSTM(self.pedestrian_num, self.input_size, self.hidden_size)
        self.decoder_net = DecoderNet(self.pedestrian_num, self.target_size, self.hidden_size, self.window_size)
        self.regression_net = RegressionNet(self.pedestrian_num, self.target_size, self.hidden_size)
        self.attn = Attention()

        if pre_train:
            self.encoder_net.load_state_dict(torch.load('%s/encoder_net.pkl' % self.pre_train_model_path))

        self.encoder_optimizer = optim.Adam(self.encoder_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.decoder_optimizer = optim.Adam(self.decoder_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.regression_optimizer = optim.Adam(self.regression_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.encoder_net.cuda()
        self.decoder_net.cuda()
        self.regression_net.cuda()

    def main(self, model_name, log, pre_train, pre_train_model_path):
        cur_time = time.strftime('%Y-%m-%d-%X', time.localtime())
        self.pre_train_model_path = pre_train_model_path
        self.model_path = 'model/%s/%s' % (model_name, cur_time)
        self.init_net(pre_train)
        self.vis = visdom.Visdom(env=model_name)

        if log:
            if not os.path.exists(self.model_path):
                os.system('mkdir -p  %s' % self.model_path)

            self.log_file = open('%s/message.log' % self.model_path, 'a')
            self.log_file.write('lr: %s, n_epochs: %d\n' % (self.lr, self.n_epochs) + '-' * 40 + '\n')
            self.log_file.flush()

        train_count = 0
        for epoch in xrange(1, self.n_epochs + 1):
            for i, (train_input_traces, train_target_traces) in enumerate(self.train_loader):
                train_input_traces = Variable(train_input_traces.cuda())
                train_target_traces = Variable(train_target_traces.cuda())

                L2_square_loss, MSE_loss = self.train(train_input_traces, train_target_traces)
                loss_msg = 'Epoch: [%d/%d], L2_suqare_loss: %.9f, MSE_loss: %.9f\n' % (epoch, self.n_epochs, L2_square_loss, MSE_loss)
                train_count += 1

                self.train_loss_list.append(MSE_loss)
                self.vis.line(Y=np.array(self.train_loss_list), X=np.arange(train_count), win='train_loss', opts={'title': 'train_loss'})

                print loss_msg
                if log:
                    self.log_file.write(loss_msg)
                    self.log_file.flush()

            test_loss = self.test()
            test_loss_msg = '----TEST----\n' + 'MSE Loss:%s\n\n' % test_loss
            self.test_loss_list.append(test_loss)
            self.vis.line(Y=np.array(self.test_loss_list), X=np.arange(epoch), win='test_loss', opts={'title': 'test_loss'})

            print test_loss_msg
            if log:
                self.log_file.write(test_loss_msg)
                self.log_file.flush()

            if epoch % 100 is 0:
                self.save_model(model_name, epoch)

        self.save_model(model_name, self.n_epochs)


def main():
    model = Trace2Trace_LN()
    model.load_data('data/TSD/data_set_tr5.npz')
    # model.main('fine-tune-GC-TSD', log=True, pre_train=True, pre_train_model_path='model/LSTM-NN')
    model.main('direct-TSD', log=True, pre_train=False, pre_train_model_path='model/LSTM-NN')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main()
