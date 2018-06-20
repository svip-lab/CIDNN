# -*- coding: utf-8 -*-
# @Time    : 2017/10/18 上午1:47
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import scipy.io as sio

from base.base_network import *


class Trace2Trace:
    def __init__(self, data_path, multi_frame_data_path):
        self.data_path = data_path
        self.multi_frame_data_path = multi_frame_data_path
        self.pedestrian_num = 20
        self.hidden_size = 40

        # input
        self.input_frame = 5
        self.input_size = 2 * self.input_frame

        # target
        self.target_frame = 5
        self.target_size = 2
        self.window_size = 1

        self.min_target_frame = 1
        self.max_target_frame = 25

    def load_multi_frame_data(self):
        with open(self.multi_frame_data_path, 'r') as f:
            data = pickle.load(f)

        train_X, train_Y = data['train_X'], data['train_Y']
        test_X, test_Y = data['test_X'], data['test_Y']

        if self.use_gpu:
            self.train_input_traces_frame_list = [Variable(torch.FloatTensor(f_train_X).cuda()) for f_train_X in train_X]
            self.train_target_traces_frame_list = [Variable(torch.FloatTensor(f_train_Y).cuda()) for f_train_Y in train_Y]

            self.test_input_traces_frame_list = [Variable(torch.FloatTensor(f_test_X).cuda()) for f_test_X in test_X]
            self.test_target_traces_frame_list = [Variable(torch.FloatTensor(f_test_Y).cuda()) for f_test_Y in test_Y]
        else:
            self.train_input_traces_frame_list = [Variable(torch.FloatTensor(f_train_X)) for f_train_X in train_X]
            self.train_target_traces_frame_list = [Variable(torch.FloatTensor(f_train_Y)) for f_train_Y in train_Y]

            self.test_input_traces_frame_list = [Variable(torch.FloatTensor(f_test_X)) for f_test_X in test_X]
            self.test_target_traces_frame_list = [Variable(torch.FloatTensor(f_test_Y)) for f_test_Y in test_Y]

    def load_model(self, model_path):
        self.encoder_net = EncoderNetWithLSTM(self.pedestrian_num, self.input_size, self.hidden_size, self.use_gpu)
        self.decoder_net = DecoderNet(self.pedestrian_num, self.target_size, self.hidden_size, self.window_size)
        self.regression_net = RegressionNet(self.pedestrian_num, self.target_size, self.hidden_size)
        self.attn = Attention()

        self.encoder_net.load_state_dict(torch.load('%s/encoder_net.pkl' % model_path))
        self.decoder_net.load_state_dict(torch.load('%s/decoder_net.pkl' % model_path))
        self.regression_net.load_state_dict(torch.load('%s/regression_net.pkl' % model_path))

        if self.use_gpu:
            self.encoder_net.cuda()
            self.decoder_net.cuda()
            self.regression_net.cuda()

    def main_compute_step(self, batch_input_traces, batch_target_traces):
        batch_size = batch_input_traces.size(0)
        target_traces = batch_input_traces[:, :, self.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in xrange(self.input_frame - 1):
            encoder_output, encoder_hidden = self.encoder_net(batch_input_traces[:, :, i], encoder_hidden)

        regression_list = []
        Attn_list = []
        inner_Attn_list = []
        current_batch_input_traces = batch_input_traces[:, :, self.input_frame - 1]

        for i in xrange(self.target_frame):
            # NN with Attention
            target_hidden_traces = self.decoder_net(target_traces)
            Attn_nn, inner_Attn = self.attn(target_hidden_traces, target_hidden_traces)
            Attn_list.append(Attn_nn)
            inner_Attn_list.append(inner_Attn)

            # target_attn_hidden_traces_nn = torch.bmm(Attn_nn, target_hidden_traces)

            # LSTM with Attention
            input_hidden_traces, encoder_hidden = self.encoder_net(current_batch_input_traces, encoder_hidden)
            # Attn_lstm = self.attn(input_hidden_traces, input_hidden_traces)
            input_attn_hidden_traces_lstm = torch.bmm(Attn_nn, input_hidden_traces)

            # concat LSTM hidden with Attention and NN hidden with Attention
            # input_attn_hidden_traces = torch.cat((input_attn_hidden_traces_lstm, target_attn_hidden_traces_nn), 2)
            input_attn_hidden_traces = input_attn_hidden_traces_lstm

            # predict next frame traces
            regression_traces = self.regression_net(input_attn_hidden_traces, target_hidden_traces, target_traces)
            target_traces = regression_traces
            regression_list.append(regression_traces)

            current_batch_input_traces = regression_traces

        regression_traces = torch.stack(regression_list, 2)
        Attn_cube = torch.stack(Attn_list, 1)
        inner_Attn_cube = torch.stack(inner_Attn_list, 1)

        # compute loss

        L2_square_loss = ((batch_target_traces - regression_traces) ** 2).sum() / self.pedestrian_num
        MSE_loss = ((batch_target_traces - regression_traces) ** 2).sum(3).sqrt().mean()

        self.loss = L2_square_loss

        return L2_square_loss.data[0], MSE_loss.data[0], regression_traces, Attn_cube, inner_Attn_cube

    def get_multi_data(self, target_frame, data_type, batch_size):
        if data_type == 'test':
            input_traces = self.test_input_traces_frame_list[target_frame][:batch_size]
            target_traces = self.test_target_traces_frame_list[target_frame][:batch_size]
        elif data_type == 'train':
            input_traces = self.train_input_traces_frame_list[target_frame][:batch_size]
            target_traces = self.train_target_traces_frame_list[target_frame][:batch_size]

        else:
            return None, None

        if batch_size <= 0:
            return input_traces, target_traces
        else:
            return input_traces[:batch_size], target_traces[:batch_size]

    def main(self, model_path, use_gpu=True):
        self.use_gpu = use_gpu
        self.load_model(model_path)

    def evaluate_frame(self, data_type='test', batch_size=-1):
        self.load_multi_frame_data()
        for target_frame in xrange(self.min_target_frame, self.max_target_frame + 1):
            self.target_frame = target_frame
            input_traces, ground_truth_traces = self.get_multi_data(target_frame - self.min_target_frame, data_type, batch_size)
            L2_square_loss, MSE_loss, regression_traces, Attn_cube, _ = self.main_compute_step(input_traces, ground_truth_traces)

            gt_delta = ((input_traces[:, :, 0, :] - ground_truth_traces[:, :, target_frame - 1, :]) ** 2).sum(2).sqrt().mean()
            xy_delta = ((input_traces[:, :, 0, :] - regression_traces[:, :, target_frame - 1, :]) ** 2).sum(2).sqrt().mean()

            print 'target_frame: %d L2_square_loss:%s,  MSE_loss: %s' % (target_frame, L2_square_loss, MSE_loss)
            print 'average gt_xy_delta: ', gt_delta.data[0]
            print 'average regression_xy_delta: ', xy_delta.data[0]


def main():
    model = Trace2Trace('data/GC/xy_data_set.npz', 'data/GC/multi_frame_GC.data')
    model.main(model_path='model/LSTM-NN', use_gpu=True)
    model.evaluate_frame()

    # model.draw_scatter()


if __name__ == '__main__':
    main()
