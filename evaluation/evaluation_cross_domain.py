# -*- coding: utf-8 -*-
# @Time    : 2018/1/25 下午7:23
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn


import sys
import visdom

sys.path.append("./")
from base.base_network import *
from base.unfixed_network import *
import cv2


class Trace2Trace:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pedestrian_num = 20
        self.hidden_size = 40

        # input
        self.input_frame = 5
        self.input_size = 2 * self.input_frame

        # target
        self.target_frame = 5
        self.target_size = 2
        self.window_size = 1

        self.min_target_frame = 5
        self.max_target_frame = 25

    def load_data(self):
        # load data
        data = np.load(self.data_path)
        if self.data_path.find('GC') != -1:
            test_X, test_Y = data['test_X'], data['test_Y']
        else:
            test_X, test_Y = data['test_X'], data['test_Y']

        if self.use_gpu:
            self.test_input_traces = Variable(torch.from_numpy(test_X).cuda())
            self.test_target_traces = Variable(torch.from_numpy(test_Y).cuda())

        else:
            self.test_input_traces = Variable(torch.from_numpy(test_X))
            self.test_target_traces = Variable(torch.from_numpy(test_Y))

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

    def get_data(self, data_type, batch_size):
        if data_type == 'test':
            input_traces = self.test_input_traces
            target_traces = self.test_target_traces
        elif data_type == 'train':
            input_traces = None
            target_traces = None

        else:
            return None, None

        if batch_size <= 0:
            self.batch_size = input_traces.size(0)
            return input_traces, target_traces
        else:
            self.batch_size = batch_size
            return input_traces[:batch_size], target_traces[:batch_size]

    def run(self, model_path, use_gpu=True, batch_size=-1, data_type='test'):
        self.use_gpu = use_gpu
        self.load_model(model_path)

        self.load_data()
        self.input_traces, self.ground_truth_traces = self.get_data(data_type, batch_size)
        L2_square_loss, MSE_loss, self.regression_traces = self.main_compute_step(self.input_traces, self.ground_truth_traces)
        print ' L2_square_loss:%s,  MSE_loss: %s' % (L2_square_loss, MSE_loss)


class UnfixedTrace2Trace_LN:
    def __init__(self, ):
        self.hidden_size = 40
        self.max_pedestrian_num = 40
        self.unfixed_data_set_id = [1, 2, 3, 4, 5]
        self.save_epoch = 100

        # input
        self.input_frame = 8
        self.input_size = 2  # * self.input_frame

        # target
        self.target_frame = 12
        self.target_size = 2
        self.window_size = 1

        # learn
        self.lr = 2e-3
        self.weight_decay = 5e-3
        self.batch_size = -1
        self.n_epochs = 1000

        # data
        self.train_loss_list = []
        self.test_loss_list = []

    def load_unfixed_data(self, test_on):
        # load train test data
        def load_train_test_data():
            data_path_map = {i: 'data/other5dataset/xy_8_12_train_set_0%d.data' % i for i in self.unfixed_data_set_id}
            train_input_data, train_target_data = [], []
            test_input_data, test_target_data = [], []

            for i, data_path in data_path_map.iteritems():
                with open(data_path, 'r') as f:
                    data_set = pickle.load(f)
                    if i == test_on:
                        test_input_data += data_set['train_X']
                        test_target_data += data_set['train_Y']
                    else:
                        train_input_data += data_set['train_X']
                        train_target_data += data_set['train_Y']

            return train_input_data, train_target_data, test_input_data, test_target_data

        def create_zero_pad(trace, trace_type):
            frame_len = self.input_frame if trace_type == 'input' else self.target_frame
            zero_part = np.zeros((self.max_pedestrian_num - len(trace), frame_len, self.input_size))
            expand_trace = np.concatenate([np.array(trace), zero_part], axis=0)

            return expand_trace

        def expand_traces(input_data, target_data):
            input_traces, target_traces, traces_pedestrian_num = [], [], []
            for input_trace, target_trace in zip(input_data, target_data):
                input_traces.append(create_zero_pad(input_trace, 'input'))
                target_traces.append(create_zero_pad(target_trace, 'target'))

                traces_pedestrian_num.append(len(input_trace))

            input_traces = Variable(torch.FloatTensor(input_traces)).cuda()
            target_traces = Variable(torch.FloatTensor(target_traces)).cuda()

            batch_size = input_traces.size(0)
            hidden_mask, regression_mask = [], []
            for i in xrange(batch_size):
                p_n = traces_pedestrian_num[i]
                hidden_mask.append(np.concatenate([np.ones((p_n, self.hidden_size)),
                                                   np.zeros((self.max_pedestrian_num - p_n, self.hidden_size))], axis=0))
                regression_mask.append(np.concatenate([np.ones((p_n, self.input_size)),
                                                       np.zeros((self.max_pedestrian_num - p_n, self.input_size))], axis=0))

            hidden_mask = Variable(torch.FloatTensor(hidden_mask)).cuda()
            regression_mask = Variable(torch.FloatTensor(regression_mask)).cuda()
            traces_pedestrian_num = Variable(torch.FloatTensor(traces_pedestrian_num)).cuda()

            mask_list = [hidden_mask, regression_mask, traces_pedestrian_num]
            return input_traces, target_traces, mask_list

        train_input_data, train_target_data, test_input_data, test_target_data = load_train_test_data()

        if self.batch_size <= 0:
            self.batch_size = len(train_input_data)

        self.train_input_traces, self.train_target_traces, train_mask_list = expand_traces(train_input_data, train_target_data)
        self.train_hidden_mask, self.train_regression_mask, self.train_traces_pedestrian_num = train_mask_list

        self.test_input_traces, self.test_target_traces, test_mask_list = expand_traces(test_input_data, test_target_data)
        self.test_hidden_mask, self.test_regression_mask, self.test_traces_pedestrian_num = test_mask_list

    def main_compute_step(self, batch_input_traces, batch_target_traces, hidden_mask, regression_mask, batch_traces_pedestrian_num):
        target_traces = batch_input_traces[:, :, self.input_frame - 1]
        batch_size = target_traces.size(0)
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in xrange(self.input_frame - 1):
            encoder_output, encoder_hidden = self.encoder_net(batch_input_traces[:, :, i], encoder_hidden)

        regression_list = []
        current_batch_input_traces = batch_input_traces[:, :, self.input_frame - 1]

        for i in xrange(self.target_frame):
            # NN with Attention
            target_hidden_traces = self.decoder_net(target_traces)
            Attn_nn = self.attn(target_hidden_traces, target_hidden_traces, hidden_mask)

            # LSTM with Attention
            input_hidden_traces, encoder_hidden = self.encoder_net(current_batch_input_traces, encoder_hidden)
            input_attn_hidden_traces_lstm = torch.bmm(Attn_nn, input_hidden_traces)

            # concat LSTM hidden with Attention and NN hidden with Attention
            input_attn_hidden_traces = input_attn_hidden_traces_lstm

            # predict next frame traces
            regression_traces = self.regression_net(input_attn_hidden_traces, target_hidden_traces, target_traces, regression_mask)
            target_traces = regression_traces
            regression_list.append(regression_traces)

        # regression traces with mask
        regression_traces = torch.stack(regression_list, 2)

        # compute loss
        L2_square_loss = ((batch_target_traces - regression_traces) ** 2).sum() / self.max_pedestrian_num
        MSE_loss = (((batch_target_traces - regression_traces) ** 2).sum(3).sqrt().mean(2).sum(1) / batch_traces_pedestrian_num).mean()

        self.loss = L2_square_loss

        return L2_square_loss.data[0], MSE_loss.data[0], regression_traces

    def test(self):
        L2_square_loss, MSE_loss, _ = self.main_compute_step(self.test_input_traces, self.test_target_traces,
                                                             self.test_hidden_mask, self.test_regression_mask, self.test_traces_pedestrian_num)
        return MSE_loss

    def init_net(self):
        self.encoder_net = EncoderNetWithLSTM(self.max_pedestrian_num, self.input_size, self.hidden_size)
        self.decoder_net = DecoderNet(self.max_pedestrian_num, self.target_size, self.hidden_size, self.window_size)
        self.regression_net = UnfixedRegressionNet(self.max_pedestrian_num, self.target_size, self.hidden_size)
        self.attn = UnfixedAttention()

        self.encoder_net.load_state_dict(torch.load('%s/encoder_net.pkl' % self.pre_train_model_path))
        self.decoder_net.load_state_dict(torch.load('%s/decoder_net.pkl' % self.pre_train_model_path))

        self.encoder_net.cuda()
        self.decoder_net.cuda()
        self.regression_net.cuda()

    def main(self, model_name, pre_train_model_path):
        self.pre_train_model_path = pre_train_model_path
        self.init_net()

        test_loss = self.test()
        print test_loss


def main():
    # model = Trace2Trace('data/TSD/data_set_tr5.npz')
    # model.run(model_path='model/LSTM-NN', use_gpu=True)
    #
    # model = Trace2Trace('data/GC/xy_data_set.npz')
    # model.run(model_path='model/fine-tune-GC-TSD/2018-01-23-19:28:32/lr_0.01_iter_6000', use_gpu=True)

    test_on = 5
    model = UnfixedTrace2Trace_LN()
    model.load_unfixed_data(test_on=test_on)
    model.main('cross-domain-on-%d' % test_on, pre_train_model_path='model/LSTM-NN')

    pass


if __name__ == '__main__':
    main()
