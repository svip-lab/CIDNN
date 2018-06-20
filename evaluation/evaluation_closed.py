# -*- coding: utf-8 -*-
# @Time    : 2018/1/24 下午4:30
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import sys
import visdom

sys.path.append("./")
from base.base_network import *
import cv2


class Trace2Trace:
    def __init__(self, data_path, img_dir_path):
        self.data_path = data_path
        self.img_dir_path = img_dir_path
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

        # show
        self.vis = None
        self.closed_frame_idx = None

    def load_data(self, closed_frame_idx):
        # load data
        data = np.load(self.data_path)
        test_F = data['train_F']
        batch_idx = np.where(test_F == closed_frame_idx)[0][0]

        test_X = data['train_X'][batch_idx:batch_idx + 1]
        test_Y = data['train_Y'][batch_idx:batch_idx + 1]

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

    def show_closed_scatter_img(self, obs_list):
        def get_trace_point(X):
            X = X[0].data.cpu().numpy()
            if obs_list == []:
                X = X.reshape(-1, 2)
                X[:, 1] = (1 - X[:, 1])
            else:
                X = X[obs_list].reshape(-1, 2)
                X[:, 1] = (1 - X[:, 1])

            X[:, 0] = X[:, 0] * 1920
            X[:, 1] = X[:, 1] * 1080
            return X

        X = get_trace_point(self.input_traces)
        Y = get_trace_point(self.ground_truth_traces)
        P = get_trace_point(self.regression_traces)

        final_X = np.concatenate((X, Y, P), axis=0)
        final_Y = np.concatenate((np.ones(X.shape[0]), np.ones(Y.shape[0]) * 2, np.ones(Y.shape[0]) * 3), axis=0)

        self.vis.scatter(X=final_X, Y=final_Y, win='scatter-%d' % self.closed_frame_idx,
                         opts={'xtickmin': 0, 'xtickmax': 1, 'markersize': 7, 'title': 'scatter-%d' % self.closed_frame_idx, 'legend': ['obs', 'gt', 'pred'],
                               'showlegend': True})

    def show_closed_raw_img(self, raw_obs_list):
        X = self.input_traces.data.cpu().numpy()[0]
        Y = self.ground_truth_traces.data.cpu().numpy()[0]
        P = self.regression_traces.data.cpu().numpy()[0]

        p_num, f_num = X.shape[:2]

        # default img mode is BGR, convert to RGB
        img = cv2.imread('%s/%.6d.jpg' % (self.img_dir_path, self.closed_frame_idx * 20), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # color
        ORANGE = (255, 123, 36)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)

        if raw_obs_list == []:
            raw_obs_list = range(p_num)

        # obs
        for i in raw_obs_list:
            for j in xrange(f_num):
                x, y = X[i, j]
                x, y = int(x * 1920), int(y * 1080)
                cv2.circle(img, (x, y), radius=3, color=BLUE, thickness=-1)

        # gt
        for i in raw_obs_list:
            for j in xrange(f_num):
                x, y = Y[i, j]
                x, y = int(x * 1920), int(y * 1080)
                cv2.circle(img, (x, y), radius=3, color=ORANGE, thickness=-1)

                if j == 0:
                    cv2.putText(img, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        # pred
        for i in raw_obs_list:
            for j in xrange(f_num):
                x, y = P[i, j]
                x, y = int(x * 1920), int(y * 1080)
                cv2.circle(img, (x, y), radius=3, color=GREEN, thickness=-1)

        cv2.imwrite('data/batch_%d.jpg' % self.closed_frame_idx, img)
        self.vis.image(img.transpose(2, 0, 1), win='show-%d' % self.closed_frame_idx)

    def run(self, model_path, closed_frame_idx, use_gpu=True, batch_size=-1, data_type='test', scatter_obs_list=[], raw_obs_list=[]):
        self.use_gpu = use_gpu
        self.closed_frame_idx = closed_frame_idx
        self.load_model(model_path)
        self.load_data(closed_frame_idx)
        self.vis = visdom.Visdom(env='closed-batch-%d' % closed_frame_idx)

        self.input_traces, self.ground_truth_traces = self.get_data(data_type, batch_size)
        L2_square_loss, MSE_loss, self.regression_traces = self.main_compute_step(self.input_traces, self.ground_truth_traces)

        self.show_closed_scatter_img(scatter_obs_list)
        self.show_closed_raw_img(raw_obs_list)


def main():
    model = Trace2Trace('data/GC/all_data_set.npz', 'data/GC_WalkingPath/Frame')
    model.run(model_path='model/LSTM-NN', closed_frame_idx=295, use_gpu=True, scatter_obs_list=[1, 11], raw_obs_list=[1, 11])


if __name__ == '__main__':
    main()
