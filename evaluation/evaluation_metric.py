# -*- coding: utf-8 -*-
# @Time    : 2018/1/20 下午2:00
# @Author  : Zhixin Piao
# @Email   : piaozhx@shanghaitech.edu.cn


import sys
import visdom

sys.path.append("./")
from base.base_network import *
import cv2


MEAN_A = 0.00927538


def compute_a_X(X):
    # assume that time interval equals to 1
    v_X = []
    frame_num = X.shape[2]
    for i in xrange(frame_num - 1):
        v_X.append(X[:, :, i + 1, :] - X[:, :, i, :])

    a_X = []
    for i in xrange(frame_num - 2):
        a_X.append(v_X[i + 1] - v_X[i])

    # (frame - 2，B, p_num, 2)
    a_X = np.array(a_X)

    # (frame -2, B, p_num)  which each element present norm of a
    a_X = np.sqrt(np.sum(a_X ** 2, axis=3))

    # (B, p_num)
    a_X = np.max(a_X, axis=0)

    return a_X


def compute_v_X(X):
    # assume that time interval equals to 1
    v_X = []
    frame_num = X.shape[2]
    for i in xrange(frame_num - 1):
        v_X.append(X[:, :, i + 1, :] - X[:, :, i, :])

    # (B, p, f-1, 2)
    v_X = np.stack(v_X, axis=2)
    return v_X


def compute_ANE(obs, pred, gt, rate):
    '''
    input type: torch.Tensor
    obs: (batch_size, p_num, f_num, 2)
    pred: (batch_size, p_num, f_num, 2)
    gt: (batch_size, p_num, f_num, 2)
    rate: hard rate, int
    '''

    a_X = compute_a_X(obs.data.cpu().numpy())
    batch_size, p_num, f_num = obs.shape[:3]

    non_linear_total_error = 0.0
    non_linear_count = 0.0

    linear_total_error = 0.0
    linear_count = 0.0

    for i in xrange(batch_size):
        for j in xrange(p_num):
            err = ((pred[i, j] - gt[i, j]) ** 2).sum(1).sqrt().mean()
            if a_X[i, j] > rate * MEAN_A:
                non_linear_total_error += err
                non_linear_count += 1
            else:
                linear_total_error += err
                linear_count += 1

    ANE = (non_linear_total_error / non_linear_count).data[0]
    C_ANE = (linear_total_error / linear_count).data[0]

    return ANE, C_ANE


def compute_FDE(pred, gt):
    '''
    input type: torch.Tensor
    pred: (batch_size, p_num, f_num, 2)
    gt: (batch_size, p_num, f_num, 2)
    '''
    err = ((pred[:, :, -1, :] - gt[:, :, -1, :]) ** 2).sum(2).sqrt().mean()
    return err.data[0]


def compute_ACDE(obs, pred, gt, curve_angle_threshold):
    '''
    input type: torch.Tensor
    obs: (batch_size, p_num, f_num, 2)
    pred: (batch_size, p_num, f_num, 2)
    gt: (batch_size, p_num, f_num, 2)
    '''
    v_X = compute_v_X(obs.data.cpu().numpy())
    batch_size, p_num, f_num = obs.shape[:3]

    cosine_X = []
    for i in xrange(f_num - 2):
        dot_prod = (v_X[:, :, i + 1] * v_X[:, :, i]).sum(2)
        norm_v1 = np.sqrt((v_X[:, :, i + 1] ** 2).sum(2))
        norm_v0 = np.sqrt((v_X[:, :, i] ** 2).sum(2))

        cosine = dot_prod / (norm_v0 * norm_v1)
        cosine[np.isnan(cosine)] = 1.0

        cosine_X.append(cosine)

    # (B, p_num, f-2)
    cosine_X = np.stack(cosine_X, axis=2)
    # (B, p_num)
    avg_cosine_X = cosine_X.mean(axis=2)

    curve_cosine_threshold = np.cos(curve_angle_threshold / 360.0 * 2 * np.pi)

    curve_total_error = 0.0
    curve_count = 0.0
    non_curve_total_error = 0.0
    non_curve_count = 0.0

    batch_count = 0.0
    for i in xrange(batch_size):
        batch_flag = False
        for j in xrange(p_num):
            err = ((pred[i, j] - gt[i, j]) ** 2).sum(1).sqrt().mean()
            if avg_cosine_X[i, j] < curve_cosine_threshold:
                curve_total_error += err
                curve_count += 1
                if not batch_flag:
                    batch_count += 1
                    batch_flag = True
            else:
                non_curve_total_error += err
                non_curve_count += 1

    ACDE = (curve_total_error / curve_count).data[0]
    C_ACDE = (non_curve_total_error / non_curve_count).data[0]

    print 'curve situation pedestrian count:', curve_count
    print 'batch percent:', batch_count / batch_size
    print 'person percent:', curve_count / (batch_size * p_num)
    return ACDE, C_ACDE


def show_hard_trace():
    vis = visdom.Visdom(env='hard-')

    data = np.load('data/GC/xy_hard_dataset.npz')
    test_X = data['test_X']
    test_Y = data['test_Y']

    a_X = compute_a_X(test_X)

    batch_size, p_num, f_num = test_X.shape[:3]
    rd_idx_list = random.sample(range(batch_size), 100)

    for i in rd_idx_list:
        for j in xrange(p_num):
            if a_X[i, j] > 3 * MEAN_A:
                X = test_X[i, j].reshape(-1, 2)
                Y = test_Y[i, j].reshape(-1, 2)

                final_X = np.concatenate((X, Y), axis=0)
                final_Y = np.concatenate((np.ones(X.shape[0]), np.ones(Y.shape[0]) * 2), axis=0)

                vis.scatter(X=final_X, Y=final_Y, win='hard-%.3d' % i,
                            opts={'xtickmin': 0, 'xtickmax': 1, 'markersize': 7, 'title': 'hard-%.3d' % i, 'legend': ['obs', 'pred'],
                                  'showlegend': True})


def show_very_hard_trace():
    vis = visdom.Visdom(env='very_hard')

    data = np.load('data/GC/xy_very_hard_dataset.npz')
    test_X = data['test_X']
    test_Y = data['test_Y']

    a_X = compute_a_X(test_X)

    batch_size, p_num, f_num = test_X.shape[:3]
    rd_idx_list = random.sample(range(batch_size), 100)

    for i in rd_idx_list:
        for j in xrange(p_num):
            if a_X[i, j] > 5 * MEAN_A:
                X = test_X[i, j].reshape(-1, 2)
                Y = test_Y[i, j].reshape(-1, 2)

                final_X = np.concatenate((X, Y), axis=0)
                final_Y = np.concatenate((np.ones(X.shape[0]), np.ones(Y.shape[0]) * 2), axis=0)

                vis.scatter(X=final_X, Y=final_Y, win='very_hard-%.3d' % i,
                            opts={'xtickmin': 0, 'xtickmax': 1, 'markersize': 7, 'title': 'very_hard-%.3d' % i, 'legend': ['obs', 'pred'],
                                  'showlegend': True})


def show_curve_trace():
    vis = visdom.Visdom(env='curve_trace')

    data = np.load('data/GC/xy_data_set.npz')
    test_X = data['test_X']
    test_Y = data['test_Y']

    curve_angle_threshold = 60

    v_X = compute_v_X(test_X)
    batch_size, p_num, f_num = test_X.shape[:3]

    cosine_X = []
    for i in xrange(f_num - 2):
        dot_prod = (v_X[:, :, i + 1] * v_X[:, :, i]).sum(2)
        norm_v1 = np.sqrt((v_X[:, :, i + 1] ** 2).sum(2))
        norm_v0 = np.sqrt((v_X[:, :, i] ** 2).sum(2))

        cosine = dot_prod / (norm_v0 * norm_v1)
        cosine[np.isnan(cosine)] = 1.0

        cosine_X.append(cosine)

    # (B, p_num, f-2)
    cosine_X = np.stack(cosine_X, axis=2)
    # (B, p_num)
    avg_cosine_X = cosine_X.mean(axis=2)

    curve_cosine_threshold = np.cos(curve_angle_threshold / 360.0 * 2 * np.pi)

    batch_count = 0
    for i in xrange(batch_size):
        batch_flag = False
        for j in xrange(p_num):
            if avg_cosine_X[i, j] < curve_cosine_threshold:
                X = test_X[i, j].reshape(-1, 2)
                Y = test_Y[i, j].reshape(-1, 2)

                final_X = np.concatenate((X, Y), axis=0)
                final_Y = np.concatenate((np.ones(X.shape[0]), np.ones(Y.shape[0]) * 2), axis=0)

                vis.scatter(X=final_X, Y=final_Y, win='curve-%.3d' % i,
                            opts={'xtickmin': 0, 'xtickmax': 1, 'markersize': 7, 'title': 'curve-%.3d' % i, 'legend': ['obs', 'pred'],
                                  'showlegend': True})

                batch_flag = True

        if batch_flag:
            batch_count += 1
        if batch_count == 50:
            break


def show_noise_trace(rate):
    vis = visdom.Visdom(env='noise-%d' % rate)
    vis.close()

    data = np.load('data/GC/noise_%d_xy_dataset.npz' % rate)
    test_X = data['test_X']
    test_Y = data['test_Y']

    batch_size, p_num, f_num = test_X.shape[:3]
    rd_idx_list = random.sample(range(batch_size), 50)

    for i in rd_idx_list:
        X = test_X[i].reshape(-1, 2)
        Y = test_Y[i].reshape(-1, 2)

        final_X = np.concatenate((X, Y), axis=0)
        final_Y = np.concatenate((np.ones(X.shape[0]), np.ones(Y.shape[0]) * 2), axis=0)

        vis.scatter(X=final_X, Y=final_Y, win='noise-%.3d' % i,
                    opts={'xtickmin': 0, 'xtickmax': 1, 'markersize': 7, 'title': 'noise-%.3d' % i, 'legend': ['obs', 'pred'],
                          'showlegend': True})


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
        test_X, test_Y = data['test_X'] + 0.5, data['test_Y'] + 0.5

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
        print 'start compute error!!'

        hard_ANE, hard_C_ANE = compute_ANE(self.input_traces, self.regression_traces, self.ground_truth_traces, 3)
        # very_hard_ANE = compute_ANE(self.input_traces, self.regression_traces, self.ground_truth_traces, 5)
        FDE = compute_FDE(self.regression_traces, self.ground_truth_traces)
        ACDE, C_ACDE = compute_ACDE(self.input_traces, self.regression_traces, self.ground_truth_traces, 60)

        print ' L2_square_loss:%s,  MSE_loss: %s' % (L2_square_loss, MSE_loss)
        print 'hard_ANE: ', hard_ANE, 'hard_C_ANE: ', hard_C_ANE
        # print 'very_hard_ANE: ', very_hard_ANE
        print 'FDE: ', FDE
        print 'ACDE: ', ACDE, 'C_ACDE: ', C_ACDE


def main():
    # model = Trace2Trace('data/GC/xy_very_hard_dataset.npz')
    # model = Trace2Trace('data/GC/xy_data_set.npz')
    # model.run(model_path='model/LSTM-NN', use_gpu=True)

    model = Trace2Trace('data/TSD/data_set_tr5.npz')
    model.run(model_path='model/fine-tune-GC-TSD/2018-01-23-19:28:32/lr_0.01_iter_6000', use_gpu=True)

    # show_hard_trace()
    # show_very_hard_trace()
    # show_curve_trace()
    # for rate in [5, 10, 20]:
    #     show_noise_trace(rate)
    pass


if __name__ == '__main__':
    main()
