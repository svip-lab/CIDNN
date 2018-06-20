# -*- coding: utf-8 -*-
# @Time    : 2017/9/11 下午8:08
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn


import numpy as np
import cPickle as pickle


class ConstV:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pedestrian_num = 20
        self.obs_frame = 5
        self.predict_frame = 5

    def load_data(self):
        data = np.load(self.data_path)
        self.test_input_traces, self.test_target_traces = data['test_X'], data['test_Y']
        self.batch_size = self.test_input_traces.shape[0]

    def load_unfixed_data(self, test_on):
        # load test data
        with open('data/xy_8_12_train_set_0%d.data' % test_on, 'r') as f:
            data_set = pickle.load(f)
            self.test_input_traces = data_set['train_X']
            self.test_target_traces = data_set['train_Y']

        self.batch_size = len(self.test_input_traces)
        self.test_traces_pedestrian_num = [len(input_trace) for input_trace in self.test_input_traces]

    def main(self):
        # return (1,5)
        def predict(X, Y, Z):
            A = np.array([[self.obs_frame, X.sum()],
                          [X.sum(), (X ** 2).sum()]])
            b = np.array([Y.sum(), (X * Y).sum()])

            params = np.linalg.solve(A, b).reshape(1, -1)
            Z = Z.reshape(-1, 1)
            Z = np.concatenate((np.ones_like(Z), Z), axis=1)

            return (Z * params).sum(axis=1).reshape(1, -1)

        self.load_data()

        predict_T = np.array([6, 7, 8, 9, 10])
        obs_T = np.array([1, 2, 3, 4, 5])
        predict_traces = [[0 for _ in xrange(self.pedestrian_num)] for _ in xrange(self.batch_size)]

        for i in xrange(self.batch_size):
            for j in xrange(self.pedestrian_num):
                obs_X = self.test_input_traces[i, j, :, 0]
                obs_Y = self.test_input_traces[i, j, :, 1]

                predict_X = predict(obs_T, obs_X, predict_T)
                predict_Y = predict(obs_T, obs_Y, predict_T)

                # size (5,2)
                predict_trace = np.concatenate([predict_X, predict_Y], axis=0).transpose()
                predict_traces[i][j] = predict_trace

        predict_traces = np.array(predict_traces)

        MSE_loss = np.sqrt(((predict_traces - self.test_target_traces) ** 2).sum(axis=3)).mean()
        print MSE_loss

    def unfixed_main(self, test_on):
        # return (1,5)
        self.load_unfixed_data(test_on)
        self.obs_frame = 8
        self.predict_frame = 12

        def predict(X, Y, Z):
            A = np.array([[self.obs_frame, X.sum()],
                          [X.sum(), (X ** 2).sum()]])
            b = np.array([Y.sum(), (X * Y).sum()])

            params = np.linalg.solve(A, b).reshape(1, -1)
            Z = Z.reshape(-1, 1)
            Z = np.concatenate((np.ones_like(Z), Z), axis=1)

            return (Z * params).sum(axis=1).reshape(1, -1)

        predict_T = np.array(range(9, 21))
        obs_T = np.array(range(1, 9))
        MSE_loss = 0.0
        count = 0

        for i in xrange(self.batch_size):
            for j in xrange(self.test_traces_pedestrian_num[i]):
                obs_X = np.array(self.test_input_traces[i][j])[:, 0]
                obs_Y = np.array(self.test_input_traces[i][j])[:, 1]

                predict_X = predict(obs_T, obs_X, predict_T)
                predict_Y = predict(obs_T, obs_Y, predict_T)

                # size (5,2)
                predict_trace = np.concatenate([predict_X, predict_Y], axis=0).transpose()
                target_traces = self.test_target_traces[i][j]

                MSE_loss += np.sqrt(((predict_trace - target_traces) ** 2).sum(axis=1)).mean()
                count += 1

        print MSE_loss / count


class ConstA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pedestrian_num = 20
        self.obs_frame = 5
        self.predict_frame = 5

    def load_data(self):
        data = np.load(self.data_path)
        self.test_input_traces, self.test_target_traces = data['test_X'], data['test_Y']
        self.batch_size = self.test_input_traces.shape[0]

    def main(self):
        # return (1,5)
        def predict(X, Y, Z):
            A = np.array([[self.obs_frame, X.sum(), (X ** 2).sum()],
                          [X.sum(), (X ** 2).sum(), (X ** 3).sum()],
                          [(X ** 2).sum(), (X ** 3).sum(), (X ** 4).sum()]])
            b = np.array([Y.sum(), (X * Y).sum(), ((X ** 2) * Y).sum()])

            params = np.linalg.solve(A, b).reshape(1, -1)
            Z = Z.reshape(-1, 1)
            Z = np.concatenate((np.ones_like(Z), Z, Z ** 2), axis=1)

            return (Z * params).sum(axis=1).reshape(1, -1)

        self.load_data()

        predict_T = np.array([6, 7, 8, 9, 10])
        obs_T = np.array([1, 2, 3, 4, 5])
        predict_traces = [[0 for _ in xrange(self.pedestrian_num)] for _ in xrange(self.batch_size)]

        for i in xrange(self.batch_size):
            for j in xrange(self.pedestrian_num):
                obs_X = self.test_input_traces[i, j, :, 0]
                obs_Y = self.test_input_traces[i, j, :, 1]

                predict_X = predict(obs_T, obs_X, predict_T)
                predict_Y = predict(obs_T, obs_Y, predict_T)

                # size (5,2)
                predict_trace = np.concatenate([predict_X, predict_Y], axis=0).transpose()
                predict_traces[i][j] = predict_trace

        predict_traces = np.array(predict_traces)

        MSE_loss = np.sqrt(((predict_traces - self.test_target_traces) ** 2).sum(axis=3)).mean()
        print MSE_loss


class NaiveConstV:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pedestrian_num = 20
        self.obs_frame = 5
        self.predict_frame = 5

    def load_data(self):
        data = np.load(self.data_path)
        self.test_input_traces, self.test_target_traces = data['test_X'], data['test_Y']
        self.batch_size = self.test_input_traces.shape[0]

    def main(self):
        # return (1,5)
        def predict(X, Y, Z):
            v = (Y[len(X) - 1] - Y[0]) / (len(X) - 1)
            return np.array([Y[len(X) - 1] + (i + 1) * v for i in xrange(len(Z))]).reshape(1, -1)

        self.load_data()

        predict_T = np.array([6, 7, 8, 9, 10])
        obs_T = np.array([1, 2, 3, 4, 5])
        predict_traces = [[0 for _ in xrange(self.pedestrian_num)] for _ in xrange(self.batch_size)]

        for i in xrange(self.batch_size):
            for j in xrange(self.pedestrian_num):
                obs_X = self.test_input_traces[i, j, :, 0]
                obs_Y = self.test_input_traces[i, j, :, 1]

                predict_X = predict(obs_T, obs_X, predict_T)
                predict_Y = predict(obs_T, obs_Y, predict_T)

                # size (5,2)
                predict_trace = np.concatenate([predict_X, predict_Y], axis=0).transpose()
                predict_traces[i][j] = predict_trace

        predict_traces = np.array(predict_traces)

        MSE_loss = np.sqrt(((predict_traces - self.test_target_traces) ** 2).sum(axis=3)).mean()
        print MSE_loss


class NaiveConstA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pedestrian_num = 20
        self.obs_frame = 5
        self.predict_frame = 5

    def load_data(self):
        data = np.load(self.data_path)
        self.test_input_traces, self.test_target_traces = data['test_X'], data['test_Y']
        self.batch_size = self.test_input_traces.shape[0]

    def main(self):
        # return (1,5)
        def predict(X, Y, Z):
            obs_t = len(X) - 1
            dst = Y[obs_t] - Y[obs_t - 1]
            dst_1 = Y[obs_t - 1] - Y[obs_t - 2]

            a = (dst - dst_1) / (obs_t ** 2)
            vt_1 = (Y[obs_t] - Y[obs_t - 2]) / 2
            vt = vt_1 + a

            return np.array([Y[obs_t] + vt * (i + 1) + 0.5 * a * (i + 1) ** 2 for i in xrange(len(Z))]).reshape(1, -1)

        self.load_data()

        predict_T = np.array([6, 7, 8, 9, 10])
        obs_T = np.array([1, 2, 3, 4, 5])
        predict_traces = [[0 for _ in xrange(self.pedestrian_num)] for _ in xrange(self.batch_size)]

        for i in xrange(self.batch_size):
            for j in xrange(self.pedestrian_num):
                obs_X = self.test_input_traces[i, j, :, 0]
                obs_Y = self.test_input_traces[i, j, :, 1]

                predict_X = predict(obs_T, obs_X, predict_T)
                predict_Y = predict(obs_T, obs_Y, predict_T)

                # size (5,2)
                predict_trace = np.concatenate([predict_X, predict_Y], axis=0).transpose()
                predict_traces[i][j] = predict_trace

        predict_traces = np.array(predict_traces)

        MSE_loss = np.sqrt(((predict_traces - self.test_target_traces) ** 2).sum(axis=3)).mean()
        print MSE_loss


def main():
    data_path = 'data/GC/xy_very_hard_dataset.npz'
    # data_path = 'data/Tokyo/data_set_tr5.npz'

    print 'const_v:'
    const_v = ConstV(data_path)
    const_v.main()
    print 'naive_const_v:'
    const_v = NaiveConstV(data_path)
    const_v.main()

    print 'const_a'
    const_a = ConstA(data_path)
    const_a.main()
    print 'naive_const_a'
    const_a = NaiveConstA(data_path)
    const_a.main()


if __name__ == '__main__':
    main()
