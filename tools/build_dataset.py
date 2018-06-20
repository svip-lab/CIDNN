# -*- coding: utf-8 -*-
# @Time    : 2018/1/23 下午10:27
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import scipy.io as sio
import random
import visdom
import cv2
import cPickle as pickle


def create_noise_GC_dataset(src_data_path, dest_data_path, rate):
    def convert_to_noise_X(X):
        batch_size, p_num = X.shape[:2]

        for i in xrange(batch_size):
            for j in xrange(p_num):
                if is_noise():
                    X[i, j] = get_noise_trace(X[i, j])

    def get_noise_trace(trace):
        trace_vx = np.abs(trace[:-1, 0] - trace[1:, 0])
        trace_vy = np.abs(trace[:-1, 1] - trace[1:, 1])
        f_num = trace.shape[0]

        mean_vx = trace_vx.mean()
        mean_vy = trace_vy.mean()

        dx = np.random.normal(0, mean_vx / 3.0, f_num)
        dy = np.random.normal(0, mean_vy / 3.0, f_num)

        return trace + np.stack([dx, dy], axis=1)

    def is_noise():
        return random.randint(1, 100) <= rate

    data = np.load(src_data_path)
    train_X, train_Y = data['train_X'], data['train_Y']
    test_X, test_Y = data['test_X'], data['test_Y']

    src_train_X = np.copy(train_X)
    src_test_X = np.copy(test_X)

    convert_to_noise_X(train_X)
    convert_to_noise_X(test_X)

    print 'rate:', rate
    print 'train_X:'
    print 'dis**2 sum:', ((src_train_X - train_X) ** 2).sum()
    print 'dis sum:', (src_train_X - train_X).sum()

    print 'test_X:'
    print 'dis**2 sum:', ((src_test_X - test_X) ** 2).sum()
    print 'dis sum:', (src_test_X - test_X).sum()

    np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


def create_closed_GC_test_dataset(src_data_path, img_dir_path, dest_data_path, closed_frame_idx):
    def show_trace(X, Y, title):

        final_X = np.concatenate((X, Y), axis=0)
        final_Y = np.concatenate((np.ones(X.shape[0]), np.ones(Y.shape[0]) * 2), axis=0)

        vis.scatter(X=final_X, Y=final_Y, win=title,
                    opts={'xtickmin': 0, 'xtickmax': 1, 'markersize': 7, 'title': title, 'legend': ['obs', 'gt'],
                          'showlegend': True})

    data = np.load(src_data_path)
    train_X, train_Y, train_F = data['train_X'], data['train_Y'], data['train_F']
    batch_idx = np.where(train_F == closed_frame_idx)[0][0]
    p_num, f_num = train_X.shape[1:3]
    vis = visdom.Visdom(env='closed-batch')

    # X = train_X[batch_idx].reshape(-1, 2)
    # X[:, 1] = 1 - X[:, 1]
    # Y = train_Y[batch_idx].reshape(-1, 2)
    # Y[:, 1] = 1 - Y[:, 1]
    # show_trace(X, Y, title='batch-%d' % closed_frame_idx)

    # default img mode is BGR, convert to RGB
    img = cv2.imread('%s/%.6d.jpg' % (img_dir_path, closed_frame_idx * 20), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # color
    ORANGE = (255, 123, 36)
    BLUE = (0, 0, 255)

    # obs
    for i in xrange(p_num):
        for j in xrange(f_num):
            x, y = train_X[batch_idx, i, j]
            x, y = int(x * 1920), int(y * 1080)
            cv2.circle(img, (x, y), radius=3, color=BLUE, thickness=-1)

    # pred
    for i in xrange(p_num):
        for j in xrange(f_num):
            x, y = train_Y[batch_idx, i, j]
            x, y = int(x * 1920), int(y * 1080)
            cv2.circle(img, (x, y), radius=3, color=ORANGE, thickness=-1)

    vis.image(img.transpose(2, 0, 1), win='show')


def convert_npz_to_mat(src_data_path, dest_data_path):
    data = np.load(src_data_path)
    sio.savemat(dest_data_path, data)


def main():
    # for rate in [5, 10, 20]:
    #     create_noise_GC_dataset('data/GC/xy_data_set.npz', 'data/GC/noise_%d_xy_dataset.npz' % rate, rate)
    # create_closed_GC_test_dataset('data/GC/all_data_set.npz', 'data/GC_WalkingPath/Frame', 'data/GC/closed_test_dataset.npz', closed_frame_idx=140)

    # convert_npz_to_mat('data/GC/xy_data_set.npz', 'data/GC/xy_data_set.mat')
    convert_npz_to_mat('data/TSD/data_set_tr5.npz', 'data/TSD/data_set_tr5.mat')


if __name__ == '__main__':
    main()
