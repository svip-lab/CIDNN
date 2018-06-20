# -*- coding: utf-8 -*-
# @Time    : 2017/7/12 下午9:17
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import scipy.io as sio
import cPickle as pickle
import numpy as np
import random
import json
import visdom
import csv


def create_data_set(src_data_path, dest_data_path):
    data = sio.loadmat(src_data_path)
    frame_num = data['Time_indexed_infor_test'].shape[0]

    f_data = [[] for _ in xrange(frame_num)]
    p_data = [None] * 100000

    for i in xrange(frame_num):
        count = data['Time_indexed_infor_test'][i][0]['pedestrain_count'][0][0] - 1
        if count == 0:
            continue

        pedestrian = data['Time_indexed_infor_test'][i][0]['pedestrain'][0][0]

        for x, y, pid in pedestrian:
            pid = int(pid) - 1
            if p_data[pid] is None:
                p_data[pid] = {}
            p_data[pid][i] = (x, y)
            f_data[i].append(pid)

    with open(dest_data_path, 'w') as f:
        pickle.dump({'frame_data': f_data, 'pedestrian_data': p_data}, f)


def create_train_test_set():
    def get_samples_from_frame(f_i):
        if f_i - step < 0 or f_i + step + 1 > frame_num:
            return [], []

        p_set = set(f_data[f_i - step])
        for i in xrange(f_i - step + 1, f_i + step + 1):
            p_set &= set(f_data[i])

        p_set = list(p_set)
        samples_X = []
        samples_Y = []
        p_num = len(p_set)
        for si in xrange(p_num / sample_size):
            sample_X = []
            sample_Y = []
            for i in xrange(si * sample_size, (si + 1) * sample_size):
                p = p_data[p_set[i]]
                sx, sy = p[f_i - step][0], p[f_i - step][1]

                sample_X.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i - step + 1, f_i + 1)])
                sample_Y.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i + 1, f_i + step + 1)])
            samples_X.append(sample_X)
            samples_Y.append(sample_Y)

        return samples_X, samples_Y

    def collect_sample(f_idx):
        ret_X = []
        ret_Y = []
        for i in f_idx:
            samples_X, samples_Y = get_samples_from_frame(i)
            ret_X += samples_X
            ret_Y += samples_Y

        return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

    # hyper-parameter
    frame_num = 5000
    train_rate = 0.9
    step = 5
    sample_size = 20

    # raw data
    with open('data/data_set.data', 'r') as f:
        data = pickle.load(f)
    f_data = data['frame_data'][:frame_num]
    p_data = data['pedestrian_data']

    # create random center frame idx
    # random_idx = random.sample(range(frame_num), frame_num)
    random_idx = range(frame_num)
    train_idx = random_idx[:int(frame_num * train_rate)]
    test_idx = random_idx[int(frame_num * train_rate):]

    # train set, test set
    train_X, train_Y = collect_sample(train_idx)
    test_X, test_Y = collect_sample(test_idx)

    print train_X.dtype
    print train_X.shape, train_Y.shape
    print test_X.shape, test_Y.shape

    np.savez('data/data_set_2.npz', train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


def create_xy_train_test_set(src_data_path, dest_data_path):
    def get_samples_from_frame(f_i):
        if f_i - step < 0 or f_i + step > frame_num:
            return [], []

        p_set = set(f_data[f_i - step])
        for i in xrange(f_i - step + 1, f_i + step):
            p_set &= set(f_data[i])

        p_set = list(p_set)
        samples_X = []
        samples_Y = []
        p_num = len(p_set)
        for si in xrange(p_num / sample_size):
            sample_X = []
            sample_Y = []
            for i in xrange(si * sample_size, (si + 1) * sample_size):
                p = p_data[p_set[i]]
                sample_X.append([[p[j][0], p[j][1]] for j in xrange(f_i - step, f_i)])
                sample_Y.append([[p[j][0], p[j][1]] for j in xrange(f_i, f_i + step)])
            samples_X.append(sample_X)
            samples_Y.append(sample_Y)

        return samples_X, samples_Y

    def collect_sample(f_idx):
        ret_X = []
        ret_Y = []
        for i in f_idx:
            samples_X, samples_Y = get_samples_from_frame(i)
            ret_X += samples_X
            ret_Y += samples_Y

        return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

    # hyper-parameter
    frame_num = 5000
    train_rate = 0.9
    step = 5
    sample_size = 20

    # raw data
    with open(src_data_path, 'r') as f:
        data = pickle.load(f)
    f_data = data['frame_data'][:frame_num]
    p_data = data['pedestrian_data']

    # create random center frame idx
    random_idx = random.sample(range(frame_num), frame_num)
    train_idx = random_idx[:int(frame_num * train_rate)]
    test_idx = random_idx[int(frame_num * train_rate):]

    # train set, test set
    train_X, train_Y = collect_sample(train_idx)
    test_X, test_Y = collect_sample(test_idx)

    print train_X.dtype
    print 'train_X: %s, train_Y: %s' % (train_X.shape, train_Y.shape)
    print 'test_X: %s, test_Y: %s' % (test_X.shape, test_Y.shape)

    np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


def create_xy_train_set(src_data_path, dest_data_path):
    def get_samples_from_frame(f_i):
        if f_i - obs_step < 0 or f_i + pred_step > frame_num:
            return [], []

        p_set = set(f_data[f_i - obs_step])
        for i in xrange(f_i - obs_step + 1, f_i + pred_step):
            p_set &= set(f_data[i])

        p_set = list(p_set)
        p_num = len(p_set)

        sample_X = []
        sample_Y = []
        for i in xrange(p_num):
            p = p_data[p_set[i]]
            sample_X.append([[p[j][0], p[j][1]] for j in xrange(f_i - obs_step, f_i)])
            sample_Y.append([[p[j][0], p[j][1]] for j in xrange(f_i, f_i + pred_step)])

        return sample_X, sample_Y

    def collect_sample():
        ret_X = []
        ret_Y = []
        for i in xrange(frame_num):
            sample_X, sample_Y = get_samples_from_frame(i)
            if sample_X != []:
                ret_X.append(sample_X)
                ret_Y.append(sample_Y)

        return ret_X, ret_Y

    # hyper-parameter
    obs_step = 8
    pred_step = 12

    # raw data
    with open(src_data_path, 'r') as f:
        data = pickle.load(f)

    f_data = data['frame_data']
    p_data = data['pedestrian_data']
    frame_num = len(f_data)

    # train set, test set
    train_X, train_Y = collect_sample()
    print max([len(x) for x in train_X])
    print min([len(x) for x in train_X])
    print [len(x) for x in train_X]

    with open(dest_data_path, 'w') as f:
        pickle.dump({'train_X': train_X, 'train_Y': train_Y}, f)


def create_multiple_frame_xy_train_test_set(src_data_path, dest_data_path, min_step, max_step):
    def get_samples_from_frame(f_i):
        samples_X = [[] for _ in xrange(max_step - min_step + 1)]
        samples_Y = [[] for _ in xrange(max_step - min_step + 1)]

        if f_i - obs_step < 0 or f_i + max_step > frame_num:
            return samples_X, samples_Y

        p_set = set(f_data[f_i - obs_step])
        for i in xrange(f_i - obs_step + 1, f_i + max_step):
            p_set &= set(f_data[i])

        p_set = list(p_set)

        p_num = len(p_set)

        for si in xrange(p_num / sample_size):
            for step in xrange(min_step, max_step + 1):
                # get single sample
                sample_X = []
                sample_Y = []
                for i in xrange(si * sample_size, (si + 1) * sample_size):
                    p = p_data[p_set[i]]
                    sample_X.append([[p[j][0], p[j][1]] for j in xrange(f_i - obs_step, f_i)])
                    sample_Y.append([[p[j][0], p[j][1]] for j in xrange(f_i, f_i + step)])

                samples_X[step - min_step].append(sample_X)
                samples_Y[step - min_step].append(sample_Y)

        return samples_X, samples_Y

    def collect_sample(f_idx):
        ret_X = [[] for _ in xrange(max_step - min_step + 1)]
        ret_Y = [[] for _ in xrange(max_step - min_step + 1)]
        for i in f_idx:
            samples_X, samples_Y = get_samples_from_frame(i)
            for step in xrange(min_step, max_step + 1):
                ret_X[step - min_step] += samples_X[step - min_step]
                ret_Y[step - min_step] += samples_Y[step - min_step]

        return ret_X, ret_Y

        # return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

    # hyper-parameter
    frame_num = 5000
    obs_step = 5
    train_rate = 0.9
    sample_size = 20

    # raw data
    with open(src_data_path, 'r') as f:
        data = pickle.load(f)
    f_data = data['frame_data'][:frame_num]
    p_data = data['pedestrian_data']

    # create random center frame idx
    random_idx = random.sample(range(frame_num), frame_num)
    train_idx = random_idx[:int(frame_num * train_rate)]
    test_idx = random_idx[int(frame_num * train_rate):]

    # train set, test set
    train_X, train_Y = collect_sample(train_idx)
    test_X, test_Y = collect_sample(test_idx)

    print len(train_X[0]), len(test_X[0])

    with open(dest_data_path, 'w') as f:
        pickle.dump({'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y}, f)


def create_multiple_frame_from_GC(src_data_path, dest_data_path, min_step, max_step):
    def get_samples_from_frame(f_i):
        samples_X = [[] for _ in xrange(max_step - min_step + 1)]
        samples_Y = [[] for _ in xrange(max_step - min_step + 1)]

        if f_i - obs_step < 0 or f_i + max_step > frame_num:
            return samples_X, samples_Y

        p_set = set(f_data[f_i - obs_step])
        for i in xrange(f_i - obs_step + 1, f_i + max_step):
            p_set &= set(f_data[i])

        p_set = list(p_set)

        p_num = len(p_set)

        for si in xrange(p_num / sample_size):
            for step in xrange(min_step, max_step + 1):
                # get single sample
                sample_X = []
                sample_Y = []
                for i in xrange(si * sample_size, (si + 1) * sample_size):
                    p = p_data[p_set[i]]
                    sample_X.append([[p[j][0], p[j][1]] for j in xrange(f_i - obs_step, f_i)])
                    sample_Y.append([[p[j][0], p[j][1]] for j in xrange(f_i, f_i + step)])

                samples_X[step - min_step].append(sample_X)
                samples_Y[step - min_step].append(sample_Y)

        return samples_X, samples_Y

    def collect_sample(f_idx):
        ret_X = [[] for _ in xrange(max_step - min_step + 1)]
        ret_Y = [[] for _ in xrange(max_step - min_step + 1)]
        for i in f_idx:
            samples_X, samples_Y = get_samples_from_frame(i)
            for step in xrange(min_step, max_step + 1):
                ret_X[step - min_step] += samples_X[step - min_step]
                ret_Y[step - min_step] += samples_Y[step - min_step]

        return ret_X, ret_Y

        # return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

    # hyper-parameter
    frame_num = 5000
    obs_step = 5
    train_rate = 0.9
    sample_size = 20

    # raw data
    with open(src_data_path, 'r') as f:
        data = pickle.load(f)
    f_data = data['frame_data'][:frame_num]
    p_data = data['pedestrian_data']

    # create random center frame idx
    random_idx = random.sample(range(frame_num), frame_num)
    train_idx = random_idx[:int(frame_num * train_rate)]
    test_idx = random_idx[int(frame_num * train_rate):]

    # train set, test set
    train_X, train_Y = collect_sample(train_idx)
    test_X, test_Y = collect_sample(test_idx)

    print len(train_X[0]), len(test_X[0])

    with open(dest_data_path, 'w') as f:
        pickle.dump({'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y}, f)

        # np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)


def create_xy_train_test_set_with_frame(src_data_path, dest_data_path):
    def get_samples_from_frame(f_i):
        if f_i - step < 0 or f_i + step > frame_num:
            return [], []

        p_set = set(f_data[f_i - step])
        for i in xrange(f_i - step + 1, f_i + step):
            p_set &= set(f_data[i])

        p_set = list(p_set)
        samples_X = []
        samples_Y = []
        p_num = len(p_set)
        for si in xrange(p_num / sample_size):
            sample_X = []
            sample_Y = []
            for i in xrange(si * sample_size, (si + 1) * sample_size):
                p = p_data[p_set[i]]
                sample_X.append([[p[j][0], p[j][1]] for j in xrange(f_i - step, f_i)])
                sample_Y.append([[p[j][0], p[j][1]] for j in xrange(f_i, f_i + step)])
            samples_X.append(sample_X)
            samples_Y.append(sample_Y)

        return samples_X, samples_Y

    def collect_sample(f_idx):
        ret_X = []
        ret_Y = []
        ret_F = []
        for i in f_idx:
            samples_X, samples_Y = get_samples_from_frame(i)
            ret_X += samples_X
            ret_Y += samples_Y
            ret_F += [i] * len(samples_X)

        return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32), np.array(ret_F)

    # hyper-parameter
    frame_num = 5000
    step = 5
    sample_size = 20

    # raw data
    with open(src_data_path, 'r') as f:
        data = pickle.load(f)
    f_data = data['frame_data'][:frame_num]
    p_data = data['pedestrian_data']

    # create random center frame idx
    train_idx = range(frame_num)

    # train set, test set
    train_X, train_Y, train_F = collect_sample(train_idx)
    print train_X.dtype
    print 'train_X: %s, train_Y: %s, train_F: %s' % (train_X.shape, train_Y.shape, train_F.shape)

    np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, train_F=train_F)


def create_TSD_dataset():
    # change .mat format to .data format for clear data usage.
    def mat2data(src_data_path, dest_data_path, scale):
        data = sio.loadmat(src_data_path)
        p_num = data['trks'].shape[1]

        f_data = [[] for _ in xrange(100000)]
        p_data = [None] * 100000

        max_f = -1
        for i in xrange(p_num):
            x_pts = list(data['trks'][0, i]['x'].reshape(-1))
            y_pts = list(data['trks'][0, i]['y'].reshape(-1))
            frames = list(data['trks'][0, i]['t'].reshape(-1))

            p_data[i] = {}
            for x, y, f in zip(x_pts, y_pts, frames):
                f = int(f) - 1
                f_20 = f / scale

                if f_20 not in p_data[i]:
                    max_f = max(max_f, f_20)
                    p_data[i][f_20] = (x / 720.0, y / 480.0)
                    f_data[f_20].append(i)

        p_data = filter(lambda x: x != None and x != {}, p_data)
        f_data = f_data[:max_f + 1]

        with open(dest_data_path, 'w') as f:
            pickle.dump({'frame_data': f_data, 'pedestrian_data': p_data}, f)

    # change .data format to .npz format and do same sampling for train and test data set
    def data2npz(src_data_path, dest_data_path):
        def get_samples_from_frame(f_i):
            if f_i - step < 0 or f_i + step + 1 > frame_num:
                return [], []

            p_set = set(f_data[f_i - step])
            for i in xrange(f_i - step + 1, f_i + step + 1):
                p_set &= set(f_data[i])

            p_set = list(p_set)
            samples_X = []
            samples_Y = []
            p_num = len(p_set)
            for si in xrange(p_num / sample_size):
                sample_X = []
                sample_Y = []
                for i in xrange(si * sample_size, (si + 1) * sample_size):
                    p = p_data[p_set[i]]
                    sx, sy = p[f_i - step][0], p[f_i - step][1]

                    sample_X.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i - step + 1, f_i + 1)])
                    sample_Y.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i + 1, f_i + step + 1)])
                samples_X.append(sample_X)
                samples_Y.append(sample_Y)

            return samples_X, samples_Y

        def collect_sample(f_idx):
            ret_X = []
            ret_Y = []
            for i in f_idx:
                samples_X, samples_Y = get_samples_from_frame(i)
                ret_X += samples_X
                ret_Y += samples_Y

            return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

        # hyper-parameter
        max_frame_num = 5000
        train_rate = 0.5
        step = 5
        sample_size = 20

        # raw data
        with open(src_data_path, 'r') as f:
            data = pickle.load(f)
        frame_num = min(len(data['frame_data']), max_frame_num)
        f_data = data['frame_data'][:frame_num]
        p_data = data['pedestrian_data']

        # create random center frame idx
        # random_idx = random.sample(range(frame_num), frame_num)
        random_idx = range(frame_num)
        train_idx = random_idx[:int(frame_num * train_rate)]
        test_idx = random_idx[int(frame_num * train_rate):]

        # train set, test set
        train_X, train_Y = collect_sample(train_idx)
        test_X, test_Y = collect_sample(test_idx)

        print train_X.dtype
        print train_X.shape, train_Y.shape
        print test_X.shape, test_Y.shape

        np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

    # mat2data('data/TSD/matlab_tracklets.mat', 'data/TSD/data_set.data', scale=20)
    data2npz('data/TSD/data_set.data', 'data/TSD/data_set_tr5.npz')


def statistic_pedestrian_exist_frame_rate():
    with open('data/GC/data_set.data', 'r') as f:
        data = pickle.load(f)
    f_data = data['frame_data']
    p_data = data['pedestrian_data']
    p_data = filter(lambda x: x != None, p_data)

    frame_list = []

    for p in p_data:
        f_list = p.keys()
        ideal_sum = (max(f_list) + min(f_list)) * len(f_list) / 2
        real_sum = sum(f_list)

        if ideal_sum == real_sum:
            frame_list.append(len(p))

    print sum(frame_list) / float(len(frame_list))
    print len(filter(lambda x: x < 30, frame_list)) / float(len(frame_list))
    print len(filter(lambda x: x < 30, frame_list)) / float(len(p_data))


def clean_other_data_set():
    with open('data/ohter5dataset/xy_8_12_train_set_03.data', 'r') as f:
        data = pickle.load(f)

    train_X = []
    train_Y = []
    for data_x, data_y in zip(data['train_X'], data['train_Y']):
        if len(data_x) > 20:
            train_X.append(data_x[:20])
            train_Y.append(data_y[:20])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    print 'train_X: ', train_X.shape
    print 'train_Y: ', train_Y.shape

    np.savez('data/ohter5dataset/xy_8_12_data_set_3.npz', train_X=train_X, train_Y=train_Y)


def create_Tokyo_dataset():
    # change .mat format to .data format for clear data usage.
    def mat2data(src_data_path, dest_data_path, scale):
        data = sio.loadmat(src_data_path)
        p_num = data['trks'].shape[1]

        f_data = [[] for _ in xrange(100000)]
        p_data = [None] * 100000

        max_f = -1
        for i in xrange(p_num):
            x_pts = list(data['trks'][0, i]['x'].reshape(-1))
            y_pts = list(data['trks'][0, i]['y'].reshape(-1))
            frames = list(data['trks'][0, i]['t'].reshape(-1))

            p_data[i] = {}
            for x, y, f in zip(x_pts, y_pts, frames):
                f = int(f) - 1
                f_20 = f / scale

                if f_20 not in p_data[i]:
                    max_f = max(max_f, f_20)
                    p_data[i][f_20] = (x / 856.0, y / 480.0)
                    f_data[f_20].append(i)

        p_data = filter(lambda x: x != None and x != {}, p_data)
        f_data = f_data[:max_f + 1]

        print 'total_frame:', len(f_data)

        with open(dest_data_path, 'w') as f:
            pickle.dump({'frame_data': f_data, 'pedestrian_data': p_data}, f)

    # change .data format to .npz format and do same sampling for train and test data set
    def data2npz(src_data_path, dest_data_path, train_rate):
        def get_samples_from_frame(f_i):
            if f_i - step < 0 or f_i + step + 1 > frame_num:
                return [], []

            p_set = set(f_data[f_i - step])
            for i in xrange(f_i - step + 1, f_i + step + 1):
                p_set &= set(f_data[i])

            p_set = list(p_set)
            samples_X = []
            samples_Y = []
            p_num = len(p_set)
            for si in xrange(p_num / sample_size):
                sample_X = []
                sample_Y = []
                for i in xrange(si * sample_size, (si + 1) * sample_size):
                    p = p_data[p_set[i]]
                    sx, sy = p[f_i - step][0], p[f_i - step][1]

                    sample_X.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i - step + 1, f_i + 1)])
                    sample_Y.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i + 1, f_i + step + 1)])
                samples_X.append(sample_X)
                samples_Y.append(sample_Y)

            return samples_X, samples_Y

        def collect_sample(f_idx):
            ret_X = []
            ret_Y = []
            for i in f_idx:
                samples_X, samples_Y = get_samples_from_frame(i)
                ret_X += samples_X
                ret_Y += samples_Y

            return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

        # hyper-parameter
        max_frame_num = 5000
        step = 5
        sample_size = 20

        # raw data
        with open(src_data_path, 'r') as f:
            data = pickle.load(f)
        frame_num = min(len(data['frame_data']), max_frame_num)
        f_data = data['frame_data'][:frame_num]
        p_data = data['pedestrian_data']

        # create random center frame idx
        # random_idx = random.sample(range(frame_num), frame_num)
        random_idx = range(frame_num)
        train_idx = random_idx[:int(frame_num * train_rate)]
        test_idx = random_idx[int(frame_num * train_rate):]

        # train set, test set
        train_X, train_Y = collect_sample(train_idx)
        test_X, test_Y = collect_sample(test_idx)

        print train_X.dtype
        print train_X.shape, train_Y.shape
        print test_X.shape, test_Y.shape

        np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

    mat2data('data/Tokyo/Tokyo.mat', 'data/Tokyo/data_set.data', scale=1)
    data2npz('data/Tokyo/data_set.data', 'data/Tokyo/data_set_tr5.npz', train_rate=0.5)
    data2npz('data/Tokyo/data_set.data', 'data/Tokyo/data_set_tr9.npz', train_rate=0.9)


def create_sec2_dataset():
    # change .mat format to .data format for clear data usage.
    def mat2data(src_data_path, dest_data_path, scale):
        data = sio.loadmat(src_data_path)
        p_num = data['trks'].shape[1]

        f_data = [[] for _ in xrange(100000)]
        p_data = [None] * 100000

        max_f = -1
        for i in xrange(p_num):
            x_pts = list(data['trks'][0, i]['x'].reshape(-1))
            y_pts = list(data['trks'][0, i]['y'].reshape(-1))
            frames = list(data['trks'][0, i]['t'].reshape(-1))

            p_data[i] = {}
            for x, y, f in zip(x_pts, y_pts, frames):
                f = int(f) - 1
                f_20 = f / scale

                if f_20 not in p_data[i]:
                    max_f = max(max_f, f_20)
                    p_data[i][f_20] = (x / 856.0, y / 480.0)
                    f_data[f_20].append(i)

        p_data = filter(lambda x: x != None and x != {}, p_data)
        f_data = f_data[:max_f + 1]

        print 'total_frame:', len(f_data)

        with open(dest_data_path, 'w') as f:
            pickle.dump({'frame_data': f_data, 'pedestrian_data': p_data}, f)

    # change .data format to .npz format and do same sampling for train and test data set
    def data2npz(src_data_path, dest_data_path, train_rate):
        def get_samples_from_frame(f_i):
            if f_i - step < 0 or f_i + step + 1 > frame_num:
                return [], []

            p_set = set(f_data[f_i - step])
            for i in xrange(f_i - step + 1, f_i + step + 1):
                p_set &= set(f_data[i])

            p_set = list(p_set)
            samples_X = []
            samples_Y = []
            p_num = len(p_set)
            for si in xrange(p_num / sample_size):
                sample_X = []
                sample_Y = []
                for i in xrange(si * sample_size, (si + 1) * sample_size):
                    p = p_data[p_set[i]]
                    sx, sy = p[f_i - step][0], p[f_i - step][1]

                    sample_X.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i - step + 1, f_i + 1)])
                    sample_Y.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i + 1, f_i + step + 1)])
                samples_X.append(sample_X)
                samples_Y.append(sample_Y)

            return samples_X, samples_Y

        def collect_sample(f_idx):
            ret_X = []
            ret_Y = []
            for i in f_idx:
                samples_X, samples_Y = get_samples_from_frame(i)
                ret_X += samples_X
                ret_Y += samples_Y

            return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

        # hyper-parameter
        max_frame_num = 5000
        step = 5
        sample_size = 20

        # raw data
        with open(src_data_path, 'r') as f:
            data = pickle.load(f)
        frame_num = min(len(data['frame_data']), max_frame_num)
        f_data = data['frame_data'][:frame_num]
        p_data = data['pedestrian_data']

        # create random center frame idx
        # random_idx = random.sample(range(frame_num), frame_num)
        random_idx = range(frame_num)
        train_idx = random_idx[:int(frame_num * train_rate)]
        test_idx = random_idx[int(frame_num * train_rate):]

        # train set, test set
        train_X, train_Y = collect_sample(train_idx)
        test_X, test_Y = collect_sample(test_idx)

        print train_X.dtype
        print train_X.shape, train_Y.shape
        print test_X.shape, test_Y.shape

        np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

    mat2data('data/sec2/sec2.mat', 'data/sec2/data_set.data', scale=1)
    data2npz('data/sec2/data_set.data', 'data/sec2/data_set_tr5.npz', train_rate=0.5)
    data2npz('data/sec2/data_set.data', 'data/sec2/data_set_tr9.npz', train_rate=0.9)


def create_social_LSTM_format_data(src_data_path, dest_train_data_path, dest_test_data_path, train_rate):
    def get_ban_list(p_data):
        ban_p_list = []
        for pid, p in enumerate(p_data):
            if p == None:
                break
            f_list = p.keys()

            sorted_f_list = sorted(f_list)
            flag = False
            for i in xrange(len(sorted_f_list) - 1):
                if sorted_f_list[i] + 1 != sorted_f_list[i + 1]:
                    flag = True
                    break

            if flag or len(f_list) < 20:
                ban_p_list.append(pid)

        return ban_p_list

    def check_and_create_data_set(cut_point, forward):
        s_lstm_format_data = []
        p_len = {}

        if forward:
            scan_range = range(cut_point)
        else:
            scan_range = range(cut_point, total_frame)

        # check
        for f in scan_range:
            pids = f_data[f]
            for pid in pids:
                if pid in ban_p_list:
                    continue

                x, y = p_data[pid][f]
                if pid not in p_len.keys():
                    p_len[pid] = 1
                else:
                    p_len[pid] += 1

        second_ban_list = []
        for pid, num in p_len.iteritems():
            if num < 20:
                second_ban_list.append(pid)

        cur_f = 1
        for f in scan_range:
            pids = f_data[f]
            count = 0
            for pid in pids:
                if pid in ban_p_list or pid in second_ban_list:
                    continue

                x, y = p_data[pid][f]
                if pid not in p_len.keys():
                    p_len[pid] = 1
                else:
                    p_len[pid] += 1

                count += 1
                s_lstm_format_data.append([cur_f, pid + 1, x, y])
            if count != 0:
                cur_f += 1

        return np.array(s_lstm_format_data).T

    with open(src_data_path, 'r') as f:
        data = pickle.load(f)

    f_data = data['frame_data']
    p_data = data['pedestrian_data']

    ban_p_list = get_ban_list(p_data)
    print 'ban list size:', len(ban_p_list)

    total_frame = 500
    cut_point = int(total_frame * train_rate)

    s_lstm_format_train_data = check_and_create_data_set(cut_point, forward=True)
    s_lstm_format_test_data = check_and_create_data_set(cut_point, forward=False)

    print 'train data shape:', s_lstm_format_train_data.shape
    print 'test data shape:', s_lstm_format_test_data.shape

    s_lstm_format_train_data[2:, :] = (s_lstm_format_train_data[2:, :] - 0.5) * 2
    s_lstm_format_test_data[2:, :] = (s_lstm_format_test_data[2:, :] - 0.5) * 2

    with open(dest_train_data_path, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(s_lstm_format_train_data)

    with open(dest_test_data_path, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(s_lstm_format_test_data)


def create_structured_dataset():
    # change .mat format to .data format for clear data usage.
    def mat2data(src_data_path, dest_data_path, scale):
        data = sio.loadmat(src_data_path)
        p_num = data['trks'].shape[1]

        f_data = [[] for _ in xrange(100000)]
        p_data = [None] * 100000

        max_f = -1
        for i in xrange(p_num):
            x_pts = list(data['trks'][0, i]['x'].reshape(-1))
            y_pts = list(data['trks'][0, i]['y'].reshape(-1))
            frames = list(data['trks'][0, i]['t'].reshape(-1))

            p_data[i] = {}
            for x, y, f in zip(x_pts, y_pts, frames):
                f = int(f) - 1
                f_20 = f / scale

                if f_20 not in p_data[i]:
                    max_f = max(max_f, f_20)
                    p_data[i][f_20] = (x / 856.0, y / 480.0)
                    f_data[f_20].append(i)

        p_data = filter(lambda x: x != None and x != {}, p_data)
        f_data = f_data[:max_f + 1]

        print 'total_frame:', len(f_data)

        with open(dest_data_path, 'w') as f:
            pickle.dump({'frame_data': f_data, 'pedestrian_data': p_data}, f)

    # change .data format to .npz format and do same sampling for train and test data set
    def data2npz(src_data_path, dest_data_path, train_rate):
        def get_samples_from_frame(f_i):
            if f_i - step < 0 or f_i + step + 1 > frame_num:
                return [], []

            p_set = set(f_data[f_i - step])
            for i in xrange(f_i - step + 1, f_i + step + 1):
                p_set &= set(f_data[i])

            p_set = list(p_set)
            samples_X = []
            samples_Y = []
            p_num = len(p_set)
            for si in xrange(p_num / sample_size):
                sample_X = []
                sample_Y = []
                for i in xrange(si * sample_size, (si + 1) * sample_size):
                    p = p_data[p_set[i]]
                    sx, sy = p[f_i - step][0], p[f_i - step][1]

                    sample_X.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i - step + 1, f_i + 1)])
                    sample_Y.append([[p[j][0] - sx, p[j][1] - sy] for j in xrange(f_i + 1, f_i + step + 1)])
                samples_X.append(sample_X)
                samples_Y.append(sample_Y)

            return samples_X, samples_Y

        def collect_sample(f_idx):
            ret_X = []
            ret_Y = []
            for i in f_idx:
                samples_X, samples_Y = get_samples_from_frame(i)
                ret_X += samples_X
                ret_Y += samples_Y

            return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)

        # hyper-parameter
        max_frame_num = 5000
        step = 5
        sample_size = 20

        # raw data
        with open(src_data_path, 'r') as f:
            data = pickle.load(f)
        frame_num = min(len(data['frame_data']), max_frame_num)
        f_data = data['frame_data'][:frame_num]
        p_data = data['pedestrian_data']

        # create random center frame idx
        # random_idx = random.sample(range(frame_num), frame_num)
        random_idx = range(frame_num)
        train_idx = random_idx[:int(frame_num * train_rate)]
        test_idx = random_idx[int(frame_num * train_rate):]

        # train set, test set
        train_X, train_Y = collect_sample(train_idx)
        test_X, test_Y = collect_sample(test_idx)

        print train_X.dtype
        print train_X.shape, train_Y.shape
        print test_X.shape, test_Y.shape

        np.savez(dest_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

    # soldiers1
    for i in xrange(1, 4):
        mat_path = 'data/soldiers1/soldiers1_%d.mat' % i
        data_path = 'data/soldiers1/soldiers1_%d.data' % i
        npz_path = 'data/soldiers1/soldiers1_%d.npz' % i

        mat2data(mat_path, data_path, scale=1)
        data2npz(data_path, npz_path, train_rate=0.5)

    # soldiers2
    for i in xrange(1, 5):
        mat_path = 'data/soldiers2/soldiers2_%d.mat' % i
        data_path = 'data/soldiers2/soldiers2_%d.data' % i
        npz_path = 'data/soldiers2/soldiers2_%d.npz' % i

        mat2data(mat_path, data_path, scale=1)
        data2npz(data_path, npz_path, train_rate=0.5)


def concat_structured_dataset():
    structured_train_X = []
    structured_train_Y = []
    structured_test_X = []
    structured_test_Y = []

    # soldiers1 1-3
    for i in xrange(1, 4):
        npz_path = 'data/soldiers1/soldiers1_%d.npz' % i
        data = np.load(npz_path)

        structured_train_X.append(data['train_X'])
        structured_train_X.append(data['test_X'])
        structured_train_Y.append(data['train_Y'])
        structured_train_Y.append(data['test_Y'])

    # soldiers2 1-3
    for i in xrange(1, 4):
        npz_path = 'data/soldiers2/soldiers2_%d.npz' % i
        data = np.load(npz_path)

        structured_train_X.append(data['train_X'])
        structured_train_X.append(data['test_X'])
        structured_train_Y.append(data['train_Y'])
        structured_train_Y.append(data['test_Y'])

    # test: soldiers2 4
    data = np.load('data/soldiers2/soldiers2_4.npz')
    structured_test_X = [data['train_X'], data['test_X']]
    structured_test_Y = [data['train_Y'], data['test_Y']]

    # concat
    structured_train_X = np.concatenate(structured_train_X, axis=0)
    structured_train_Y = np.concatenate(structured_train_Y, axis=0)
    structured_test_X = np.concatenate(structured_test_X, axis=0)
    structured_test_Y = np.concatenate(structured_test_Y, axis=0)

    print structured_train_X.dtype
    print structured_train_X.shape, structured_train_Y.shape
    print structured_test_X.shape, structured_test_Y.shape

    np.savez('data/structured/data_set.npz', train_X=structured_train_X, train_Y=structured_train_Y, test_X=structured_test_X, test_Y=structured_test_Y)


def create_non_linear_GC_dataset(rate):
    data = np.load('data/GC/xy_data_set.npz')
    test_X = data['test_X']
    test_Y = data['test_Y']

    # assume that time interval equals to 1
    v_X = []
    frame_num = test_X.shape[2]
    for i in xrange(frame_num - 1):
        v_X.append(test_X[:, :, i + 1, :] - test_X[:, :, i, :])

    a_X = []
    for i in xrange(frame_num - 2):
        a_X.append(v_X[i + 1] - v_X[i])

    # (frame - 2，B, p_num, 2)
    a_X = np.array(a_X)

    # (frame -2, B, p_num)  which each element present norm of a
    a_X = np.sqrt(np.sum(a_X ** 2, axis=3))

    # (B, p_num)
    a_X = np.max(a_X, axis=0)
    mean_a = a_X.mean()
    print 'mean_a: ', mean_a

    b, p_num = a_X.shape

    hard_idx_list = []
    for i in xrange(b):
        hard = False
        for j in xrange(p_num):
            if a_X[i][j] > rate * mean_a:
                hard = True
                break
        if hard:
            hard_idx_list.append(i)

    print 'hard batch number:', len(hard_idx_list)

    test_X = test_X[hard_idx_list]
    test_Y = test_Y[hard_idx_list]

    # np.savez('data/GC/xy_very_hard_dataset.npz', test_X=test_X, test_Y=test_Y)


def create_non_linear_social_LSTM_format_data(src_test_data_path, dest_test_data_path):
    data = np.load(src_test_data_path)
    # (b, p_num, frame_num, 2)
    test_X = data['test_X']
    test_Y = data['test_Y']

    s_lstm_format_data = []

    batch_size, p_num, frame_num = test_X.shape[:3]
    for i in xrange(batch_size):
        for j in xrange(frame_num):
            for k in xrange(p_num):
                frame_idx = i * frame_num * 2 + j + 1
                p_idx = i * p_num + k + 1 + 10216
                x, y = test_X[i, k, j]

                s_lstm_format_data.append([frame_idx, p_idx, x, y])

        for j in xrange(frame_num):
            for k in xrange(p_num):
                frame_idx = i * frame_num * 2 + j + frame_num + 1
                p_idx = i * p_num + k + 1 + 10216
                x, y = test_Y[i, k, j]

                s_lstm_format_data.append([frame_idx, p_idx, x, y])

    s_lstm_format_data = np.array(s_lstm_format_data).T
    print 's_lstm_format_data shape:', s_lstm_format_data.shape

    with open(dest_test_data_path, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(s_lstm_format_data)


def main():
    # create_xy_train_test_set_with_frame('data/data_set.data', 'all_data_set.npz')
    # create_data_set('data/Time_indexed_infor_02.mat', 'data/data_set_02.data')
    # for i in xrange(2, 6):
    #     create_xy_train_set('data/data_set_0%d.data' % i, 'data/xy_8_12_train_set_0%d.data' % i)

    # create_multiple_frame_xy_train_test_set('data/GC/data_set.data', 'data/GC/multi_frame_GC.data', 1, 25)

    # create_Tokyo_dataset()
    # create_Tokyo_dataset()
    # create_sec2_dataset()
    # create_social_LSTM_format_data('data/Tokyo/data_set.data', 'data/Tokyo/Tokyo_train.csv', 'data/Tokyo/Tokyo_test.csv')
    create_social_LSTM_format_data('data/GC/data_set.data', 'data/GC/TINY_SLGC_train.csv', 'data/GC/TINY_SLGC_test.csv', train_rate=0.9)

    # create_structured_dataset()
    # concat_structured_dataset()

    # create_non_linear_GC_dataset(5)
    # create_non_linear_social_LSTM_format_data('data/GC/xy_hard_dataset.npz', 'data/GC/SLGC_hard_test.csv')
    # create_non_linear_social_LSTM_format_data('data/GC/xy_very_hard_dataset.npz', 'data/GC/SLGC_very_hard_test.csv')
    pass


if __name__ == '__main__':
    main()
