# -*- coding: utf-8 -*-
# @Time    : 2017/7/12 下午9:17
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn


import scipy.io as sio
import numpy as np
import random
import os
import sys
import json
import visdom
import csv


def create_GC_metadata(raw_data_path, meta_data_path):
    """
    create data from downloaded raw data to meta data ( a data structure to read easily)
    :param raw_data_path: downloaded raw data path
    :param meta_data_path: meta data path
    :return:
    """

    GC_IMAGE_WIDTH = 1920
    GC_IMAGE_HEIGHT = 1080

    dir_list = sorted(os.listdir(raw_data_path))
    p_num = len(dir_list)

    # + 1 because raw GC txt start from 1,just add a fake person whose pid = 0
    p_data_list = [{} for _ in range(p_num + 1)]

    # fill p_data
    max_t = 0
    for dir_name in dir_list:
        person_trajectory_txt_path = os.path.join(raw_data_path, dir_name)
        pid = int(dir_name.replace('.txt', ''))

        with open(person_trajectory_txt_path, 'r') as f:
            trajectory_list = f.read().split()
            for i in range(len(trajectory_list) // 3):
                x = int(trajectory_list[3 * i]) / GC_IMAGE_WIDTH
                y = int(trajectory_list[3 * i + 1]) / GC_IMAGE_HEIGHT
                t = int(trajectory_list[3 * i + 2]) // 20
                max_t = max(max_t, t)
                p_data_list[pid][t] = (x, y)

    # fill f_data
    f_data_list = [[] for _ in range(max_t + 1)]
    for pid, p_data in enumerate(p_data_list):
        for t in p_data.keys():
            f_data_list[t].append(pid)

    # show some message
    print('pedestrian_data_list size: ', len(p_data_list))
    print('frame_data_list size: ', len(f_data_list))

    with open(meta_data_path, 'w') as f:
        json.dump({'frame_data_list': f_data_list, 'pedestrian_data_list': p_data_list}, f)

    print('create %s successfully!' % meta_data_path)


def get_samples_from_frame(f_i, step, frame_num, f_data_list, p_data_list, sample_size):
    if f_i - step < 0 or f_i + step > frame_num:
        return [], []

    p_set = set(f_data_list[f_i - step])
    for i in range(f_i - step + 1, f_i + step):
        p_set &= set(f_data_list[i])

    p_set = list(p_set)
    samples_X = []
    samples_Y = []
    p_num = len(p_set)
    for si in range(p_num // sample_size):
        sample_X = []
        sample_Y = []
        for i in range(si * sample_size, (si + 1) * sample_size):
            p_data = p_data_list[p_set[i]]
            sample_X.append([[p_data[str(j)][0], p_data[str(j)][1]] for j in range(f_i - step, f_i)])
            sample_Y.append([[p_data[str(j)][0], p_data[str(j)][1]] for j in range(f_i, f_i + step)])
        samples_X.append(sample_X)
        samples_Y.append(sample_Y)

    return samples_X, samples_Y


def collect_sample(f_idx, step, frame_num, f_data_list, p_data_list, sample_size):
    ret_X = []
    ret_Y = []
    for i in f_idx:
        samples_X, samples_Y = get_samples_from_frame(i, step, frame_num, f_data_list, p_data_list, sample_size)
        ret_X += samples_X
        ret_Y += samples_Y

    return np.array(ret_X).astype(np.float32), np.array(ret_Y).astype(np.float32)


def create_GC_train_test_data(meta_data_path, train_test_data_path, random_seed=0):
    # hyper-parameter
    frame_num = 5000
    train_rate = 0.9
    step = 5
    sample_size = 20

    # random seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    # raw data
    with open(meta_data_path, 'r') as f:
        data = json.load(f)
    f_data_list = data['frame_data_list'][:frame_num]
    p_data_list = data['pedestrian_data_list']

    # create random center frame idx
    random_idx = random.sample(range(frame_num), frame_num)
    train_idx = random_idx[:int(frame_num * train_rate)]
    test_idx = random_idx[int(frame_num * train_rate):]

    # train set, test set
    train_X, train_Y = collect_sample(train_idx, step, frame_num, f_data_list, p_data_list, sample_size)
    test_X, test_Y = collect_sample(test_idx, step, frame_num, f_data_list, p_data_list, sample_size)

    print(train_X.dtype)
    print('train_X: %s, train_Y: %s' % (train_X.shape, train_Y.shape))
    print('test_X: %s, test_Y: %s' % (test_X.shape, test_Y.shape))

    np.savez(train_test_data_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

    print('create %s successfully!' % train_test_data_path)


def main():
    GC_raw_data_path = 'data/GC/Annotation'
    GC_meta_data_path = 'data/GC_meta_data.json'
    GC_train_test_data_path = 'data/GC.npz'

    create_GC_metadata(GC_raw_data_path, GC_meta_data_path)
    create_GC_train_test_data(GC_meta_data_path, GC_train_test_data_path)


if __name__ == '__main__':
    main()
