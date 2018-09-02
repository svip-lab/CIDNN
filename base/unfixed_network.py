# -*- coding: utf-8 -*-
# @Time    : 2017/9/9 下午11:47
# @Author  : Zhixin Piao
# @Email   : piaozhx@shanghaitech.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import torchvision
from torch import optim

import numpy as np
import os


class TraceDataSet(Dataset):
    def __init__(self, input_traces, target_traces, traces_pedestrian_num):
        self.input_traces = torch.FloatTensor(input_traces)
        self.target_traces = torch.FloatTensor(target_traces)
        self.traces_pedestrian_num = traces_pedestrian_num

    def __getitem__(self, index):
        return self.input_traces[index], self.target_traces[index], self.traces_pedestrian_num[index]

        # return img, target

    def __len__(self):
        return self.input_traces.size(0)


class UnfixedAttention(nn.Module):
    def __init__(self):
        super(UnfixedAttention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces, hidden_mask):
        # print input_hidden_traces.size()
        # print hidden_mask.size()

        mask_input_hidden_traces = input_hidden_traces * hidden_mask
        mask_target_hidden_traces = target_hidden_traces * hidden_mask

        Attn = torch.bmm(mask_input_hidden_traces, mask_target_hidden_traces.transpose(1, 2))

        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)

        # batch-based softmax
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn


class UnfixedRegressionNet(nn.Module):
    def __init__(self, pedestrian_num, regression_size, hidden_size):
        super(UnfixedRegressionNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.regression_size = regression_size  # 2
        self.hidden_size = hidden_size

        hidden1_size = 32
        hidden2_size = 64

        self.fc1 = torch.nn.Linear(hidden_size, regression_size)
        self.fc2 = torch.nn.Linear(regression_size, hidden1_size)
        self.fc3 = torch.nn.Linear(hidden1_size, regression_size)

    def forward(self, input_attn_hidden_traces, target_hidden_traces, target_traces, regression_mask):
        # target_hidden_trace: (B, pedestrian_num, hidden_size)
        batch_size = input_attn_hidden_traces.size(0)

        regression_list = []
        for i in range(self.pedestrian_num):
            input_attn_hidden_trace = input_attn_hidden_traces[:, i]
            target_delta_trace = self.fc1(input_attn_hidden_trace)

            regression_list.append(target_delta_trace)
        regression_traces = torch.stack(regression_list, 1)

        mask_regression_traces = regression_traces * regression_mask
        regression_traces = mask_regression_traces + target_traces

        # regression_traces: (B, pedestrian_num, regression_size)
        return regression_traces
