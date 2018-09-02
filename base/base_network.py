# -*- coding: utf-8 -*-
# @Time    : 2017/9/3 下午6:49
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import torchvision
from torch import optim

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class L2_DistanceAttention(nn.Module):
    def __init__(self):
        super(L2_DistanceAttention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        standard_size = (input_hidden_traces.size(0), input_hidden_traces.size(1), input_hidden_traces.size(1))

        # L2 distance
        target_hidden_traces_square = (target_hidden_traces ** 2).sum(2).unsqueeze(2).expand(standard_size)
        input_hidden_traces_square = (input_hidden_traces ** 2).transpose(1, 2).sum(1).unsqueeze(1).expand(standard_size)
        input_target_mm = torch.bmm(target_hidden_traces, input_hidden_traces.transpose(1, 2))
        inner_distance = target_hidden_traces_square + input_hidden_traces_square - 2 * input_target_mm

        Attn = -inner_distance

        # exp
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(standard_size)
        exp_Attn = torch.exp(Attn)

        # batch-based softmax
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(standard_size)
        return Attn, inner_distance


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        Attn = torch.bmm(target_hidden_traces, input_hidden_traces.transpose(1, 2))

        inner_Attn = Attn

        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)

        # batch-based softmax
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn


# pedestrian_num is indeterminate
class EncoderNet(nn.Module):
    def __init__(self, pedestrian_num, input_size, hidden_size):
        super(EncoderNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.hidden_size = hidden_size

        hidden1_size = 32
        hidden2_size = 64

        self.fc1 = torch.nn.Linear(input_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = torch.nn.Linear(hidden2_size, hidden_size)

    def forward(self, input_traces):
        # input_trace: (B, pedestrian_num, input_size(input_frame * 2))
        hidden_list = []

        for i in range(self.pedestrian_num):
            input_trace = input_traces[:, i, :]
            hidden_trace = F.relu(self.fc1(input_trace))

            hidden_trace = F.relu(self.fc2(hidden_trace))
            hidden_trace = self.fc3(hidden_trace)

            hidden_list.append(hidden_trace)

        hidden_traces = torch.stack(hidden_list, 1)

        return hidden_traces


# target_frame is indeterminate
class DecoderNet(nn.Module):
    def __init__(self, pedestrian_num, target_size, hidden_size, window_size):
        super(DecoderNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.target_size = target_size  # 2
        self.hidden_size = hidden_size
        self.window_size = window_size

        hidden1_size = 32
        hidden2_size = 64

        self.fc1 = torch.nn.Linear(target_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = torch.nn.Linear(hidden2_size, hidden_size)

    def forward(self, target_traces):
        # target_trace: (B, pedestrian_num, 2)
        hidden_list = []

        for i in range(self.pedestrian_num):
            target_trace = target_traces[:, i, :]
            hidden_trace = F.relu(self.fc1(target_trace))
            hidden_trace = F.relu(self.fc2(hidden_trace))
            hidden_trace = self.fc3(hidden_trace)

            hidden_list.append(hidden_trace)

        # stack all person
        hidden_traces = torch.stack(hidden_list, 1)

        # hidden_trace: (B, pedestrian_num, hidden_size)
        return hidden_traces


class RegressionNet(nn.Module):
    def __init__(self, pedestrian_num, regression_size, hidden_size):
        super(RegressionNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.regression_size = regression_size  # 2
        self.hidden_size = hidden_size

        hidden1_size = 32
        hidden2_size = 64

        self.fc1 = torch.nn.Linear(hidden_size, regression_size)
        self.fc2 = torch.nn.Linear(regression_size, hidden1_size)
        self.fc3 = torch.nn.Linear(hidden1_size, regression_size)

    def forward(self, input_attn_hidden_traces, target_hidden_traces, target_traces):
        # target_hidden_trace: (B, pedestrian_num, hidden_size)

        regression_list = []
        for i in range(self.pedestrian_num):
            input_attn_hidden_trace = input_attn_hidden_traces[:, i]
            target_delta_trace = self.fc1(input_attn_hidden_trace)

            regression_list.append(target_delta_trace)
        regression_traces = torch.stack(regression_list, 1)
        regression_traces = regression_traces + target_traces

        # regression_traces: (B, pedestrian_num, regression_size)
        return regression_traces


class EncoderNetWithLSTM(nn.Module):
    def __init__(self, pedestrian_num, input_size, hidden_size, n_layers=2):
        super(EncoderNetWithLSTM, self).__init__()
        input_size = 2
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, self.n_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, self.n_layers)

    def forward(self, input_traces, hidden):
        # input_trace: (B, pedestrian_num, 2)
        next_hidden_list = []
        output_list = []
        for i in range(self.pedestrian_num):
            input_trace = input_traces[:, i, :].unsqueeze(0)
            output, next_hidden = self.lstm(input_trace, (hidden[i][0], hidden[i][1]))

            next_hidden_list.append(next_hidden)
            output_list.append(output.squeeze(0))

        output_traces = torch.stack(output_list, 1)

        return output_traces, next_hidden_list

    def init_hidden(self, batch_size):
        return [[torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).cuda()
                 for _ in range(2)]
                for _ in range(self.pedestrian_num)]
