# -*- coding: utf-8 -*-
# @Time    : 2018/9/2 11:28 PM
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import math


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizers, init_lr, epoch):
    lr = init_lr * (0.5 ** (epoch // 30))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
