# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: decoder.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/20
# -----------------------------------------
import torch.nn as nn


# -------------------------------
# decoder implemented by a simple MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=128 + 3, out_dim=1, depth=4, width=256):
        super(MLP, self).__init__()
        stage_one = []
        stage_two = []
        for i in range(depth):
            if i == 0:
                stage_one.append(nn.Linear(in_dim, width))
                stage_two.append(nn.Linear(in_dim, width))
            elif i == depth - 1:
                stage_one.append(nn.Linear(width, in_dim))
                stage_two.append(nn.Linear(width, out_dim))
            else:
                stage_one.append(nn.Linear(width, width))
                stage_two.append(nn.Linear(width, width))
            stage_one.append(nn.ReLU())
            stage_two.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one)
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        h = self.stage_one(x)
        return self.stage_two(x + h)
