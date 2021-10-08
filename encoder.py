# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: encoder.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/20
# -----------------------------------------
import torch.nn as nn
import torch


# -------------------------------
# RDN encoder network
# <Zhang, Yulun, et al. "Residual dense network for image super-resolution.">
# Here code is modified from: https://github.com/yjn870/RDN-pytorch/blob/master/models.py
# -------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, feature_dim=128, num_features=64, growth_rate=64, num_blocks=8, num_layers=3):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        # shallow feature extraction
        self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv3d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv3d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.output = nn.Conv3d(self.G0, feature_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.output(x)
        return x


# -------------------------------
# ResCNN encoder network
# <Du, Jinglong, et al. "Super-resolution reconstruction of single
# anisotropic 3D MR images using residual convolutional neural network.">
# -------------------------------
class ResCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResCNN, self).__init__()
        self.conv_start = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_end = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=feature_dim, kernel_size=3, padding=3 // 2),
        )

    def forward(self, x):
        in_block1 = self.conv_start(x)
        out_block1 = self.block1(in_block1)
        in_block2 = out_block1 + in_block1
        out_block2 = self.block2(in_block2)
        in_block3 = out_block2 + in_block2
        out_block3 = self.block3(in_block3)
        res_img = self.conv_end(out_block3 + in_block3)
        return x + res_img


# -------------------------------
# SRResNet
# <Ledig, Christian, et al. "Photo-realistic single image super-resolution
# using a generative adversarial network.">
# -------------------------------
def conv(ni, nf, kernel_size=3, actn=False):
    layers = [nn.Conv3d(ni, nf, kernel_size, padding=kernel_size // 2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x): return x + self.m(x) * self.res_scale


def res_block(nf):
    return ResSequential(
        [conv(nf, nf, actn=True), conv(nf, nf)],
        1.0)  # this is best one


class SRResnet(nn.Module):
    def __init__(self, nf=64, feature_dim=128):
        super().__init__()
        features = [conv(1, nf)]
        for i in range(18): features.append(res_block(nf))
        features += [conv(nf, nf),
                     conv(nf, feature_dim)]
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)
