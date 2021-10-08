# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: models.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
import decoder
import encoder


# -------------------------------
# ArSSR model
# -------------------------------
class ArSSR(nn.Module):
    def __init__(self, encoder_name, feature_dim, decoder_depth, decoder_width):
        super(ArSSR, self).__init__()
        if encoder_name == 'RDN':
            self.encoder = encoder.RDN(feature_dim=feature_dim)
        if encoder_name == 'SRResnet':
            self.encoder = encoder.SRResnet(feature_dim=feature_dim)
        if encoder_name == 'ResCNN':
            self.encoder = encoder.ResCNN(feature_dim=feature_dim)
        self.decoder = decoder.MLP(in_dim=feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width)

    def forward(self, img_lr, xyz_hr):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image
        """
        # extract feature map from LR image
        feature_map = self.encoder(img_lr)  # N×1×h×w×d
        # generate feature vector for coordinate through trilinear interpolation (Equ. 4 & Fig. 3).
        feature_vector = F.grid_sample(feature_map, xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        # concatenate coordinate with feature vector
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr], dim=-1)  # N×K×(3+feature_dim)
        # estimate the voxel intensity at the coordinate by using decoder.
        N, K = xyz_hr.shape[:2]
        intensity_pre = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1)
        return intensity_pre
