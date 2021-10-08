# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: dataset.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import utils
import numpy as np
import random
import SimpleITK as sitk
from torch.utils import data
from scipy import ndimage as nd


# -----------------------
# Training data
# -----------------------

class ImgTrain(data.Dataset):
    def __init__(self, in_path_hr, sample_size, is_train):
        self.is_train = is_train
        self.sample_size = sample_size
        self.patch_hr = utils.read_img(in_path=in_path_hr)

    def __len__(self):
        return len(self.patch_hr)

    def __getitem__(self, item):
        patch_hr = self.patch_hr[item]
        # randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        # compute the size of HR patch according to the scale
        hr_h, hr_w, hr_d = (np.array([10, 10, 10]) * s).astype(int)
        # generate HR patch by cropping
        patch_hr = patch_hr[:hr_h, :hr_w, :hr_d]
        # simulated LR patch by down-sampling HR patch
        patch_lr = nd.interpolation.zoom(patch_hr, 1 / s, order=3)
        # generate coordinate set
        xyz_hr = utils.make_coord(patch_hr.shape, flatten=True)
        # randomly sample voxel coordinates
        if self.is_train:
            sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]

        return patch_lr, xyz_hr, patch_hr


def loader_train(in_path_hr, batch_size, sample_size, is_train):
    """
    :param in_path_hr: the path of HR patches
    :param batch_size: N in Equ. 3
    :param sample_size: K in Equ. 3
    :param is_train:
    :return:
    """
    return data.DataLoader(
        dataset=ImgTrain(in_path_hr=in_path_hr, sample_size=sample_size, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )


# -----------------------
# Testing data
# -----------------------

class ImgTest(data.Dataset):
    def __init__(self, in_path_lr, scale):
        self.img_lr = []
        self.xyz_hr = []
        # load lr image
        lr_vol = sitk.GetArrayFromImage(sitk.ReadImage(in_path_lr))
        self.img_lr.append(lr_vol)
        for img_lr in self.img_lr:
            temp_size = np.array(img_lr.shape).astype(float)
            temp_size *= scale
            temp_size = list(temp_size.astype(int))
            self.xyz_hr.append(utils.make_coord(temp_size, flatten=True))

    def __len__(self):
        return len(self.img_lr)

    def __getitem__(self, item):
        return self.img_lr[item], self.xyz_hr[item]


def loader_test(in_path_lr, scale):
    return data.DataLoader(
        dataset=ImgTest(in_path_lr=in_path_lr, scale=scale),
        batch_size=1,
        shuffle=False
    )
