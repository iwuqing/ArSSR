# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: utils.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity
from tqdm import tqdm


def read_img(in_path):
    img_lit = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_lit.append(img_vol)
    return img_lit


# -------------------------------
# here coder is from https://github.com/yinboc/liif/blob/main/utils.py
# -------------------------------
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def psnr(image, ground_truth):
    mse = np.mean((image - ground_truth) ** 2)
    if mse == 0.:
        return float('inf')
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)


def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)


def normal(in_image):
    value_max = np.max(in_image)
    value_min = np.min(in_image)
    return (in_image - value_min) / (value_max - value_min)
