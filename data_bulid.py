import os
import utils
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np

hr_path = 'your raw HCP-1200 dataset dir'
save_path = 'your saved HCP-1200 training dataset dir'

filename = os.listdir(hr_path)
train_filename = filename[0:780]
val_filename = filename[780:890]

for f in tqdm(train_filename):
    name = f.split('.')[0]
    for i in range(6):
        img_in = sitk.GetArrayFromImage(sitk.ReadImage('{}/{}.nii.gz'.format(hr_path, f)))
        h, w, d = img_in.shape
        # ±40 for avoiding black background region
        x0 = np.random.randint(40, h-40-40)
        y0 = np.random.randint(40, w-40-40)
        z0 = np.random.randint(40, d-40-40)
        img_in = img_in[x0:x0+40, y0:y0+40, z0:z0+40]
        utils.write_img(vol=img_in, ref_path='{}/{}'.format(hr_path, f), 
                        out_path='{}/train/{}_{}.nii.gz'.format(save_path, name, i))
                        
for f in tqdm(val_filename):
    name = f.split('.')[0]
    for i in range(6):
        img_in = sitk.GetArrayFromImage(sitk.ReadImage('{}/{}.nii.gz'.format(hr_path, f)))
        h, w, d = img_in.shape
        # ±40 for avoiding black background region
        x0 = np.random.randint(40, h-40-40)
        y0 = np.random.randint(40, w-40-40)
        z0 = np.random.randint(40, d-40-40)
        img_in = img_in[x0:x0+40, y0:y0+40, z0:z0+40]
        utils.write_img(vol=img_in, ref_path='{}/{}'.format(hr_path, f), 
                        out_path='{}/val/{}_{}.nii.gz'.format(save_path, name, i))
        