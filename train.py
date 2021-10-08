# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: train.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import data
import torch
import model
import argparse
import time
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    writer = SummaryWriter('./log')

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about ArSSR model
    parser.add_argument('-encoder_name', type=str, default='RDN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=256, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')

    # about training and validation data
    parser.add_argument('-hr_data_train', type=str, default='./data/hr_train', dest='hr_data_train',
                        help='the file path of HR patches for training')
    parser.add_argument('-hr_data_val', type=str, default='./data/hr_val', dest='hr_data_val',
                        help='the file path of HR patches for validation')

    # about training hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr',
                        help='the initial learning rate')
    parser.add_argument('-lr_decay_epoch', type=int, default=200, dest='lr_decay_epoch',
                        help='learning rate multiply by 0.5 per lr_decay_epoch .')
    parser.add_argument('-epoch', type=int, default=2500, dest='epoch',
                        help='the total number of epochs for training')
    parser.add_argument('-summary_epoch', type=int, default=200, dest='summary_epoch',
                        help='the current model will be saved per summary_epoch')
    parser.add_argument('-bs', type=int, default=15, dest='batch_size',
                        help='the number of LR-HR patch pairs (i.e., N in Equ. 3)')
    parser.add_argument('-ss', type=int, default=8000, dest='sample_size',
                        help='the number of sampled voxel coordinates (i.e., K in Equ. 3)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    hr_data_train = args.hr_data_train
    hr_data_val = args.hr_data_val
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    batch_size = args.batch_size
    sample_size = args.sample_size
    gpu = args.gpu

    # -----------------------
    # display parameters
    # -----------------------
    print('Parameter Settings')
    print('')
    print('------------File------------')
    print('hr_data_train: {}'.format(hr_data_train))
    print('hr_data_val: {}'.format(hr_data_val))
    print('------------Train-----------')
    print('lr: {}'.format(lr))
    print('batch_size_train: {}'.format(batch_size))
    print('sample_size: {}'.format(sample_size))
    print('gpu: {}'.format(gpu))
    print('epochs: {}'.format(epoch))
    print('summary_epoch: {}'.format(summary_epoch))
    print('lr_decay_epoch: {}'.format(lr_decay_epoch))
    print('------------Model-----------')
    print('encoder_name : {}'.format(encoder_name))
    print('decoder feature_dim: {}'.format(feature_dim))
    print('decoder depth: {}'.format(decoder_depth))
    print('decoder width: {}'.format(decoder_width))
    for i in range(5):
        print(i + 1, end="s,")
        time.sleep(1)

    # -----------------------
    # load data
    # -----------------------
    train_loader = data.loader_train(in_path_hr=hr_data_train, batch_size=batch_size,
                                     sample_size=sample_size, is_train=True)
    val_loader = data.loader_train(in_path_hr=hr_data_val, batch_size=1,
                                   sample_size=sample_size, is_train=False)

    # -----------------------
    # model & optimizer
    # -----------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    ArSSR = model.ArSSR(encoder_name=encoder_name, feature_dim=feature_dim,
                        decoder_depth=int(decoder_depth / 2), decoder_width=decoder_width).to(DEVICE)
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=ArSSR.parameters(), lr=lr)

    # -----------------------
    # training & validation
    # -----------------------
    for e in range(epoch):
        ArSSR.train()
        loss_train = 0
        for i, (img_lr, xyz_hr, img_hr) in enumerate(train_loader):
            # forward
            img_lr = img_lr.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d
            img_hr = img_hr.to(DEVICE).float().view(batch_size, -1).unsqueeze(-1)  # N×K×1 (K Equ. 3)
            xyz_hr = xyz_hr.view(batch_size, -1, 3).to(DEVICE).float()  # N×K×3
            img_pre = ArSSR(img_lr, xyz_hr)  # N×K×1
            loss = loss_fun(img_pre, img_hr)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record and print loss
            loss_train += loss.item()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('(TRAIN) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.10f}'.format(e + 1,
                                                                                   epoch,
                                                                                   i + 1,
                                                                                   len(train_loader),
                                                                                   current_lr,
                                                                                   loss.item()))

        writer.add_scalar('MES_train', loss_train / len(train_loader), e + 1)
        # release memory
        img_lr = None
        img_hr = None
        xyz_hr = None
        img_pre = None

        ArSSR.eval()
        with torch.no_grad():
            loss_val = 0
            for i, (img_lr, xyz_hr, img_hr) in enumerate(val_loader):
                img_lr = img_lr.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d
                xyz_hr = xyz_hr.view(1, -1, 3).to(DEVICE).float()  # N×Q×3 (Q=H×W×D)
                H, W, D = img_hr.shape[-3:]
                img_hr = img_hr.to(DEVICE).float().view(1, -1).unsqueeze(-1)  # N×Q×1 (Q=H×W×D)
                img_pre = ArSSR(img_lr, xyz_hr)  # N×Q×1 (Q=H×W×D)
                loss_val += loss_fun(img_hr, img_pre)
                # save validation
                if (e + 1) % summary_epoch == 0:
                    # save model
                    torch.save(ArSSR.state_dict(), 'model/model_param_{}.pkl'.format(e + 1))

        writer.add_scalar('MES_val', loss_val / len(val_loader), e + 1)
        # release memory
        img_lr = None
        img_hr = None
        xyz_hr = None
        img_pre = None

        # learning rate decays by half every some epochs.
        if (e + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    writer.flush()
