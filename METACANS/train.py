import os
import sys
import argparse
import torch.nn as nn
import torch
import numpy as np
import time
import pickle, gzip
import matplotlib.pyplot as plt
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)
from models.model_ALNM import GatedAttention
from models.utils import train_model, test_model, save_model, adjust_learning_rate
from models.generator import dataloader

import random

def main_worker(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    save_dir = args.exp_dir + '_fold_' + str(args.val_fold) + '_' + str(args.num_epochs)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = GatedAttention()
    # model = GatedAttention_softmax_temp()
    model.cuda()
    optimizer     = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    pos_weight    = torch.tensor([2.69]).cuda() # 2.69
    criterion     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    loader_train  = dataloader(path_data=args.path_data,
                               path_excel=args.path_data_excel,
                               hospital=args.hospital_tr,
                               batch_size=1,
                               num_workers=args.num_workers,
                               val_fold=args.val_fold,
                               mode='train')
    loader_valid  = dataloader(path_data=args.path_data,
                               path_excel=args.path_data_excel,
                               hospital=args.hospital_tr,
                               batch_size=1,
                               num_workers=args.num_workers,
                               val_fold=args.val_fold,
                               mode='valid')
    print(len(loader_train))
    print('data complete')

    loss_tr_all = []
    loss_val_all = []
    AUC_tr_all = []
    AUC_val_all = []
    for epoch in range(args.start_epochs, args.num_epochs):
        time_start = time.time()
        print("{} epoch: ".format(save_dir), epoch, "-" * 100)
        optimizer = adjust_learning_rate(args, epoch, optimizer, args.lr)
        loss_tr , AUC_tr, model, ids                    = train_model(args, model, loader_train, criterion, optimizer)
        loss_val, AUC_val, A, pred_all, label_all, ids  =  test_model(args, model, loader_valid, criterion)
        save_model(save_dir, epoch, model, loss_tr, loss_val, AUC_tr, AUC_val)

        loss_tr_all.append(loss_tr)
        loss_val_all.append(loss_val)
        AUC_tr_all.append(AUC_tr)
        AUC_val_all.append(AUC_val)

        plt.figure(1)
        plt.plot(loss_tr_all, color='#1f77b4')
        plt.plot(loss_val_all, color='#2ca02c')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train','Valid'])
        plt.title('Loss Curve')
        plt.savefig(save_dir + '/_LossCurve.png', dpi=300)

        plt.figure(2)
        plt.plot(AUC_tr_all, color='#1f77b4')
        plt.plot(AUC_val_all, color='#2ca02c')
        plt.xlabel('epoch')
        plt.ylabel('AUC')
        plt.legend(['Train','Valid'])
        plt.title('AUC Curve')
        plt.savefig(save_dir + '/_AUCCurve.png', dpi=300)

        time_end = time.time()
        # print(time_end-time_start)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        default='6', type=str, help='GPU Number')
    parser.add_argument('--hospital_tr',
                        default='ss_tr', type=str, help='')
    parser.add_argument('--val_fold',
                        default=5, type=int, help='')
    
    parser.add_argument('--exp-dir',
                        default='/media/data1/doohyun/wsi/code/MIL_base/trained_model_1.0', type=str, help='')
    parser.add_argument('--path_data',
                        default='/media/data1/doohyun/wsi/data/svs_feature_CTransPath_224_1.0_sn',
                        type=str, help='train set')
    parser.add_argument('--path_data_excel',
                        default='/media/data1/doohyun/wsi/data/excel',
                        type=str, help='')
    
    parser.add_argument('--start-epochs', type=int, default=0, help='final epochs')
    parser.add_argument('--num-epochs', type=int, default=400, help='final epochs')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Strength of weight decay regularization')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of jobs')
    parser.add_argument('--seed', type=int, default=42, help='')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    main_worker(args)