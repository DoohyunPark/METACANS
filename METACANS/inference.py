import os
import sys
import argparse
import torch.nn as nn
import torch
import numpy as np
import time
import pickle, gzip
import shutil
import matplotlib.pyplot as plt
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)
from models.model_ALNM import GatedAttention
from models.utils import min_model_finder, test_model
from models_infer.visualization import ROC_curve
from models.generator import dataloader


def main_worker(args):
    save_path = args.save_path + '/' + args.hospital
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/maps', exist_ok=True)
    
    loader  = dataloader(path_data=args.path_data,
                        path_excel=args.path_data_excel,
                        hospital=args.hospital,
                        batch_size=1,
                        num_workers=args.num_workers,
                        val_fold=args.val_fold,
                        mode=args.mode)
    
    opt_model_path = min_model_finder(args.model_path)
    opt_model_path = args.model_path + '/' + opt_model_path
    print(opt_model_path)

    model = GatedAttention()
    model.cuda()
    model_state_dict = torch.load(opt_model_path)
    model.load_state_dict(model_state_dict)

    pos_weight    = torch.tensor([2.69]).cuda()
    criterion     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    losses, auc, As, pred_all, label_all, ids = test_model(args, model, loader, criterion)
    ROC_curve(label_all, pred_all, save_path)

    ## Attention Map Generation
    for i, item in enumerate(ids):
        patch_path = '/media/data1/doohyun/wsi/data/svs_patch_224_1.0_sn/' + args.hospital + '/' + item[0:10]
        patch_path_ = os.listdir(patch_path)
        patch_path_.sort()
        locs = np.zeros((2,len(patch_path_)))
        for j,item2 in enumerate(patch_path_):
            patch_name = item2.split('_')
            locs[0,j] = int(patch_name[4][0:3])
            locs[1,j] = int(patch_name[3])

        As_tmp = np.squeeze(As[i]) * len(As[i]) # normalize attention scores
        sort_index = np.argsort(As_tmp)
        # index_lower = sort_index[0:int(np.floor(len(As_tmp)*0.10))]
        # index_upper = sort_index[int(np.floor(len(As_tmp)*0.90))::]
        index_lower = sort_index[0:1]
        index_upper = sort_index[-1::]
        # minlen = min([len(index_lower), len(index_upper)])
        # index_lower = index_lower[0:minlen]
        # index_upper = index_upper[0:minlen]

        # save_path0 = '/media/data1/doohyun/wsi/data/patch_trained_model_1.0_fold_7_400/' + args.hospital + '/' + str(label_all[i]) + '/attention_min'
        # save_path1 = '/media/data1/doohyun/wsi/data/patch_trained_model_1.0_fold_7_400/' + args.hospital + '/' + str(label_all[i]) + '/attention_max'
        # os.makedirs(save_path0, exist_ok=True)
        # os.makedirs(save_path1, exist_ok=True)
        
        # for k, lower_item in enumerate(index_lower):
        #     patch_tmp = patch_path + '/' + patch_path_[lower_item]
        #     shutil.copy(patch_tmp, save_path0 + '/' + patch_path_[lower_item])
        # for k, upper_item in enumerate(index_upper):
        #     patch_tmp = patch_path + '/' + patch_path_[upper_item]
        #     shutil.copy(patch_tmp, save_path1 + '/' + patch_path_[upper_item])
        
        
        # Med = (min(As_tmp) + max(As_tmp))/2
        # maps = np.zeros((max(locs[0,:]).astype(int)+2, max(locs[1,:]).astype(int)+2))
        
        # # maps = maps + 0.5
        # # As_lower = As_tmp[sort_index[int(np.floor(len(As_tmp)*0.10))]]
        # # As_upper = As_tmp[sort_index[int(np.floor(len(As_tmp)*0.90))]]
        # # for j in range(len(patch_path_)):
        # #     maps[locs[0,j].astype(int),locs[1,j].astype(int)] = 0.45
        # #     if As_tmp[j] >= As_upper:
        # #         maps[locs[0,j].astype(int),locs[1,j].astype(int)] = 1
        # #     elif As_tmp[j] <= As_lower:
        # #         maps[locs[0,j].astype(int),locs[1,j].astype(int)] = 0

        # Med = (min(As_tmp) + max(As_tmp))/2
        # maps = maps + 0.5
        # for j, item2 in enumerate(As_tmp):
        #     maps[locs[0,j].astype(int),locs[1,j].astype(int)] = item2
        
        # plt.figure(dpi=600)
        # plt.imshow(maps, cmap='seismic')
        # plt.axis('off')
        # plt.show()
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        # plt.clim(0,1)
        # # plt.colorbar()
        # plt.savefig(save_path + '/maps/' + item[0:10] + '.png')
        # plt.close()
        
        # print(i, item)

    f = open(save_path + '/pred_' + str(args.val_fold) + '.txt','w')
    for name in pred_all:
        f.write(str(name)+'\n')
    f.close()
    f = open(save_path + '/label.txt','w')
    for name in label_all:
        f.write(str(name)+'\n')
    f.close()
    f = open(save_path + '/As_' + str(args.val_fold) + '.txt','w')
    for name in As:
        f.write(str(name)+'\n')
    f.close()
    f = open(save_path + '/ids.txt','w')
    for name in ids:
        f.write(str(name)+'\n')
    f.close()

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='GPU Number')
    
    parser.add_argument('--hospital', type=str,  default='ss_tr',\
                        help='ss_te / dk / ewha / gc / gs / km')
    
    parser.add_argument('--val_fold', type=int,  default='7',\
                        help='')
    
    parser.add_argument('--mode', type=str,  default='train',\
                        help='')
    
    parser.add_argument('--model_path',
                        default='/media/data1/doohyun/wsi/code/MIL_base/trained_model_1.0/fold_7_400',
                        type=str, help='test set')
    parser.add_argument('--path_data',
                        default='/media/data1/doohyun/wsi/data/svs_feature_CTransPath_224_1.0_sn',
                        type=str, help='train set')
    parser.add_argument('--path_data_excel',
                        default='/media/data1/doohyun/wsi/data/excel',
                        type=str, help='')
    
    parser.add_argument('--save_path', type=str, default='./results', help='save path')
    
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    main_worker(args)