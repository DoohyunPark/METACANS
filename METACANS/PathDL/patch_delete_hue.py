# sudo apt-get install openslide-tools
# sudo apt-get install python-openslide
# pip install openslide-python
import sys
import argparse
import openslide
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import shutil
import pandas as pd
    
def main_worker(args):
    test_path = args.test_path + '_' + str(args.target_mpp) + '/' + args.hospital
    test_path_ = os.listdir(test_path)
    test_path_.sort()
    for i, item1 in enumerate(test_path_):
        # if item1 == 'BC_02_0001':
        test_path_2 = os.listdir(test_path + '/' + item1)
        test_path_2.sort()
        for j, item2 in enumerate(test_path_2):
            path_patch = test_path + '/' + item1 + '/' + item2
            img = cv2.imread(path_patch)
            
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            
            if np.mean(h) <= 70:
                os.remove(path_patch)
            # print(item2, np.mean(h))
    

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', default=224, type=int, help='')
    parser.add_argument('--gray_filter_tol', default=8, type=int, help='')
    parser.add_argument('--tissue_ratio_threshold', default=50, type= int, help='')
    parser.add_argument('--target_mpp', default=1.0, type=float, help='') # for 20X, 0.25 for 40X
    
    parser.add_argument('--hospital', default='gs', type=str, help='')
    parser.add_argument('--output_path', default='/media/data1/doohyun/wsi/data/new/svs_patch_224', type=str, help='')
    parser.add_argument('--test_path', default='/media/data1/doohyun/wsi/data/svs_patch_224', type=str, help='')
    parser.add_argument('--wsi_folder', default='/media/hdd_svs/svs', type=str, help='')
    
    parser.add_argument('--path_data_excel',
                    default='/media/data1/doohyun/wsi/data/excel',
                    type=str, help='')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main_worker(args)