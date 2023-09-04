import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import gzip, pickle
import time
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

def dataloader(path_data, path_excel, hospital, batch_size, num_workers, val_fold, mode):
    data_set1 = generator(path_data, path_excel, hospital, val_fold, mode)
    loader1 = DataLoader(dataset=data_set1,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         drop_last=False,
                         pin_memory=False)
    return loader1

class generator(Dataset):
    def __init__(self, path_data, path_excel, hospital, val_fold, mode):
        self.hospital       = hospital
        self.path_data      = path_data
        self.path_data      = self.path_data + '/' + self.hospital
        self.pathes_data    = os.listdir(self.path_data)
        self.pathes_data.sort()
        self.path_excel     = path_excel + '/' + self.hospital + '.xlsx'
        self.val_fold       = val_fold
        self.mode           = mode
        
        # train, validation split
        if self.hospital == 'ss_tr':
            kf = KFold(n_splits=8, shuffle=True, random_state=42)
            folds = []
            for _, fold in kf.split(self.pathes_data):
                folds.append([self.pathes_data[i] for i in fold])
            
            pathes_data_valid = folds.pop(self.val_fold)
            pathes_data_train = [item for sublist in folds for item in sublist]
            if mode == 'train':
                self.pathes_data = pathes_data_train
            else:
                self.pathes_data = pathes_data_valid
                

        self.datalist = []
        df, ids             = self.getexcel(self.path_excel, self.hospital)
        for i,item in enumerate(self.pathes_data):
            time_start  = time.time()
            features, clinic, label = self.getdata(self.path_data, df, ids, item)
            self.datalist.append([features, clinic, label, item[0:10]])
            time_end = time.time()
            print(i, item, time_end-time_start)   


    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, idx):
        features, clinic, label, item = self.datalist[idx]
        return features, clinic, label, item

    def getexcel(self, path_excel, hospital):
        usecols = [2,4,5,7,8, 9,10,11,12,13, 14,15,16,17,18, 19,20,21,22,23,24,25,29,33,34] #16: DCIS, 18: T-category
        
        df          = pd.read_excel(path_excel, sheet_name='data', index_col=None, usecols=usecols, engine='openpyxl')
        df          = df.to_numpy()
        ids         = []
        if hospital[0:2] == 'dk':
            for i,e in enumerate(df):
                id = e[0][0:6] + e[0][6::].zfill(4)
                ids.append(id)
        elif hospital[0:4] == 'ewha':
            for i,e in enumerate(df):
                id = e[0][:-5] + e[0][-4:]
                ids.append(id)
        else:
            for i,e in enumerate(df):
                id = e[0]
                ids.append(id)
        return df, ids
    
        
    def getdata(self, path_data, df, ids, item):
        feature = np.load(path_data + '/' + item)
        idx     = ids.index(item[0:10])
        label   = int(df[idx][18]!=0)
        clinic = label
        return feature, clinic, label


    def data_to_tensor(self, data):
        [x, y] = data.shape
        new_data = np.reshape(data, [1, x, y])
        return new_data