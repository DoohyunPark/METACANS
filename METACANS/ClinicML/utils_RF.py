import os
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import *
import matplotlib.pyplot as plt
import argparse

def make_df(path_data, path_id, hospital):
    if hospital[0:2] == 'ss':
        df = pd.read_excel(path_data + '/' + hospital[0:2] + '.xlsx', engine='openpyxl')
    else:
        df = pd.read_excel(path_data + '/' + hospital + '.xlsx', engine='openpyxl')
    
    ids = []
    with open(path_id + '/' + hospital + '/ids.txt', 'r') as file:
        for i, line in enumerate(file):
            if i == len(ids):
                ids.append(line.strip())

    if hospital=='ewha':
        ids = ['{}_{}_{:05d}'.format(x.split('_')[0],x.split('_')[1],int(x.split('_')[2])) for x in ids]
    elif hospital=='dk':
        ids = ['{}_{}_{}'.format(x.split('_')[0],x.split('_')[1],int(x.split('_')[2])) for x in ids]

    df = df[df['ID'].isin(ids)].drop_duplicates(subset='ID', keep='first')
    df = df.replace({"'": ""}, regex=True)
    
    df['LABEL'].replace(1, 1, inplace=True)
    df['LABEL'].replace(2, 1, inplace=True)
    df['LABEL'].replace(3, 1, inplace=True)

    df['N category'].replace(1, 1, inplace=True)
    df['N category'].replace(2, 1, inplace=True)
    df['N category'].replace(3, 1, inplace=True)

    df['NG'].replace('1', 1, inplace=True)
    df['NG'].replace('2', 2, inplace=True)
    df['NG'].replace('3', 3, inplace=True)
    
    df['HG'].replace('1', 1, inplace=True)
    df['HG'].replace('2', 2, inplace=True)
    df['HG'].replace('3', 3, inplace=True)

    df = df.applymap(lambda x: 0 if isinstance(x, str) else x)
    df.fillna(0, inplace=True)
    
    df['AGE']   = df['AGE'].apply(lambda x: 1 if x >= 55 else 0)
    df['ER']    = df['ER'].apply(lambda x: 1 if x == 1 else 0)
    df['PR']    = df['PR'].apply(lambda x: 1 if x == 1 else 0)
    
    feature_names =[
        'AGE',
        'NUM CANCER',
        'SIZE',
        # 'NG',
        # 'HG',
        # 'HG1',
        # 'HG2',
        # 'HG3',
        # 'DCIS',
        # 'KI-67 LI',
        # 'ER',
        # 'PR',
        # 'HER2'
        ]
    X = df[feature_names]
    y = df['N category']

    return [X, y, feature_names]


def test_and_print(X_test, y_test, hospital, RFC):
    y_pred = RFC.predict(X_test)
    y_prob = RFC.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    save_dir = '/media/data1/doohyun/wsi/code/RF/results_230728/' + hospital
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(save_dir + '/pred.txt','w')
    for name in y_prob:
        f.write(str(name)+'\n')
    f.close()
    f = open(save_dir + '/label.txt','w')
    for name in y_test:
        f.write(str(name)+'\n')
    f.close()
    ROC_curve(y_test, y_prob, save_dir)

def ROC_curve(label1,score1, save_path):
    fpr1, tpr1, thresholds1 = roc_curve(label1, score1)
    auc1 = auc(fpr1, tpr1)
    
    plt.figure(figsize=(5,5), dpi= 600)
    plt.plot(fpr1, tpr1, color='#FC5A50', linestyle='-', label=r'AUC=%.2f' %auc1)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ALNM prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.savefig(save_path + '/ROC.png', dpi=600)