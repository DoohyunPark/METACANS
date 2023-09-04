import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import numpy as np
import time
import math
import random

def save_model(exp_dir, epoch, model, loss_tr, loss_val, AUC_tr, AUC_val):
    torch.save(model.state_dict(), f=f'{exp_dir}/{str(epoch).zfill(4)}_{loss_tr:.4f}_{loss_val:.4f}_{AUC_tr:.4f}_{AUC_val:.4f}.pt')
    
    
def min_model_finder(model_path):
    p = os.listdir(model_path)
    p.sort()
    p = p[:-2]
    losses = []
    for i,e in enumerate(p):
        p2 = e.split('_')
        losses.append([p2[2]])
    idx = losses.index(min(losses))
    selected_model = p[idx]
    return selected_model


def get_performance(label, pred):
    auc = roc_auc_score(label, pred)
    auc = np.round_(auc,4)
    return auc


def optimal_thresh(fpr, tpr, threshold):
    loss = (tpr - fpr)
    idx = np.argmax(loss, axis=0)
    return fpr[idx], tpr[idx], threshold[idx]


def train_model(args, model, data_loader, criterion, optimizer):
    ''' training '''
    time_start = time.time()
    model.train()
    pred_all = []
    label_all = []
    losses = 0
    ids = []
    for i, sample in enumerate(data_loader):
        inputs, clinics, label, id = sample
        inputs = inputs.squeeze()
        inputs = inputs.type(torch.float).cuda()
        clinics = clinics.type(torch.float).cuda()
        label = label.cuda()
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward to get output
        output = model(inputs)
        logit_prob, A = output
        logit_prob = logit_prob[0]
        A = A[0]
        
        # BCE Loss
        total_loss = criterion(logit_prob, label.float())
        total_loss.backward()

        # Updating parameters
        optimizer.step()

        m = nn.Sigmoid()
        y_pred      = m(logit_prob)
        y_pred      = y_pred.data.cpu().numpy()
        
        y_true      = label.cpu().numpy()
        
        losses      += total_loss.item()
        pred_all.append(y_pred[0])
        label_all.append(y_true[0])
        ids.append(id[0])
    auc             = get_performance(label_all, pred_all)
    losses          = np.round_(losses/len(data_loader),4)
    time_end        = time.time()
    print('Number of Cases: %d, w/o LNM: %d, w/ LNM: %d' % (len(label_all), len(label_all)-np.sum(label_all), np.sum(label_all)))
    print('\ntrain loss: %.4f, AUC: %.4f, time: %f' % (losses, auc, time_end-time_start))
    return losses, auc, model, ids


def test_model(args, model, data_loader, criterion):
    ''' testing '''
    time_start = time.time()
    model.eval()
    losses = 0
    pred_all = []
    label_all = []
    As = []
    ids = []
    for i, sample in enumerate(data_loader):
        inputs, clinics, label, id = sample
        inputs = inputs.squeeze()
        inputs = inputs.type(torch.float).cuda()
        clinics = clinics.type(torch.float).cuda()
        label = label.type(torch.float).cuda()

        # Forward to get output
        output = model(inputs)
        logit_prob, A = output
        logit_prob = logit_prob[0]
        # A = A[0]
        
        # BCE Loss
        total_loss = criterion(logit_prob, label)

        m = nn.Sigmoid()
        y_pred = m(logit_prob)
        y_pred = y_pred.data.cpu().numpy()
        
        y_true = label.cpu().numpy()
        
        losses      += total_loss.item()
        
        pred_all.append(y_pred[0])
        label_all.append(y_true[0])
        ids.append(id[0])
        As.append(A[0].data.cpu().numpy())
    auc             = get_performance(label_all, pred_all)
    losses          = np.round_(losses/len(data_loader),4)
    time_end        = time.time()
    
    print('Number of Cases: %d, w/o LNM: %d, w/ LNM: %d' % (len(label_all), len(label_all)-np.sum(label_all), np.sum(label_all)))
    print('\tvalidation loss: %.4f, AUC: %.4f, time: %f' % (losses, auc, time_end-time_start))
    return losses, auc, As, pred_all, label_all, ids


def adjust_learning_rate(args, epoch, optimizer, init_lr, warmup=10):
    """Decay the learning rate based on schedule"""
    if epoch >= warmup:
        epoch -= warmup
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr
                
    return optimizer



