from sklearn.preprocessing import MinMaxScaler
import torch
import math
import numpy as np
import pandas as pd
import datetime
import os
import json
import data
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import load, dump

"""
Change local region range
"""

gridI = 20
gridJ = 20



def get_data(matrix, lookback=24, size=3, LOAD=True):
    prefix = str(lookback) + '_' + str(size) + '_'
  
    if LOAD:
        print('==============')
        print('Load data')
        print('Lookback: ' + str(lookback))
        print('==============')
        scaler = load(open(prefix + 'scaler.pkl', 'rb'))
        dat = np.load(prefix + 'data.npz')
        x_train = dat['x_train']
        x1_train = dat['x1_train']
        y_train = dat['y_train']
        x_test = dat['x_test']
        x1_test = dat['x1_test']
        y_test = dat['y_test']
        xdow_train = dat['xdow_train']
        xdow_test = dat['xdow_test']
        gid_train = dat['gid_train']
        gid_test = dat['gid_test']
        tid_train = dat['tid_train']
        tid_test = dat['tid_test']
        nid_train = dat['nid_train']
        nid_test = dat['nid_test']
   
    print(x_train.shape)
    print(x1_train.shape)
    print(y_train.shape)
    print(xdow_train.shape)
    print(gid_train.shape)
    print(nid_train.shape)
    print(x_test.shape)
    print(x1_test.shape)
    print(y_test.shape)
    print(xdow_test.shape)
    print(gid_test.shape)
    print(nid_test.shape)

    if not LOAD:
        np.savez(prefix + 'data', x_train=x_train, x1_train=x1_train, y_train=y_train, x_test=x_test, x1_test=x1_test, y_test=y_test, xdow_train=xdow_train, xdow_test=xdow_test, gid_train=gid_train, gid_test=gid_test, tid_train=tid_train, tid_test=tid_test, nid_train=nid_train, nid_test=nid_test)
    
    x_train = torch.FloatTensor(x_train)
    x1_train = torch.FloatTensor(x1_train)
    y_train = torch.FloatTensor(y_train).squeeze()
    x_test = torch.FloatTensor(x_test)
    x1_test = torch.FloatTensor(x1_test)
    y_test = torch.FloatTensor(y_test).squeeze()
    xdow_train = torch.IntTensor(xdow_train)
    xdow_test = torch.IntTensor(xdow_test)
    gid_train = torch.LongTensor(gid_train)
    gid_test = torch.LongTensor(gid_test)
    tid_train = torch.LongTensor(tid_train)
    tid_test = torch.LongTensor(tid_test)
    nid_train = torch.LongTensor(nid_train)
    nid_test = torch.LongTensor(nid_test)
    return  x_train, x1_train, y_train, x_test, x1_test, y_test, xdow_train, xdow_test, gid_train, gid_test, tid_train, tid_test, nid_train, nid_test, scaler 


if __name__ == '__main__':

    matrix = np.load('matrix.npz')['matrix']    
    get_data(matrix, lookback=24, size=3, LOAD=True)
 
