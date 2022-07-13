from sklearn.preprocessing import MinMaxScaler
import torch
import math
import numpy as np
import pandas as pd
import datetime
import os
import json
from utils import check_nta
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

def load_adjmatrix(load=False):
    if load:
        return np.load('adj_matrix.npz')['arr_0']
    adj_matrix = np.zeros([gridI*gridJ, gridI*gridJ])
    for m in range(gridI*gridJ):
        for n in range(gridI*gridJ):
            if m == n:
                continue
            i1 = int(m/gridJ)
            j1 = m%gridJ
            i2 = int(n/gridJ)
            j2 = n%gridJ
            val = math.sqrt((i1-i2)*(i1-i2) + (j1-j2)*(j1-j2))
            adj_matrix[m][n] = val
    adj_matrix = torch.FloatTensor(adj_matrix)
    np.savez('adj_matrix.npz', adj_matrix)
    return adj_matrix

def get_features():
    NTA_list = ['null', 'BK88', 'QN51', 'QN27', 'BK23', 'QN41', 'QN08', 'BK69', 'BX10', 'BK26', 'BK46', 'BX55', 'QN19', 'BX28', 'BX26', 'QN21', 'QN55', 'BX52', 'BX41', 'BK40', 'BK50', 'BX09', 'SI32', 'MN21', 'MN12', 'BX43', 'BX05', 'BK34', 'QN42', 'QN26', 'QN18', 'QN63', 'BK68', 'BK93', 'QN35', 'BX44', 'QN49', 'QN56', 'QN03', 'QN02', 'QN66', 'BX33', 'BK79', 'BX34', 'BK60', 'QN62', 'BX35', 'BX75', 'QN37', 'QN38', 'QN52', 'BX08', 'BX30', 'BX36', 'BK72', 'BX22', 'BK44', 'QN07', 'QN05', 'QN34', 'SI08', 'QN01', 'QN76', 'BK95', 'BX29', 'QN72', 'QN54', 'BX98', 'QN48', 'QN70', 'QN06', 'BK77', 'QN61', 'QN25', 'QN20', 'QN30', 'QN29', 'QN50', 'BK35', 'BX06', 'BX17', 'BK90', 'MN17', 'BK41', 'BX01', 'BX40', 'MN35', 'MN13', 'MN15', 'MN32', 'MN01', 'BK75', 'QN47', 'BK78', 'MN09', 'SI37', 'QN45', 'MN34', 'MN11', 'QN22', 'QN17', 'BK63', 'QN33', 'QN46', 'SI35', 'QN43', 'BK85', 'QN44', 'QN28', 'BK43', 'MN27', 'QN71', 'BK61', 'BK81', 'BK91', 'BK96', 'BK58', 'BX46', 'MN24', 'MN25', 'BK32', 'MN20', 'MN50', 'QN15', 'QN23', 'BK09', 'BK38', 'QN98', 'BK73', 'MN28', 'MN31', 'QN68', 'SI11', 'SI01', 'SI54', 'SI36', 'SI14', 'SI45', 'MN33', 'SI48', 'BK45', 'BK27', 'BK28', 'BK29', 'SI25', 'SI99', 'MN06', 'BX27', 'BX39', 'BX07', 'QN60', 'MN14', 'BK17', 'BK64', 'BK30', 'BK31', 'BK33', 'BK37', 'BK19', 'BK25', 'BK21', 'SI12', 'SI28', 'QN53', 'SI07', 'SI22', 'MN19', 'BX62', 'BX99', 'BK82', 'QN57', 'SI24', 'BX31', 'BK76', 'QN31', 'MN36', 'MN40', 'MN99', 'BX37', 'BX59', 'SI05', 'BX03', 'BX13', 'MN03', 'MN04', 'BX49', 'BK83', 'QN99', 'BK42', 'BK99', 'QN10', 'QN12', 'MN22', 'MN23', 'BX14', 'BX63']

    emb_n = np.load('emb_n.npz')['emb_n'] # len(NTA_list), embedding dim
    nycmap = json.load(open("yellow-taxi-2014/nyc-neighborhood.geojson"))
    data = np.load('rowcol.npz')
    rows = data['rows']
    cols = data['cols']

    #print(cols)
    #print(rows)

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df = pd.read_csv('yellow-taxi-2014/econ_nta.csv')
    mmap = {}
    df = df[['GeoID', 'Pop16plE', 'CvEm16pl1E', 'CW_DrvAlnE', 'CW_CrpldE', 'CW_PbTrnsE', 'CW_WlkdE', 'CW_WrkdHmE', 'MdHHIncE', 'MnHHIncE', 'MnTrvTmE']]
    for i in range(len(df)):
        NTA = df.iloc[i]['GeoID']
        #NTA = np.array(df)[i][0]
        mmap[NTA] = np.array(df)[i][1:]
        #tmp = df.iloc[i]['Pop16plE', 'CvEm16pl1E', 'CW_DrvAln', 'CW_Crpld', 'CW_PbTrns', 'CW_Wlkd', 'CW_WrkdHm', 'MdHHInc', 'MnHHInc', 'MnTrvTm']
    #res = np.zeros([gridI*gridJ, 10])
    res1 = np.zeros([gridI*gridJ, 32])
    res = np.zeros([gridI*gridJ, 10])
    NTA_grid = np.zeros(gridI*gridJ, dtype=np.int16) 
    names = np.repeat('null', gridI*gridJ)
    for i in range(gridI):
        for j in range(gridJ):
            tmp = []
            #print('Grid '+ str(i) + ' ' + str(j))
            # four corner
            lat = rows[i]
            lon = cols[j]
            a = check_nta(lon, lat, nycmap)
            if a != 'null':
                tmp.append(a)
            lat = rows[i+1]
            lon = cols[j]
            b = check_nta(lon, lat, nycmap)
            if b != 'null':
                tmp.append(b)
            lat = rows[i]
            lon = cols[j+1]
            c = check_nta(lon, lat, nycmap)
            if c != 'null':
                tmp.append(c)
            lat = rows[i+1]
            lon = cols[j+1]
            d = check_nta(lon, lat, nycmap)
            if d != 'null':
                tmp.append(d)
            #res[i*gridJ + j] = str(a)+str(b)+str(c)+str(d)
            if len(tmp) == 0:
                res[i*gridJ + j] = np.zeros(10)
                res1[i*gridJ + j] = np.zeros(32)
                NTA_grid[i*gridJ + j] = 0
            else:
                NTA = max(tmp, key = tmp.count)
                res[i*gridJ + j] = mmap[NTA]
                res1[i*gridJ + j] = emb_n[NTA_list.index(NTA)]
                names[i*gridJ + j] = NTA
                NTA_grid[i*gridJ + j] = NTA_list.index(NTA)
            #print(str(a)+str(b)+str(c)+str(d))
    
    res = pd.DataFrame(data=res, columns=np.arange(10))
    for i in range(10):
        res[i] = res[i].fillna(np.mean(res[i]))
    res = scaler.fit_transform(res)
    #print(names)
   
    return res, res1, NTA_grid

def get_data(matrix, lookback=24, size=3, LOAD=True):
    prefix = str(lookback) + '_' + str(size) + '_'
    res, res1, NTA_grid = get_features()
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
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))


        matrix[matrix > 500] = 500
        mmax = np.max(matrix)
        m, d, t, I, J = matrix.shape
        PAD = int((size-1)/2)
        print('PAD: ' + str(PAD))
        pad_matrix = np.zeros((m, d, t, I+int(size-1), J+int(size-1)))
        pad_matrix[:, :, :, PAD:I+PAD, PAD:J+PAD] = matrix
        print(mmax)
        print(np.sum(matrix>500))
        print(np.sum(matrix<=500))
        #matrix /= np.max(matrix)
        #scaler = MinMaxScaler(feature_range=(-1, 1)) 
        #matrix = scaler.fit_transform(matrix)
        pre_matrix, pre_pad_matrix = None, None
        interval = 1
        print('======================')
        print('Create data')
        print('lookback ' + str(lookback))
        print('======================')
        x, y, xdow, gid, tid, x1, nid = None, None, None, None, None, None, None
        for month in range(1,13):
            if np.sum(matrix[month,:,:,:])==0:
                continue
            #print(month)
            for day in range(32):
                if np.sum(matrix[month,day,:,:])==0:
                    continue
                
                #print('-' + str(day))
                # In each day
                if pre_matrix is not None:
                    _matrix = np.concatenate((pre_matrix, matrix[month, day]), axis=0)
                    _pad_matrix = np.concatenate((pre_pad_matrix, pad_matrix[month, day]), axis=0)
                    st = int(24/interval) - lookback + int(8/interval) #start from 8AM
                    ed = int(24/interval) * 2 - lookback
                else:
                    _matrix = matrix[month, day]
                    _pad_matrix = pad_matrix[month, day]
                    st = 0
                    ed = int(24/interval) - lookback
                #for start in range(int(24/interval) - lookback):
                for start in range(st, ed):
                    dow = datetime.date(2014, month, day).weekday() #Monday 0 Sunday 6
                    x_tmp = np.zeros((gridI*gridJ, lookback, size, size))
                    for i in range(PAD, gridI+PAD):
                        for j in range(PAD, gridJ+PAD):
                            x_tmp[(i-PAD)*(j-PAD), :, :, :] = _pad_matrix[start:start+lookback, i-PAD:i+1+PAD, j-PAD:j+1+PAD]
                    y_tmp = _matrix[start+lookback, :, :].reshape(gridI*gridJ, 1)
                    x1_tmp = _matrix[start:start+lookback, :, :].reshape([lookback, gridI*gridJ])
                    x1_tmp = np.swapaxes(x1_tmp, 0, 1) 
                    
                    IDX = np.ones(len(y_tmp), dtype=bool)
                    #IDX[IDX0] = False
                    IDX[y_tmp.flatten() < 5] = False
                    #idxs = (np.arange(len(y_tmp))[y_tmp.flatten() < 5])
                    #print(','.join(str(x) for x in idxs))
                    
                    gid_tmp = np.arange(gridI*gridJ).reshape([gridI*gridJ,1])
                    tid_tmp = np.ones([gridI*gridJ, 1]) * month
                    dow_tmp = np.ones([gridI*gridJ, 1]) * dow
                    nid_tmp = NTA_grid.reshape([gridI*gridJ,1])
                    
                    x_tmp = x_tmp[IDX]
                    x1_tmp = x1_tmp[IDX]
                    y_tmp = y_tmp[IDX]
                    gid_tmp = gid_tmp[IDX]
                    tid_tmp = tid_tmp[IDX]
                    dow_tmp = dow_tmp[IDX]
                    nid_tmp = nid_tmp[IDX]
                    if x is None:
                        x = x_tmp
                        x1 = x1_tmp
                        y = y_tmp
                        xdow = dow_tmp
                        gid = gid_tmp
                        tid = tid_tmp
                        nid = nid_tmp
                    else:
                        x = np.concatenate((x, x_tmp), axis=0)
                        x1 = np.concatenate((x1, x1_tmp), axis=0)
                        y = np.concatenate((y, y_tmp), axis=0)
                        xdow = np.concatenate((xdow, dow_tmp), axis=0) 
                        gid = np.concatenate((gid, gid_tmp), axis=0)
                        tid = np.concatenate((tid, tid_tmp), axis=0)
                        nid = np.concatenate((nid, nid_tmp), axis=0)
                pre_matrix = matrix[month,day,:,:,:]  
                pre_pad_matrix = pad_matrix[month,day,:,:,:]       
        
        cut = int(y.shape[0] * 0.7)
        y = y.reshape(y.shape[0])
        xdow = xdow.reshape(xdow.shape[0])
        gid = gid.reshape(gid.shape[0])
        tid = tid.reshape(tid.shape[0])
        nid = nid.reshape(nid.shape[0])
        x_train, x1_train, y_train, xdow_train, gid_train, tid_train, nid_train = x[:cut], x1[:cut], y[:cut], xdow[:cut], gid[:cut], tid[:cut], nid[:cut]
        
        idx = np.arange(cut)
        random.seed(888)
        random.shuffle(idx)
        idx = idx[:int(len(idx) * 0.2)]
        
        #x_train, x1_train, y_train, xdow_train, gid_train, tid_train = x[idx], x1[idx], y[idx], xdow[idx], gid[idx], tid[idx]
        cut1 = int(y.shape[0])
        x_test, x1_test, y_test, xdow_test, gid_test, tid_test, nid_test = x[cut:cut1], x1[cut:cut1], y[cut:cut1], xdow[cut:cut1], gid[cut:cut1], tid[cut:cut1], nid[cut:cut1]
        
        scaler.fit(x_train.reshape(-1, 1))
        dump(scaler, open(prefix + 'scaler.pkl', 'wb'))
        x_train = scaler.transform(x_train.reshape(-1,1)).reshape(x_train.shape)
        x1_train = scaler.transform(x1_train.reshape(-1,1)).reshape(x1_train.shape)
        x_test = scaler.transform(x_test.reshape(-1,1)).reshape(x_test.shape)
        x1_test = scaler.transform(x1_test.reshape(-1,1)).reshape(x1_test.shape)
        y_train, y_test = scaler.transform(y_train.reshape(-1,1)).reshape(y_train.shape), scaler.transform(y_test.reshape(-1,1)).reshape(y_test.shape)
            
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

def plot(matrix):
    for i in range(24):
        plt.figure(i)
        #a = matrix[4, 2, i, 9:12, 8:11]
        a = matrix[4, 2, i]
        ax = sns.heatmap(a, linewidth=0.5)
        #plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.savefig('1-' + str(i) + '.png')
        plt.close()
    
    for i in range(24):
        plt.figure(i)
        #a = matrix[4, 3, i, 9:12, 8:11]
        a = matrix[4, 3, i]
        ax = sns.heatmap(a, linewidth=0.5)
        #plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.savefig('2-' + str(i) + '.png')
        plt.close()

if __name__ == '__main__':
    
    #_, _, NTA_grid = get_features()
    #print(NTA_grid)
    #print(res.shape)
    #print(np.repeat(np.arange(len(res)),10).reshape(len(res),10)[res!=res])
    #print(res[45,:])
    #print(res[46,:])
    pass
    #load_adjmatrix(load=False)
    matrix = np.load('matrix.npz')['matrix']
    #plot(matrix)
    get_data(matrix, lookback=24, size=3, LOAD=False)
    #mat = load_adjmatrix(load=True)
    #print(mat)
    #print(mat.shape)
