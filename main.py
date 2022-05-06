from data import get_data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import mean_squared_error, mean_absolute_error#, mean_absolute_percentage_error 
from pytorch_forecasting.metrics import MAPE, MAE
from utils import plot_attn, plot_line
from model import STModel

def mean_absolute_percentage_error(labels, preds):
    mask = labels != 0
    return np.mean(np.abs((labels[mask] - preds[mask]) / labels[mask]))

def MAPELoss(output, target): #(predict, label)
    mask = target != 0
    return torch.mean(torch.abs((target[mask] - output[mask]) / target[mask]))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'infer'],\
        default='train',help='Run mode')
    arg_parser.add_argument('--epoch', default='100', type=int)
    arg_parser.add_argument('--batch_size', default='4096', type=int)
    arg_parser.add_argument('--lookback', default='24', type=int)
    arg_parser.add_argument('--path', default='model0.h5', type=str)
    arg_parser.add_argument('--layer', default=3, type=int)
    arg_parser.add_argument('--size', default=3, type=int)
    #arg_parser.add_argument('--loaddata', default=False, type=bool)
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    args = arg_parser.parse_args()

    print('====================')
    print('LCT')
    print('====================')
    device = 'cuda'
    matrix = np.load('matrix.npz')['matrix']
    lookback = args.lookback
    size = args.size
    X_train, X1_train, y_train, X_test, X1_test, y_test, xdow_train, xdow_test, gid_train, gid_test, tid_train, tid_test, nid_train, nid_test, scaler = get_data(matrix, lookback=lookback, size=size, LOAD=args.loaddata)

    X_train = X_train.to(device)
    X1_train = X1_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    X1_test = X1_test.to(device)
    y_test = y_test.to(device)
    xdow_train = xdow_train.to(device)
    xdow_test = xdow_test.to(device)
    gid_train = gid_train.to(device)
    gid_test = gid_test.to(device)
    nid_train = nid_train.to(device)
    nid_test = nid_test.to(device)

    print(X_train.shape) # 5205 * 278
    print(X1_train.shape)
    print(y_train.shape) # 5205 
    print(X_test.shape) #  2231 * 278
    print(X1_test.shape)
    print(y_test.shape) # 2231
    print(xdow_train.shape)
    print(xdow_test.shape)

    model_path = args.path
    #model_path = 'model_conv3_nta.h5'
    model = STModel(lookback=lookback, num_layers=args.layer, size=args.size)
    #model = torch.load(model_path)
    model.to(device)
    
    loss_function = nn.MSELoss()
    #loss_function = MAPELoss  #y_true has zero can not divide
    #loss_function = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    num_batch = int((len(y_train) + args.batch_size)/args.batch_size)
    num_batch_test = int((len(y_test) + args.batch_size)/args.batch_size)

    last_acc = 100.0
    last_loss = 100.0
    if args.mode == 'train':
        for epoch in range(args.epoch):
            model.to(device)
            acc = []
            total_loss = 0

            acc_test, mape_test, acc_test1 = [], [], []
            Ybar0, Ylabel0 = [], []
            for i in range(num_batch):
                sys.stdout.write('\r{0}/{1}'.format(i, num_batch))
                st = i * args.batch_size
                ed = min((i+1) * args.batch_size, len(y_train))
                if st == ed:
                    break

                x = X_train[st:ed]
                x = x.to(device)

                x1 = X1_train[st:ed]
                x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))
                x1 = x1.to(device)

                xdow = xdow_train[st:ed]
                xdow = xdow.to(device)

                gid = gid_train[st:ed]
                gid = gid.to(device)

                tid = tid_train[st:ed]
                tid = tid.to(device)
    
                nid = nid_train[st:ed]
                nid = nid.to(device)                

                label = y_train[st:ed]
                label = label.to(device)

                model.zero_grad()

                output  = model(x, x1, xdow, gid, tid, nid)

                predict = torch.squeeze(output)
                _pred = scaler.inverse_transform(predict.cpu().detach().numpy().reshape(-1, 1))
                _gold = scaler.inverse_transform(label.cpu().detach().numpy().reshape(-1, 1))
                Ybar0.extend(list(_pred.flatten()))
                Ylabel0.extend(list(_gold.flatten()))
                loss = loss_function(predict, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch%10 == 0 or epoch == args.epoch - 1:
                Ybar, Ylabel = [], []
                for i in range(num_batch_test):
                    st = i * args.batch_size
                    ed = min((i+1) * args.batch_size, len(y_test))

                    if st == ed:
                        break
                    x_test = X_test[st:ed]
                    x_test = x_test.to(device)

                    x1_test = X1_test[st:ed]
                    x1_test = torch.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1], 1))
                    x1_test = x1_test.to(device)

                    xdow = xdow_test[st:ed]
                    xdow = xdow.to(device)

                    gid = gid_test[st:ed]
                    gid = gid.to(device)

                    tid = tid_test[st:ed]
                    tid = tid.to(device)

                    nid = nid_test[st:ed]
                    nid = nid.to(device)

                    label_test = y_test[st:ed]
                    label_test = label_test.to(device)

                    output_test = model(x_test, x1_test, xdow, gid, tid, nid)
                    predict_test = torch.squeeze(output_test)
                    _pred = scaler.inverse_transform(predict_test.cpu().detach().numpy().reshape(-1, 1))
                    _gold = scaler.inverse_transform(label_test.cpu().detach().numpy().reshape(-1, 1))
                    Ybar.extend(list(_pred.flatten()))
                    Ylabel.extend(list(_gold.flatten()))
                    
                Ybar = np.asarray(Ybar)
                Ylabel = np.asarray(Ylabel)            
            
                print('\nEpoch: ', epoch)
                print('\nTraining set: Loss {0:.4f} RMSE {1:.4f}.\nTest: RMSE {2:.4f} MAE {3:.4f} MAPE {4:.4f}.'.format(total_loss,  mean_squared_error(Ybar0, Ylabel0, squared=False),  mean_squared_error(Ybar, Ylabel, squared=False), mean_absolute_error(Ybar, Ylabel), mean_absolute_percentage_error(Ylabel, Ybar)))
                print(_pred.flatten()[:10])
                if total_loss < last_loss:
                    torch.save(model.state_dict(), model_path)
                    #torch.save(model.cpu(), model_path)
                    last_loss = total_loss
                    print('Model saved!')
            scheduler.step()

        print(Ylabel[:10])
        print(Ybar[:10])
    
    elif args.mode == 'infer':
        model = STModel(lookback=lookback, num_layers=args.layer, size=args.size)
        #model_path = 'h5file1/model_conv3_nta_new.h5'
        model_path = args.path
        model.load_state_dict(torch.load(model_path), strict=False)
        #model = torch.load(model_path)
        model.to(device)
        Ybar, Ylabel = [], []
        for i in range(num_batch_test):
            st = i * args.batch_size
            ed = min((i+1) * args.batch_size, len(y_test))

            if st == ed:
                break
            x_test = X_test[st:ed]
            x_test = x_test.to(device)

            x1_test = X1_test[st:ed]
            x1_test = torch.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1], 1))
            x1_test = x1_test.to(device)

            xdow = xdow_test[st:ed]
            xdow = xdow.to(device)

            gid = gid_test[st:ed]
            gid = gid.to(device)

            tid = tid_test[st:ed]
            tid = tid.to(device)

            nid = nid_test[st:ed]
            nid = nid.to(device)

            label_test = y_test[st:ed]
            label_test = label_test.to(device)

            output_test = model(x_test, x1_test, xdow, gid, tid, nid)

           
            predict_test = torch.squeeze(output_test)
            _pred = scaler.inverse_transform(predict_test.cpu().detach().numpy().reshape(-1, 1))
            _gold = scaler.inverse_transform(label_test.cpu().detach().numpy().reshape(-1, 1))
            Ybar.extend(list(_pred.flatten()))
            Ylabel.extend(list(_gold.flatten()))
       
        Ybar = np.asarray(Ybar)
        Ylabel = np.asarray(Ylabel)
        print('[No filter]')
        print('Test set: MAPE: {0:.4f}. RMSE: {1:.4f}. MAE: {2:.4f}.'.format(mean_absolute_percentage_error(Ylabel, Ybar), mean_squared_error(Ylabel, Ybar, squared=False), mean_absolute_error(Ybar, Ylabel)))
        Ybar = Ybar[Ylabel>5]
        Ylabel = Ylabel[Ylabel>5]
        print('[Filter > 5]')
        print('Test set: MAPE: {0:.4f}. RMSE: {1:.4f}. MAE: {2:.4f}.'.format(mean_absolute_percentage_error(Ylabel, Ybar), mean_squared_error(Ylabel, Ybar, squared=False), mean_absolute_error(Ybar, Ylabel)))
        Ybar = Ybar[Ylabel>10]
        Ylabel = Ylabel[Ylabel>10]
        print('[Filer > 10]')
        print('Test set: MAPE: {0:.4f}. RMSE: {1:.4f}. MAE: {2:.4f}.'.format(mean_absolute_percentage_error(Ylabel, Ybar), mean_squared_error(Ylabel, Ybar, squared=False), mean_absolute_error(Ybar, Ylabel)))
