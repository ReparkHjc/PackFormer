import torch.nn as nn
import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
# import xgboost as xgb

from math import sqrt
from datetime import datetime
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV


class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, sequencen_len=20,n_class=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*sequencen_len, n_class)

        self.CNN=nn.Conv2d(in_channels=3,out_channels=1,stride=(1,1),padding=1,kernel_size=(3,3))
        # self.CNN1 = nn.Conv2d(in_channels=3, out_channels=1, stride=1, padding=1, kernel_size=(3, 3))
        self.flatten=nn.Flatten(1,2)
    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        cell1 = x[:, 0]
        cell2 = x[:, 1]
        cell3 = x[:, 2]

        batchsize=x.shape[0]

        output1 = self.CNN(cell1)
        output2 = self.CNN(cell2)
        output3 = self.CNN(cell3)

        cell_all = torch.cat([output1, output2, output3], dim=1)

        output = self.CNN(cell_all).squeeze()

        x = output.reshape(batchsize, 20, -1)
        out, _ = self.cell(x)
        out = self.flatten(out)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        return out


class CNN(nn.Module):
    def __init__(self,n_class=1):
        super(CNN, self).__init__()

        self.CNN = nn.Conv2d(in_channels=3, out_channels=1, stride=(1,1), padding=1, kernel_size=(3, 3))
        self.CNN1 = nn.Conv2d(in_channels=3, out_channels=1, stride=(1,1), padding=1, kernel_size=(3, 3))
        self.flatten=nn.Flatten(1,3)
        # 如果四预1，28*28，20预测1 60*60
        self.linear = nn.Linear(60*60,100)
        self.linear1 = nn.Linear(100, n_class)
    def forward(self,x):
        cell1 = x[:, 0]
        cell2 = x[:, 1]
        cell3 = x[:, 2]

        batchsize=x.shape[0]

        output1 = self.CNN(cell1)
        output2 = self.CNN(cell2)
        output3 = self.CNN(cell3)

        cell_all = torch.cat([output1, output2, output3], dim=1)

        output = self.CNN(cell_all)

        out = self.flatten(output)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        out=self.linear1(out)
        return out



class LSTM(nn.Module):
    # 如果四预1，1764，20预测1 1620
    def __init__(self, input_size=1620, hidden_dim=25, num_layers=3, sequencen_len=20, n_class=1, mode='RNN'):
        super(LSTM, self).__init__()
        self.flatten=nn.Flatten(2,4)
        self.flatten1 = nn.Flatten(1, 2)
        self.sequencen_len=sequencen_len
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim * sequencen_len, n_class)

    def forward(self,x):

        x=self.flatten(x)
        batchsize=x.shape[0]
        x = x.reshape(batchsize, self.sequencen_len, -1)

        x,_ = self.cell(x)

        x=self.flatten1(x)

        out = self.linear(x)  # out shape: (batch_size, n_class=1)

        return out


class MLP(torch.nn.Module):
    def __init__(self, n_feature=10800, n_hidden=33, n_class=1):
        # 四预测1：2352, 33, 1
        super(MLP, self).__init__()
        # 两层感知机
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # self.hidden1=nn.Linear(99,1)
        self.predict = torch.nn.Linear(99, n_class)
        self.flatten=nn.Flatten(2,4)
        self.flatten1=nn.Flatten(1,2)
    def forward(self, x):
        x=self.flatten(x)
        x = F.relu(self.hidden(x))
        x = self.flatten1(x)
        x = self.predict(x)
        return x


# XGBOST机器学习方法，GP

# model = MLP(2352,33,1)
# input = torch.rand(8,3,3,28,28)
# model(input)


# class m_learn():
#     model_seed = 100
#     parameters = {'n_estimators': [90],
#                   'max_depth': [7],
#                   'learning_rate': [0.3],
#                   'min_child_weight': range(5, 21, 1),
#                   # 'subsample':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                   # 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                   # 'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                   # 'colsample_bylevel':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
#                   }
#     # parameters={'max_depth':range(2,10,1)}
#     model = xgb.XGBRegressor(seed=model_seed,
#                              n_estimators=100,
#                              max_depth=3,
#                              eval_metric='mse',
#                              learning_rate=0.1,
#                              min_child_weight=1,
#                              subsample=1,
#                              colsample_bytree=1,
#                              colsample_bylevel=1,
#                              gamma=0)
#     gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, refit=True, scoring='neg_mean_squared_error')
