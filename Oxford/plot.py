import torch
from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net,CNN,LSTM,MLP
from data_preprocess import train_val_dataset_sw2
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np

# Model = "VIT"
# epoch = 3300

Model = "CNN_LSTM"
# epoch = 1600

epoch = 300


# Model = "MLP"
# epoch = 1000

# Model = "CNN"
# epoch = 1500

# Model = "LSTM"
# epoch = 1500

# epoch = 700

train_radio = 0.8

# path=r"C:\Users\xzh\Desktop\SOH1\NASA_196_4_1\Model\CNN_LSTM\shuffle_True\0.8_params\epoch_10500_train.params"
path1='./Oxford_20_4/Model/' + Model + '/shuffle_True/' + str(
    train_radio) + '_params/epoch_' + str(
    epoch) + '.params'


# path1="./Oxford_20_4/Model/CNN/shuffle_True/0.8_params/epoch_5450_train.params"

# B0005
# SOH_path = r"C:\Users\xzh\Desktop\SOH1\result_rubbish\NASA_196_4_1\data\SOH.csv"
# data_path = r"C:\Users\xzh\Desktop\SOH1\result_rubbish\NASA_196_4_1\data\data.csv"
data_path="./data/5cell/data.csv"
SOH_path="./data/5cell/SOH.csv"

num=path1.split("/")
shuffle=num[0]
if shuffle=="shuffle_True":
    Shuffle=True
else:
    Shuffle=False
MODEL=path1.split("/")[0]


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def RMSE(y_true, y_pred):
    return np.linalg.norm(y_true-y_pred, ord=2)/len(y_true)**0.5

def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """
    n = len(y_true)
    mse = (np.square(y_true - y_pred))/ n
    sum=mse.sum()
    return sum**0.5

# 加载俩模型对比

if Model == "VIT":
    model = SE_VIT_Decoder(type=1, embed_dim=500, attn_drop_ratio=0, prediction_num=4).cuda()

if Model == "CNN_LSTM":
    # CNN_LSTM.RNN,CGU

    model = Net(cell_num=5, input_size=180, hidden_dim=25, num_layers=3, n_class=4).cuda()

if Model == "MLP":
    model = MLP(cell_num=5, n_feature=10800, n_hidden=33, n_class=4).cuda()

if Model == "CNN":
    model = CNN(cell_num=5, n_class=4).cuda()
if Model == "LSTM":
    model = LSTM(input_size=2700, n_class=4).cuda()

model.load_state_dict(torch.load(path1))

train_dataloader,val_dataloader = train_val_dataset_sw2(SOH_path, data_path,Shuffle=False,Training_ratio=0.8,batch_size=6,row_pitch=2,sw_shape=20,prediction_num=4)


model.eval()


def data(model):
    for batch_index, batch_data in enumerate(train_dataloader):
        encoder_inputs, labels = batch_data
        prediction=model(encoder_inputs)

        prediction=prediction.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        if batch_index==0:
            predict_train=prediction[:,0]
            label_train=labels[:,0]
        else:
            predict_train=np.append(predict_train,prediction[:,0])
            label_train=np.append(label_train,labels[:,0])



    for batch_index, batch_data in enumerate(val_dataloader):
        encoder_inputs, labels = batch_data
        prediction=model(encoder_inputs)

        prediction=prediction.cpu().detach().numpy()


        labels = labels.cpu().numpy()

        if batch_index==0:
            predict_val=prediction[:,0]
            label_val=labels[:,0]
        else:
            predict_val=np.append(predict_val,prediction[:,0])
            label_val=np.append(label_val,labels[:,0])


    prediction=np.append(predict_train,predict_val)
    label=np.append(label_train,label_val)

    rmse=RMSE(label_val,predict_val)
    mape = MAPE(label_val,predict_val)

    # return predict_val, label_val, rmse
    return prediction,label,rmse, mape


predict,laebl,rmse,mape=data(model)


# for i in range(len(predict2)):
#     if predict2[i]<0.6:
#         predict2[i]=(predict2[i+1]+predict2[i-1])/2

import os
import pandas as pd
plot_data_dir = "./plot_data"
if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)

predict_df = pd.DataFrame({'Prediction': predict})
label_df = pd.DataFrame({'Label': laebl})

# Save the label DataFrame as "VIT.csv" in the "plot_data" directory
output_file1 = os.path.join(plot_data_dir, Model + ".csv")
predict_df.to_csv(output_file1, index=False, header=False)

output_file2 = os.path.join(plot_data_dir, "labels.csv")
label_df.to_csv(output_file2, index=False, header=False)

# 打印画图
# print('rmse_CNN_LSTM is %f'%rmse)
print('rmse is {}, mape is {}'.format(rmse,mape))
# print(len(predict))
# x=np.linspace(1,len(predict),num=len(predict))
# plt.plot(x,laebl,label='labels',color='r')
# plt.plot(x,predict,label=MODEL+'_'+ shuffle +'_'+str(num),color='g')
# plt.legend()
# plt.show()



