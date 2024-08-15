import torch
from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net, CNN, LSTM, MLP
from data_preprocess import train_val_dataset_sw2
import numpy as np
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Model = "MLP"
# epoch = 1950

# Model = "CNN_LSTM"
# epoch = 2850


# Model = "CNN"
# epoch = 500

Model = "LSTM"
epoch = 2250

# Model = "VIT"
# epoch = 150

train_radio = 0.8
# path=r"C:\Users\xzh\Desktop\SOH1\NASA_196_4_1\Model\CNN_LSTM\shuffle_True\0.8_params\epoch_10500_train.params"
# path1="./MIT_decoder_20_4/Model/VIT/shuffle_True/0.8_params/epoch_5350_train.params"

path1 = './MIT_decoder_20_4/Model/' + Model + '/shuffle_True/' + str(train_radio) + '_params/STANDALONE_epoch_' + str(
    epoch) + '.params'
# path1 = './MIT_decoder_20_4/Model/' + Model + '/shuffle_True/' + str(train_radio) + '_params/STANDALONE_epoch_' + str(
#     epoch) + '_train.params'

# Model = "VIT"
# epoch = 500
# #
# # path1 = './MIT_params_tuner/1/model/epoch_' + str(
# #     epoch) + '.params'
#
# path1 = "/home/lunet/ttpf/battery-package/nasa-master/nasa-master/MIT/MIT_params_tuner/model/A16Io_epoch_700.params"
# B0005
# SOH_path = r"C:\Users\xzh\Desktop\SOH1\result_rubbish\NASA_196_4_1\data\SOH.csv"
# data_path = r"C:\Users\xzh\Desktop\SOH1\result_rubbish\NASA_196_4_1\data\data.csv"

# SOH_path = "./data/SOH.csv"
# data_path = "./data/data.csv"

SOH_path = "./data/SOH.csv"
data_path = "./data/data.csv"

num = path1.split("/")
# shuffle = num[0]
# if shuffle == "shuffle_True":
#     Shuffle = True
# else:
#     Shuffle = False
Shuffle = False


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def RMSE(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, ord=2) / len(y_true) ** 0.5


def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """
    n = len(y_true)
    mse = (np.square(y_true - y_pred)) / n
    sum = mse.sum()
    return sum ** 0.5


# 加载俩模型对比

if Model == "VIT":
    model = SE_VIT_Decoder(type=1, embed_dim=500, attn_drop_ratio=0.5, prediction_num=1).cuda()

if Model == "CNN_LSTM":
    # CNN_LSTM.RNN,CGU

    model = Net(cell_num=8, input_size=180, hidden_dim=25, num_layers=3, n_class=1).cuda()

if Model == "MLP":
    model = MLP(cell_num=8, n_feature=10800, n_hidden=50, n_class=1).cuda()

if Model == "CNN":
    model = CNN(cell_num=8, n_class=1).cuda()
if Model == "LSTM":
    model = LSTM(input_size=4320, hidden_dim=25, num_layers=3, n_class=1).cuda()

model.load_state_dict(torch.load(path1))

train_dataloader, val_dataloader = train_val_dataset_sw2(SOH_path, data_path, Shuffle=False, Training_ratio=train_radio,
                                                         batch_size=6, row_pitch=2, sw_shape=20, num=180,
                                                         prediction_num=1)

model.eval()


def data(model):
    for batch_index, batch_data in enumerate(train_dataloader):
        encoder_inputs, labels = batch_data

        prediction = model(encoder_inputs)

        prediction = prediction.cpu().detach().numpy()
        labels = labels.cpu().numpy()

        if batch_index == 0:
            predict_train = prediction[:, 0]
            label_train = labels[:, 0]
        else:
            predict_train = np.append(predict_train, prediction[:, 0])
            label_train = np.append(label_train, labels[:, 0])

    for batch_index, batch_data in enumerate(val_dataloader):
        encoder_inputs, labels = batch_data
        prediction = model(encoder_inputs)

        prediction = prediction.cpu().detach().numpy()

        labels = labels.cpu().numpy()

        if batch_index == 0:
            predict_val = prediction[:, 0]
            label_val = labels[:, 0]
        else:
            predict_val = np.append(predict_val, prediction[:, 0])
            label_val = np.append(label_val, labels[:, 0])
    prediction = np.append(predict_train, predict_val)
    label = np.append(label_train, label_val)

    rmse = RMSE(label_val, predict_val)
    mape = MAPE(label_val,predict_val)

    # return predict_val, label_val, rmse
    return prediction,label,rmse, mape

def compute_val_loss(model, val_loader, criterion, device):
    model.to(device)
    criterion.to(device)
    model.eval()
    with torch.no_grad():
        val_loader_length = len(val_loader)
        tmp = []

        for batch_idnex, batch_data in enumerate(val_loader):
            encoder_input, labels = batch_data
            output = model(encoder_input)
            loss = criterion(output, labels)
            tmp.append(loss.item())



    validation_loss = sum(tmp) / len(tmp)

    return validation_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss().to(device)
val_loss = compute_val_loss(model, val_dataloader, criterion, device)

print("Val_loss:", val_loss)

predict, laebl, rmse, mape = data(model)

# for i in range(len(predict2)):
#     if predict2[i]<0.6:
#         predict2[i]=(predict2[i+1]+predict2[i-1])/2


# 打印画图
# print('rmse_CNN_LSTM is %f'%rmse)
print('rmse is {}, mape is {}'.format(rmse,mape))

x = np.linspace(1, len(predict), num=len(predict))
plt.plot(x, laebl, label='labels', color='r')
plt.plot(x, predict, label=Model, color='g')
# plt.plot(x, predict-laebl, label=Model, color='g')
plt.legend()
plt.show()
