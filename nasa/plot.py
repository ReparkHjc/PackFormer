import torch
from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net, CNN, LSTM, MLP
from data_preprocess import train_val_dataset_sw2

import csv
import matplotlib.pyplot as plt
import numpy as np

# Model = "MLP"
# epoch = 150

# Model = "CNN_LSTM"
# epoch = 4650

# Model = "CNN"
# epoch = 150

Model = "LSTM"
epoch = 850

train_radio = 0.8

# path=r"C:\Users\xzh\Desktop\SOH1\NASA_196_4_1\Model\CNN_LSTM\shuffle_True\0.8_params\epoch_10500_train.params"
# path1="./NASA_decoder3_attn0.5/Model/VIT/shuffle_True/<class 'type'>/0.3_params/epoch_4800_train.params"
# path1="./NASA_decoder3_attn0.5/Model/VIT/shuffle_True/<class 'type'>/0.5_params/epoch_4700_train.params"

path1 = './NASA_decoder3_attn0.5/Model/' + Model + "/shuffle_True/" + str(
    train_radio) + '_params/epoch_' + str(
    epoch) + '.params'

# B0005
# SOH_path = r"C:\Users\xzh\Desktop\SOH1\result_rubbish\NASA_196_4_1\data\SOH.csv"
# data_path = r"C:\Users\xzh\Desktop\SOH1\result_rubbish\NASA_196_4_1\data\data.csv"
SOH_path = "./data/SOH.csv"
data_path = "./data/data.csv"

num = path1.split("/")
shuffle = num[0]
if shuffle == "shuffle_True":
    Shuffle = True
else:
    Shuffle = False
MODEL = path1.split("/")[0]


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


if Model == "VIT":
    model = SE_VIT_Decoder(type=1, embed_dim=500, attn_drop_ratio=0.5, depth=3, num_heads=4,
                           decoderFc_size=96, momentum=0.1).cuda()
elif Model == "CNN_LSTM":
    # CNN_LSTM.RNN,CGU
    model = Net(input_size=180, hidden_dim=25, num_layers=3).cuda()
elif Model == "MLP":
    model = MLP().cuda()
elif Model == "CNN":
    model = CNN().cuda()
elif Model == "LSTM":
    model = LSTM().cuda()

model.load_state_dict(torch.load(path1))

train_dataloader, val_dataloader = train_val_dataset_sw2(SOH_path, data_path, Shuffle=True, Training_ratio=train_radio,
                                                         batch_size=6, row_pitch=60, sw_shape=20, prediction_num=1)

model.eval()


def data(model):
    for batch_index, batch_data in enumerate(train_dataloader):
        encoder_inputs, labels = batch_data
        prediction = model(encoder_inputs)

        prediction = prediction.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        if batch_index == 0:
            predict_train = prediction
            label_train = labels
        else:
            predict_train = np.append(predict_train, prediction)
            label_train = np.append(label_train, labels)

    for batch_index, batch_data in enumerate(val_dataloader):
        encoder_inputs, labels = batch_data
        prediction = model(encoder_inputs)

        prediction = prediction.cpu().detach().numpy()

        labels = labels.cpu().numpy()

        if batch_index == 0:
            predict_val = prediction
            label_val = labels
        else:
            predict_val = np.append(predict_val, prediction)
            label_val = np.append(label_val, labels)

        prediction = np.append(predict_train, predict_val)
        label = np.append(label_train, label_val)

    rmse = RMSE(label_val, predict_val)
    mape = MAPE(label_val,predict_val)

    # return predict_val, label_val, rmse
    return prediction,label,rmse, mape


predict, laebl, rmse,mape = data(model)

# for i in range(len(predict2)):
#     if predict2[i]<0.6:
#         predict2[i]=(predict2[i+1]+predict2[i-1])/2


# 打印画图
# print('rmse_CNN_LSTM is %f'%rmse)
print('rmse is {}, mape is {}'.format(rmse,mape))

x = np.linspace(1, len(predict), num=len(predict))
plt.plot(x, laebl, label='labels', color='r')
plt.plot(x, predict, label=Model, color='g')
plt.legend()
plt.show()
