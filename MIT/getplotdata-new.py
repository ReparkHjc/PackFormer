import torch
from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net, CNN, LSTM, MLP
from data_preprocess import train_val_dataset_sw2
import numpy as np
import csv
import numpy as np
import os
import pandas as pd



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

def data(model, train_dataloader, val_dataloader):
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

# 加载俩模型对比
def plotdata(Model = "MLP"):

    SOH_path = "./data/SOH.csv"
    data_path = "./data/data.csv"
    train_radio = 0.8

    if Model == "CNN_LSTM":
        # CNN_LSTM.RNN,CGU

        model = Net(cell_num=2, input_size=180, hidden_dim=25, num_layers=3, n_class=4).cuda()
        path1 = './MIT_decoder_20_4/Model/CNN_LSTM/shuffle_True/0.8_params/STANDALONE_epoch_150.params'
    elif Model == "MLP":
        model = MLP(cell_num=2, n_feature=10800, n_hidden=50, n_class=4).cuda()
        path1 = './MIT_decoder_20_4/Model/MLP/shuffle_True/0.8_params/STANDALONE_epoch_4450.params'
    elif Model == "CNN":
        model = CNN(cell_num=2, n_class=4).cuda()
        path1 = './MIT_decoder_20_4/Model/CNN/shuffle_True/0.8_params/STANDALONE_epoch_950.params'

    elif Model == "LSTM":
        model = LSTM(input_size=1080, hidden_dim=25, num_layers=3, n_class=4).cuda()
        path1 = './MIT_decoder_20_4/Model/LSTM/shuffle_True/0.8_params/STANDALONE_epoch_2300.params'
    elif Model == "VIT":
        model = SE_VIT_Decoder(type=1, embed_dim=500, attn_drop_ratio=0.4, prediction_num=4).cuda()
        path1 = './MIT_decoder_20_4/Model/VIT/shuffle_True/0.8_params/STANDALONE_epoch_2850.params'

    model.load_state_dict(torch.load(path1))

    train_dataloader, val_dataloader = train_val_dataset_sw2(SOH_path, data_path, Shuffle=True, Training_ratio=train_radio,
                                                             batch_size=6, row_pitch=10, sw_shape=20, num=180,
                                                             prediction_num=4)

    model.eval()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss().to(device)
    val_loss = compute_val_loss(model, val_dataloader, criterion, device)

    print("Val_loss:", val_loss)

    predict, laebl, rmse, mape = data(model, train_dataloader, val_dataloader)



    # Create the "plot_data" directory if it doesn't exist
    plot_data_dir = "./plot_data"
    if not os.path.exists(plot_data_dir):
        os.makedirs(plot_data_dir)

    predict_df = pd.DataFrame({'Prediction': predict})

    # Save the label DataFrame as "VIT.csv" in the "plot_data" directory
    output_file1 = os.path.join(plot_data_dir, Model+".csv")
    predict_df.to_csv(output_file1, index=False, header=False)


if __name__ == "__main__":
    model = ['CNN_LSTM', 'CNN', 'MLP', 'LSTM','VIT']
    for Model in model:
        plotdata(Model)