from train_SOH import train_main
import os
import torch
from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net, CNN, LSTM, MLP

# 是否打乱顺序
shuffle = [True]
# model=['VIT','LSTM','CNN_LSTM','CNN','MLP']
model = ['VIT']
train_num = [0.3, 0.5, 0.8]
cell_num = 5


"""

atten0.5 和0 ，50%训练


"""


# data_path= "/home/zhihao/MIT_params_tuner/data/20/data.csv"
# SOH_path="/home/zhihao/MIT_params_tuner/data/20/SOH.csv"

# data_path= "/home/zhihao/MIT_params_tuner/data_mit/30/data_30.csv"
# SOH_path="/home/zhihao/MIT_params_tuner/data_mit/30/SOH_30.csv"

data_path= "/home/zhihao/MIT_params_tuner/data_mit/15/data_15.csv"
SOH_path="/home/zhihao/MIT_params_tuner/data_mit/15/SOH_15.csv"


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
root = "/home/zhihao/SOH/MIT_SW/15_atten0.5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for Model in model:
    for Shuffle in shuffle:
        if Shuffle == True:
            train_num = [0.5]
        else:
            train_num = [0.8]
        for Train_num in train_num:
            model_para_filename = root + '/' + "Model" + '/' + Model + '/' + "shuffle_" + str(
                Shuffle) + '/' + str(Train_num) + '_' + "params"
            model_para_loss = root + '/' + "loss" + '/' + "shuffle_" + str(
                Shuffle) + '/' + Model + '/' + str(Train_num) + '_' + "loss"

            if not os.path.exists(model_para_filename):
                os.makedirs(model_para_filename)
            if not os.path.exists(model_para_loss):
                os.makedirs(model_para_loss)

            if isinstance(Model, str):
                if Model == "VIT":
                    model = SE_VIT_Decoder(type=1, embed_dim=500, attn_drop_ratio=0.5, prediction_num=1).cuda()

            if model == "CNN_LSTM":
                # CNN_LSTM.RNN,CGU

                model = Net(cell_num=cell_num, input_size=180, hidden_dim=25, num_layers=3).cuda()

            if model == "MLP":
                model = MLP(cell_num=cell_num, n_feature=10800, n_hidden=33, n_output=1).cuda()

            if model == "CNN":
                model = CNN(cell_num=cell_num).cuda()
            if model == "LSTM":
                model = LSTM(input_size=2700).cuda()

            print(model_para_filename)
            print(model_para_loss)
            # 8000,15000
            train_main(learning_rate=1e-4, SOH_path=SOH_path, data_path=data_path, epoch=5500,
                       Training_ratio=Train_num, device=device, loss_path=model_para_loss,
                       params_filename=model_para_filename, model=model,
                       Shuffle=Shuffle, patch_size=2, batch_size=6, sw_size=15, prediction_num=1)


"""
    30%训练的的默认参数
    "batch_size": 6,
    "decoderFc_size": 96,
    "lr": 0.001,
    momentum 0.1
    "attn_drop_ratio": 0,
    "depth": 3,
    "num_heads": 4,
    "patch_size": 2,
    "embedding": 800

    取得不错效果的参数
    {
        "batch_size": 12,
        "decoderFc_size": 500,
        "lr": 0.001,
        "momentum": 0.1,
        "attn_drop_ratio": 0,
        "depth": 2,
        "num_heads": 10,
        "patch_size": 10,
        "embedding": 800
    }
    文章两端对其
"""
