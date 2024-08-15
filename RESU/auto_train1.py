from train_SOH import train_main
import os
import torch
from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net, CNN, LSTM, MLP


# shuffle = [False]
shuffle = [True]

# model=['VIT','CNN_LSTM','CNN','MLP','LSTM']
# model=['CNN_LSTM','CNN','MLP','LSTM']
# model=['LSTM']

model = ['VIT']
# model=['CNN_LSTM']
# model = ['ablation']

# train_num = [0.3, 0.5, 0.8]
train_num = [0.8]

cell_num = 8

data_path="./data/data.csv"
SOH_path="./data/SOH.csv"



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
root = "./RESU_decoder_20_4/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for Model in model:
    for Shuffle in shuffle:
        if Shuffle == True:
            train_num = [0.8]
        else:
            train_num = [0.3, 0.5, 0.8]
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
                    model = SE_VIT_Decoder(type=1, embed_dim=1024, attn_drop_ratio=0.5, prediction_num=1).cuda()

            if Model == "CNN_LSTM":
                # CNN_LSTM.RNN,CGU
                model = Net(cell_num=cell_num, input_size=180, hidden_dim=25, num_layers=3,n_class=1).cuda()


            if Model == "MLP":
                model = MLP(cell_num=cell_num, n_feature=10800, n_hidden=50,n_class=1).cuda()

            if Model == "CNN":
                model = CNN(cell_num=cell_num , n_class=1).cuda()
            if Model == "LSTM":
                model = LSTM(input_size=4320,hidden_dim=25,num_layers=3,n_class=1).cuda()


            print(model_para_filename)
            print(model_para_loss)
            # 8000,15000
            train_main(learning_rate=1e-4, SOH_path=SOH_path, data_path=data_path, epoch=10000,
                       Training_ratio=Train_num, device=device, loss_path=model_para_loss,
                       params_filename=model_para_filename, model=model,
                       Shuffle=Shuffle, patch_size=2, batch_size=6, sw_size=20, prediction_num=1)




