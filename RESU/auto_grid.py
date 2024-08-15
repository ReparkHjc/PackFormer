from VITNNi import SE_VIT_Decoder
from CNN_LSTM import Net, CNN, LSTM, MLP

import torch.nn as nn

import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from data_preprocess import train_val_dataset_sw2
import numpy as np
import os
import nni


def train_main(learning_rate, SOH_path, data_path, Training_ratio, epoch, loss_path, patch_size,device, batch_size,params_filename,
               sw_size,prediction_num,model, Shuffle=True):


    # 加载模型
    # model.load_state_dict(torch.load(path))

    sw = SummaryWriter(logdir=loss_path, flush_secs=5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 更新学习率
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)
    criterion = nn.MSELoss().to(device)

    global_step = 0
    num=int(3600/sw_size)
    train_dataloader, val_dataloader = train_val_dataset_sw2(SOH_path=SOH_path, data_path=data_path, Shuffle=Shuffle,
                                                             Training_ratio=Training_ratio, batch_size=batch_size,
                                                             row_pitch=patch_size,sw_shape=sw_size,prediction_num=prediction_num,num=num)

    best_val_loss = np.inf
    best_train_loss = np.inf

    # 创建文件夹
    if not os.path.exists(params_filename):
        os.mkdir(params_filename)
    if not os.path.exists(loss_path):
        os.mkdir(loss_path)
    val=[]
    for epoch in range(epoch):
        params_filename1 = os.path.join(params_filename, 'epoch_%s.params' % epoch)
        params_filename2 = os.path.join(params_filename, 'epoch_%s_train.params' % epoch)
        sum_loss_train = []

        if epoch % 50 == 0:
            val_loss = compute_val_loss(model, val_dataloader, criterion, sw, epoch, device)
            # 中间结果
            nni.report_intermediate_result(val_loss)
            # print('val_loss is = %10.9f' % (val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), params_filename1)
            val.append(val_loss)

        model.train()
        for batch_index, batch_data in enumerate(train_dataloader):
            encoder_inputs, labels = batch_data
            optimizer.zero_grad()
            outputs = model(encoder_inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # print(epoch, optimizer.state_dict()['param_groups'][0]['lr'])

            optimizer.step()
            training_loss = loss.item()
            sum_loss_train.append(training_loss)

            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)

            # if global_step % 10 == 0:
            #  print('global step: %s| training loss: %8.7f| time: %.2fs' % (
            #     global_step, training_loss, time() - start_time))
        scheduler.step()
        sw.add_scalar('training_loss1', (sum(sum_loss_train) / len(sum_loss_train)), epoch)
        print(epoch, optimizer.state_dict()['param_groups'][0]['lr'], (sum(sum_loss_train) / len(sum_loss_train)),
              training_loss)
        if sum(sum_loss_train) < best_train_loss and epoch % 50 == 0:
            best_train_loss = sum(sum_loss_train)
            # print("sum_loss_train is %4.3f,best_train_loss is %15.14f" % (sum(sum_loss_train), best_train_loss))
            torch.save(model.state_dict(), params_filename2)
    # 最终结果
    nni.report_final_result(min(val))

def compute_val_loss(model, val_loader, criterion, sw, epoch,device):
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
    sw.add_scalar("validation_loss", validation_loss, epoch)


    return validation_loss


if __name__ == "__main__":
    # 是否打乱顺序
    print("1")
    shuffle = [True]
    # model=['CNN_LSTM','CNN','MLP','LSTM','VIT']
    model = ['VIT']
    # model=['LSTM']

    train_num = [0.8]
    cell_num = 2

    data_path = "./data_mit/20/data.csv"
    SOH_path = "./data_mit/20/SOH.csv"

    params = {
        'batch_size': 6,
        'decoderFc_size': 200,
        'depth': 2,
        'num_heads': 5,
        'embedding': 400
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)



    """
    # 20->4效果  input output
    atten0 效果不错，不改了
    
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    root = "./MIT_grid_20_4/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for Model in model:
        for Shuffle in shuffle:
            for Train_num in train_num:

                model_para_filename = root + '/' + "Model" + '/' + Model + '/' + "batch_size_" + str(
                    params['batch_size']) + '/decoderFc_size_' + str(params['decoderFc_size']) + '/depth_' + str(params['depth']) + '/num_heads_' + str(
                    params['num_heads']) + '/embedding_' + str(params['embedding'])

                model_para_loss = root + '/' + "loss" + '/' + "shuffle_" + str(
                    Shuffle) + '/' + Model + '/' + "batch_size_" + str(
                    params['batch_size']) + '/decoderFc_size_' + str(params['decoderFc_size']) + '/depth_' + str(params['depth']) + '/num_heads_' + str(
                    params['num_heads']) + '/embedding_' + str(params['embedding'])

                if not os.path.exists(model_para_filename):
                    os.makedirs(model_para_filename)
                if not os.path.exists(model_para_loss):
                    os.makedirs(model_para_loss)

                if isinstance(Model, str):
                    if Model == "VIT":
                        model = SE_VIT_Decoder(type=1, embed_dim=500, attn_drop_ratio=0.5, prediction_num=4).cuda()

                if Model == "CNN_LSTM":
                    # CNN_LSTM.RNN,CGU

                    model = Net(cell_num=cell_num, input_size=180, hidden_dim=25, num_layers=3, n_class=4).cuda()

                if Model == "MLP":
                    model = MLP(cell_num=cell_num, n_feature=10800, n_hidden=50, n_class=4).cuda()

                if Model == "CNN":
                    model = CNN(cell_num=cell_num, n_class=4).cuda()
                if Model == "LSTM":
                    model = LSTM(input_size=1080, hidden_dim=25, num_layers=3, n_class=4).cuda()

                print(model_para_filename)
                print(model_para_loss)
                # 8000,15000
                train_main(learning_rate=1e-4, SOH_path=SOH_path, data_path=data_path, epoch=5500,
                           Training_ratio=Train_num, device=device, loss_path=model_para_loss,
                           params_filename=model_para_filename, model=model,
                           Shuffle=Shuffle, patch_size=2, batch_size=params['batch_size'], sw_size=20, prediction_num=4)


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
