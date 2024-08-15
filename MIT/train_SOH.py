import torch
import torch.nn as nn
from time import time
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from data_preprocess import train_val_dataset_sw2
import numpy as np
import os
import nni

from torch.nn.utils import clip_grad_norm


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
        trialName = nni.get_trial_id()
        # params_filename1 = os.path.join(params_filename, f'{trialName}_epoch_{epoch}.params')
        params_filename2 = os.path.join(params_filename, f'{trialName}_epoch_{epoch}_train.params')
        sum_loss_train = []
        if epoch % 50 == 0:
            val_loss = compute_val_loss(model, val_dataloader, criterion, sw, epoch, device)
            # 中间结果
            nni.report_intermediate_result(val_loss)
            # print('val_loss is = %10.9f' % (val_loss))
            params_filename1 = os.path.join(params_filename, f'{trialName}_epoch_{epoch}.params')
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

def compute_val_loss(model, val_loader, criterion, sw, epoch, device):
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
