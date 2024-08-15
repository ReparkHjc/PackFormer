import numpy as np
import numpy as py
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data
import random
import os
import math


def date_preprocess(data, data_num, pitch_row_num, row_num):
    """
        Args:
                data:电池全部数据
                data_num:从电池数据中随机抽取的数目
                pitch_row_num:每个小batch的数目，2*2，或者3*3或其他
                row_num:分成多少个batch
    """

    data_Matrix = py.zeros((row_num * pitch_row_num, row_num * pitch_row_num), dtype=float)

    if data_num % (pitch_row_num * pitch_row_num) != 0 or data_num % row_num != 0:
        raise ValueError("数据维度不对")

    # for i in data:
    # 行
    for j in range(row_num):
        # 列
        for k in range(row_num):
            # 行
            for j1 in range(pitch_row_num):
                # 列
                for k1 in range(pitch_row_num):
                    first_num = j * pitch_row_num * pitch_row_num * row_num + k * pitch_row_num * pitch_row_num + 1
                    first_num = first_num + j1 * pitch_row_num + k1
                    data_Matrix[pitch_row_num * j + j1, pitch_row_num * k + k1] = data[first_num - 1]

    return data_Matrix


def con(u, i, t) -> torch.Tensor:
    """
        Args:
                u:电池的电压
                i:电流
                t:温度
    """
    list_U = u
    list_I = i
    list_T = t
    list = [list_U, list_I, list_T]
    data_tensor = torch.Tensor(list)
    return data_tensor.cuda()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_val_dataset_sw2(SOH_path, data_path, Shuffle, Training_ratio, prediction_num, batch_size, num=180,
                          sw_shape=20, row_pitch=2):
    pitch_num = int(math.sqrt(num * sw_shape) / row_pitch)

    data = pd.read_csv(data_path, header=None)

    data.columns = ["U1", "I1", "T1", "t1", "U2", "I2", "T2", "t2"]
    SOH = pd.read_csv(SOH_path, header=None)


    # Time,V,i,t
    # T是温度，t时间
    Train_num = int(Training_ratio * (SOH.shape[1] - 55))

    SOH_list = SOH.values.tolist()
    SOH_list = torch.Tensor(SOH_list).reshape(-1, 1).cuda()

    U1 = py.array(data["U1"]).reshape(1, -1)

    I1 = py.array(data["I1"]).reshape(1, -1)
    T1 = py.array(data["T1"]).reshape(1, -1)
    t1 = py.array(data["t1"]).reshape(1, -1)

    U2 = py.array(data["U2"]).reshape(1, -1)

    I2 = py.array(data["I2"]).reshape(1, -1)
    T2 = py.array(data["T2"]).reshape(1, -1)
    t2 = py.array(data["t2"]).reshape(1, -1)

    random.seed(1)

    idx_v = []
    idx_t = random.sample(range(0, 800), Train_num)
    idx_t = py.sort(idx_t)
    all_idx = py.arange(0, 800, 1)
    for i in range(len(all_idx)):
        for j in range(len(idx_t)):
            if all_idx[i] == idx_t[j]:
                idx_v.append(i)
    idx_v = py.delete(all_idx, idx_v)
    idx_v = py.array(idx_v)

    is_first_run = 0

    if Shuffle == True:
        for train_num in range(0, Train_num):
            train_num = idx_t[train_num]
            V_1 = date_preprocess(U1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)])

        # SOH的维度没对其
        train_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        is_first_run = 0
        for train_num in range(0, 800 - Train_num):
            train_num = idx_v[train_num]
            V_1 = date_preprocess(U1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)])

        # SOH的维度没对其
        val_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader
    else:
        for train_num in range(0, Train_num):
            V_1 = date_preprocess(U1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)])

        SOH = SOH.unsqueeze(dim=1)
        # SOH的维度没对其
        train_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        is_first_run = 0
        for train_num in range(Train_num, 800):
            V_1 = date_preprocess(U1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num:(sw_shape + train_num) * num], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num + sw_shape:train_num + sw_shape + prediction_num].reshape(1, -1)])

        # SOH的维度没对其
        val_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader

if __name__ == "__main__":
    train_dataloader, val_dataloader = train_val_dataset_sw2("./data/SOH.csv", "./data/data.csv", False, 0.7, 1, 2, num=180,
                          sw_shape=20, row_pitch=2)
    print(train_dataloader)