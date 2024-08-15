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
                    # print(data.shape)
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

    data.columns = ["U1", "I1", "T1", "U2", "I2", "T2", "U3", "I3", "T3", "U4", "I4", "T4", "U5", "I5", "T5", "U6", "I6", "T6", "U7", "I7", "T7", "U8", "I8", "T8"]

    SOH = pd.read_csv(SOH_path, header=None)

    # Time,V,i,t
    # T是温度，t时间
    Train_num = int(Training_ratio * SOH.shape[1])

    SOH_list = SOH.values.tolist()

    SOH_list = torch.Tensor(SOH_list).reshape(-1, 1).cuda()

    U1 = py.array(data["U1"]).reshape(1, -1)
    I1 = py.array(data["I1"]).reshape(1, -1)
    T1 = py.array(data["T1"]).reshape(1, -1)

    U2 = py.array(data["U2"]).reshape(1, -1)
    I2 = py.array(data["I2"]).reshape(1, -1)
    T2 = py.array(data["T2"]).reshape(1, -1)

    U3 = py.array(data["U3"]).reshape(1, -1)
    I3 = py.array(data["I3"]).reshape(1, -1)
    T3 = py.array(data["T3"]).reshape(1, -1)

    U4 = py.array(data["U4"]).reshape(1, -1)
    I4 = py.array(data["I4"]).reshape(1, -1)
    T4 = py.array(data["T4"]).reshape(1, -1)

    U5 = py.array(data["U5"]).reshape(1, -1)
    I5 = py.array(data["I5"]).reshape(1, -1)
    T5 = py.array(data["T5"]).reshape(1, -1)

    U6 = py.array(data["U6"]).reshape(1, -1)
    I6 = py.array(data["I6"]).reshape(1, -1)
    T6 = py.array(data["T6"]).reshape(1, -1)

    U7 = py.array(data["U7"]).reshape(1, -1)
    I7 = py.array(data["I7"]).reshape(1, -1)
    T7 = py.array(data["T7"]).reshape(1, -1)

    U8 = py.array(data["U8"]).reshape(1, -1)
    I8 = py.array(data["I8"]).reshape(1, -1)
    T8 = py.array(data["T8"]).reshape(1, -1)

    random.seed(1)

    idx_v = []
    idx_t = random.sample(range(0, 19), Train_num)
    idx_t = py.sort(idx_t)
    all_idx = py.arange(0, 19, 1)
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
            V_1 = date_preprocess(U1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            V_3 = date_preprocess(U3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_3 = date_preprocess(I3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_3 = date_preprocess(T3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor3 = con(V_3, I_3, T_3)
            data_tensor3 = data_tensor3.unsqueeze(dim=0)

            V_4 = date_preprocess(U4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_4 = date_preprocess(I4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_4 = date_preprocess(T4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor4 = con(V_4, I_4, T_4)
            data_tensor4 = data_tensor4.unsqueeze(dim=0)

            V_5 = date_preprocess(U5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_5 = date_preprocess(I5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_5 = date_preprocess(T5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor5 = con(V_5, I_5, T_5)
            data_tensor5 = data_tensor5.unsqueeze(dim=0)

            V_6 = date_preprocess(U6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_6 = date_preprocess(I6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_6 = date_preprocess(T6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor6 = con(V_6, I_6, T_6)
            data_tensor6 = data_tensor6.unsqueeze(dim=0)

            V_7 = date_preprocess(U7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_7 = date_preprocess(I7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_7 = date_preprocess(T7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor7 = con(V_7, I_7, T_7)
            data_tensor7 = data_tensor7.unsqueeze(dim=0)

            V_8 = date_preprocess(U8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_8 = date_preprocess(I8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_8 = date_preprocess(T8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor8 = con(V_8, I_8, T_8)
            data_tensor8 = data_tensor8.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2, data_tensor3, data_tensor4, data_tensor5, data_tensor6, data_tensor7, data_tensor8], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num:train_num + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num:train_num + prediction_num].reshape(1, -1)])

        # SOH的维度没对其
        train_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        is_first_run = 0
        for train_num in range(0, 19-Train_num):
            train_num = idx_v[train_num]

            V_1 = date_preprocess(U1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            V_3 = date_preprocess(U3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_3 = date_preprocess(I3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_3 = date_preprocess(T3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor3 = con(V_3, I_3, T_3)
            data_tensor3 = data_tensor3.unsqueeze(dim=0)

            V_4 = date_preprocess(U4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_4 = date_preprocess(I4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_4 = date_preprocess(T4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor4 = con(V_4, I_4, T_4)
            data_tensor4 = data_tensor4.unsqueeze(dim=0)

            V_5 = date_preprocess(U5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_5 = date_preprocess(I5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_5 = date_preprocess(T5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor5 = con(V_5, I_5, T_5)
            data_tensor5 = data_tensor5.unsqueeze(dim=0)

            V_6 = date_preprocess(U6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_6 = date_preprocess(I6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_6 = date_preprocess(T6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor6 = con(V_6, I_6, T_6)
            data_tensor6 = data_tensor6.unsqueeze(dim=0)

            V_7 = date_preprocess(U7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_7 = date_preprocess(I7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_7 = date_preprocess(T7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor7 = con(V_7, I_7, T_7)
            data_tensor7 = data_tensor7.unsqueeze(dim=0)

            V_8 = date_preprocess(U8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_8 = date_preprocess(I8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_8 = date_preprocess(T8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor8 = con(V_8, I_8, T_8)
            data_tensor8 = data_tensor8.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2, data_tensor3, data_tensor4, data_tensor5, data_tensor6, data_tensor7, data_tensor8], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num:train_num + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num:train_num + prediction_num].reshape(1, -1)])

        # SOH的维度没对其
        val_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader
    else:
        for train_num in range(0, Train_num):
            V_1 = date_preprocess(U1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            V_3 = date_preprocess(U3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_3 = date_preprocess(I3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_3 = date_preprocess(T3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor3 = con(V_3, I_3, T_3)
            data_tensor3 = data_tensor3.unsqueeze(dim=0)

            V_4 = date_preprocess(U4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_4 = date_preprocess(I4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_4 = date_preprocess(T4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor4 = con(V_4, I_4, T_4)
            data_tensor4 = data_tensor4.unsqueeze(dim=0)

            V_5 = date_preprocess(U5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_5 = date_preprocess(I5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_5 = date_preprocess(T5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor5 = con(V_5, I_5, T_5)
            data_tensor5 = data_tensor5.unsqueeze(dim=0)

            V_6 = date_preprocess(U6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_6 = date_preprocess(I6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_6 = date_preprocess(T6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor6 = con(V_6, I_6, T_6)
            data_tensor6 = data_tensor6.unsqueeze(dim=0)

            V_7 = date_preprocess(U7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_7 = date_preprocess(I7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_7 = date_preprocess(T7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor7 = con(V_7, I_7, T_7)
            data_tensor7 = data_tensor7.unsqueeze(dim=0)

            V_8 = date_preprocess(U8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_8 = date_preprocess(I8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_8 = date_preprocess(T8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor8 = con(V_8, I_8, T_8)
            data_tensor8 = data_tensor8.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2, data_tensor3, data_tensor4, data_tensor5, data_tensor6, data_tensor7, data_tensor8], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num:train_num + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num:train_num + prediction_num].reshape(1, -1)])

        SOH = SOH.unsqueeze(dim=1)
        # SOH的维度没对其
        train_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=Shuffle)

        is_first_run = 0
        for train_num in range(Train_num, 19):

            V_1 = date_preprocess(U1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_1 = date_preprocess(I1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_1 = date_preprocess(T1[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)

            data_tensor1 = con(V_1, I_1, T_1)
            data_tensor1 = data_tensor1.unsqueeze(dim=0)

            V_2 = date_preprocess(U2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_2 = date_preprocess(I2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_2 = date_preprocess(T2[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor2 = con(V_2, I_2, T_2)
            data_tensor2 = data_tensor2.unsqueeze(dim=0)

            V_3 = date_preprocess(U3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_3 = date_preprocess(I3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_3 = date_preprocess(T3[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor3 = con(V_3, I_3, T_3)
            data_tensor3 = data_tensor3.unsqueeze(dim=0)

            V_4 = date_preprocess(U4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_4 = date_preprocess(I4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_4 = date_preprocess(T4[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor4 = con(V_4, I_4, T_4)
            data_tensor4 = data_tensor4.unsqueeze(dim=0)

            V_5 = date_preprocess(U5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_5 = date_preprocess(I5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_5 = date_preprocess(T5[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor5 = con(V_5, I_5, T_5)
            data_tensor5 = data_tensor5.unsqueeze(dim=0)

            V_6 = date_preprocess(U6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_6 = date_preprocess(I6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_6 = date_preprocess(T6[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor6 = con(V_6, I_6, T_6)
            data_tensor6 = data_tensor6.unsqueeze(dim=0)

            V_7 = date_preprocess(U7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_7 = date_preprocess(I7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_7 = date_preprocess(T7[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor7 = con(V_7, I_7, T_7)
            data_tensor7 = data_tensor7.unsqueeze(dim=0)

            V_8 = date_preprocess(U8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            I_8 = date_preprocess(I8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            T_8 = date_preprocess(T8[0][train_num * num*20:(1 + train_num) * num*20], num * sw_shape, row_pitch,
                                  pitch_num)
            data_tensor8 = con(V_8, I_8, T_8)
            data_tensor8 = data_tensor8.unsqueeze(dim=0)

            data_tensor_all = torch.cat([data_tensor1, data_tensor2, data_tensor3, data_tensor4, data_tensor5, data_tensor6, data_tensor7, data_tensor8], dim=0).unsqueeze(dim=0)

            if is_first_run == 0:
                is_first_run += 1
                data_tensor = data_tensor_all
                SOH = SOH_list[train_num:train_num + prediction_num].reshape(1, -1)
            else:
                data_tensor = torch.cat([data_tensor, data_tensor_all], dim=0)
                SOH = torch.cat([SOH, SOH_list[train_num:train_num + prediction_num].reshape(1, -1)])

        # SOH的维度没对其
        val_dataset = torch.utils.data.TensorDataset(data_tensor, SOH)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=Shuffle)

        return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader = train_val_dataset_sw2("./data/SOH.csv", "./data/data.csv", False, 0.7, 1, 2, num=180,
                          sw_shape=20, row_pitch=2)
    print(train_dataloader)