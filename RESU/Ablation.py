import os
import argparse
import logging
import nni
from VITNNi import SE_VIT_Decoder
from train_SOH import train_main
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms

"""
    用于消融实验程序入口
    20->4 or 1等
    nni调参之后试验
"""

logger = logging.getLogger("VIT")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型不调节参数
parser = argparse.ArgumentParser(description='default para')
parser.add_argument('--Training_ratio', type=float, default=0.8)
parser.add_argument('--epoch', type=int, default=5500)

# 20->4 or 1等
parser.add_argument('--sw_size', type=int, default=20)
parser.add_argument('--prediction_num', type=int, default=1)

# 数据路径
parser.add_argument('--data_dir', type=str, default="./data/data.csv")
parser.add_argument('--SOH_dir', type=str, default="./data/SOH.csv")

parser.add_argument('--model_shuffle_filename', type=str, default="./ablation/model/")
parser.add_argument('--loss_filename', type=str, default="./ablation/loss/")

# 模型调节默认参数
parser.add_argument("--batch_size", type=int,
                    default=6, help="data directory")
parser.add_argument('--decoderFc_size', type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--attn_drop_ratio", type=float, default=0.5)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--embedding', type=int, default=1024)
parser.add_argument('--momentum', type=float, default=0.1)

# 转为字典
params, _ = parser.parse_known_args()
params = vars(params)

if __name__ == "__main__":

    model = SE_VIT_Decoder(type=4,patch_size=params["patch_size"], embed_dim=params["embedding"], depth=params["depth"],
                           num_heads=params["num_heads"], attn_drop_ratio=params["attn_drop_ratio"],
                           momentum=params["momentum"],
                           decoderFc_size=params["decoderFc_size"], prediction_num=params["prediction_num"])

    train_main(learning_rate=params["lr"], SOH_path=params["SOH_dir"], data_path=params["data_dir"],
               Training_ratio=params["Training_ratio"], epoch=params["epoch"],
               params_filename=params["model_shuffle_filename"],
               loss_path=params['loss_filename'], device=device, patch_size=params["patch_size"],
               batch_size=params["batch_size"],
               sw_size=params["sw_size"], prediction_num=params["prediction_num"], model=model)
