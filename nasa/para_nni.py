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


logger = logging.getLogger("VIT")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型不调节参数
parser = argparse.ArgumentParser(description='default para')
parser.add_argument('--Training_ratio', type=float, default=0.8)
parser.add_argument('--epoch', type=int, default=1000)

# 数据路径
parser.add_argument('--data_dir', type=str, default="./data/data.csv")
parser.add_argument('--SOH_dir', type=str, default="./data/SOH.csv")

parser.add_argument('--model_shuffle_filename', type=str, default="./params_tuner/model/")
parser.add_argument('--loss_filename', type=str, default="./params_tuner/loss/")

parser.add_argument('--sw_size', type=int, default=20)
parser.add_argument('--prediction_num', type=int, default=1)
para, _ = parser.parse_known_args()
para = vars(para)


# 模型调节默认参数
def get_params():

    # 模型调节参数
    parser = argparse.ArgumentParser(description='VIT default param')
    parser.add_argument("--batch_size", type=int,
                        default=6, help="data directory")
    parser.add_argument('--decoderFc_size', type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--attn_drop_ratio", type=float, default=0.0)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--embedding', type=int, default=500)
    parser.add_argument('--momentum', type=float, default=0.1)

    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameters()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(type(params["lr"]))

        model = SE_VIT_Decoder(patch_size=params["patch_size"], embed_dim=params["embedding"], depth=params["depth"],
                               num_heads=params["num_heads"], attn_drop_ratio=params["attn_drop_ratio"],
                               momentum=params["momentum"],
                               decoderFc_size=params["decoderFc_size"], prediction_num=para["prediction_num"])

        train_main(learning_rate=params["lr"], SOH_path=para["SOH_dir"], data_path=para["data_dir"],
                   Training_ratio=para["Training_ratio"], epoch=para["epoch"],
                   params_filename=para["model_shuffle_filename"],
                   loss_path=para['loss_filename'], device=device, patch_size=params["patch_size"],
                   batch_size=params["batch_size"], model=model,prediction_num=para["prediction_num"],sw_size=para["sw_size"])

    except Exception as exception:
        logger.exception(exception)
        raise
