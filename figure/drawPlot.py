import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

def get_data(Dataset):
    # 检查Dataset是否合法
    if Dataset not in ['MIT', 'nasa', 'Oxford','RESU']:
        raise ValueError("Dataset必须是'MIT', 'nasa', 'RESU'或'Oxford'中的一个")

    # 所有可能的Method
    methods = ['VIT', 'LSTM', 'CNN_LSTM', 'MLP', 'CNN']

    # 存储所有Method的数据的字典
    all_data = {}

    # 存储标签数据
    label_data = None

    # 循环处理不同Method
    for Method in methods:

        # 构建文件路径
        data_folder = os.path.join('..', Dataset, 'plot_data')
        file_path = os.path.join(data_folder, Method + '.csv')

        # 检查文件是否存在
        if os.path.exists(file_path):

            # 读取.csv文件并将内容转换为数组
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 假设.csv文件包含逗号分隔的数值，将其转换为数组
            data_array = np.array([list(map(float, line.strip().split(','))) for line in lines])

            # 将数据存储在字典中
            all_data[Method] = data_array

    # 处理标签数据
    data_folder = os.path.join('..', Dataset, 'plot_data')
    file_path = os.path.join(data_folder, 'labels.csv')


    # 检查文件是否存在
    if os.path.exists(file_path):

        # 读取 label.csv 文件并将内容转换为数组
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 假设 label.csv 文件包含逗号分隔的数值，将其转换为数组
        label_data = np.array([list(map(float, line.strip().split(','))) for line in lines])

    return all_data, label_data

def get_ablation_data(Dataset):
    # 检查Dataset是否合法
    if Dataset not in ['MIT', 'nasa', 'Oxford','RESU']:
        raise ValueError("Dataset必须是'MIT', 'nasa','RESU'或'Oxford'中的一个")

    # 所有可能的Method
    methods = ['VIT', 'VIT_ablation_new']

    # 存储所有Method的数据的字典
    all_data = {}

    # 存储标签数据
    label_data = None

    # 循环处理不同Method
    for Method in methods:

        # 构建文件路径
        data_folder = os.path.join('..', Dataset, 'plot_data')
        file_path = os.path.join(data_folder, Method + '.csv')

        # 检查文件是否存在
        if os.path.exists(file_path):

            # 读取.csv文件并将内容转换为数组
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 假设.csv文件包含逗号分隔的数值，将其转换为数组
            data_array = np.array([list(map(float, line.strip().split(','))) for line in lines])

            # 将数据存储在字典中
            all_data[Method] = data_array

    # 处理标签数据
    data_folder = os.path.join('..', Dataset, 'plot_data')
    file_path = os.path.join(data_folder, 'labels.csv')


    # 检查文件是否存在
    if os.path.exists(file_path):

        # 读取 label.csv 文件并将内容转换为数组
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 假设 label.csv 文件包含逗号分隔的数值，将其转换为数组
        label_data = np.array([list(map(float, line.strip().split(','))) for line in lines])

    return all_data, label_data


def calculate_error_data(Dataset):
    # 检查Dataset是否合法
    if Dataset not in ['MIT', 'nasa', 'Oxford','RESU']:
        raise ValueError("Dataset必须是'MIT', 'nasa','RESU'或'Oxford'中的一个")

    # 获取绘图数据和标签数据
    all_data, label_data = get_data(Dataset)
    # 存储误差数据
    error_data = {}

    # 计算误差数据
    if label_data is not None:
        for Method, data_array in all_data.items():

            error_data[Method] = abs(data_array - label_data)

    return error_data

def calculate_ablation_error_data(Dataset):
    # 检查Dataset是否合法
    if Dataset not in ['MIT', 'nasa', 'Oxford','RESU']:
        raise ValueError("Dataset必须是'MIT', 'nasa','RESU'或'Oxford'中的一个")

    # 获取绘图数据和标签数据
    all_data, label_data = get_ablation_data(Dataset)
    # 存储误差数据
    error_data = {}

    # 计算误差数据
    if label_data is not None:
        for Method, data_array in all_data.items():

            error_data[Method] = abs(data_array - label_data)

    return error_data

def plot_normal_distribution(Dataset):
    # 检查Dataset是否合法
    if Dataset not in ['MIT', 'nasa', 'Oxford','RESU']:
        raise ValueError("Dataset必须是'MIT', 'nasa','RESU'或'Oxford'中的一个")


    # 获取绘图数据和标签数据
    all_data, label_data = get_ablation_data(Dataset)
    # 存储误差数据
    error_data = {}

    # 计算误差数据
    if label_data is not None:
        for Method, data_array in all_data.items():
            error_data[Method] = data_array - label_data


    plt.figure(figsize=(10, 6))

    for method, errors in error_data.items():
        # 计算均值和标准差
        mean_error = np.mean(errors)
        std_deviation = np.std(errors)

        # 生成正态分布数据
        normal_distribution = np.random.normal(mean_error, std_deviation, 1000)

        # 计算数据的最小值和最大值
        min_value = min(errors)
        max_value = max(errors)

        # 创建每0.1为间隔的bin边界
        bins = np.arange(min_value, max_value + 0.005, 0.005)

        # 画出正态分布图
        # plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=bins, density=True, alpha=0.5, label=method)

        # plt.hist(errors, bins=30, density=True, alpha=0.5, label=method)
        # plt.hist(normal_distribution, bins=30, density=True, alpha=0.5, label=method)

    plt.xlabel('ERROR')
    plt.ylabel('FREQ')
    plt.xlim(-0.10,0.10)
    plt.legend()
    plt.show()


def smooth_curve(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return smoothed_data

def plot_error_lines(Dataset, line_alpha=0.3, line_width=2, smooth=True):
    # 获取误差数据
    error_data = calculate_error_data(Dataset)

    # 设置绘图参数
    plt.figure(figsize=(10, 6))

    # cmap = get_cmap('RdYlBu')
    # cmap = get_cmap('RdYlGn')
    # cmap = get_cmap('coolwarm_r')
    cmap = get_cmap('winter')

    i = 0
    # 绘制每个Method的误差直线
    for Method, error_array in error_data.items():
        color = cmap(i / len(error_data))
        # plt.plot(error_array, label=Method, alpha=line_alpha, linewidth=line_width, color=color)
        plt.plot(error_array[650:], alpha=line_alpha, linewidth=line_width, color=color)

        if smooth:

            smooth_error = smooth_curve(error_array[:, -1], 20)
            # plt.plot(x, smooth_error, label=f'{Method} (Smoothed)', alpha=1, linewidth=line_width)
            plt.plot(np.arange(len(error_array[650:])), smooth_error[650:], label=Method, alpha=1, linewidth=line_width, color=color)

        i += 1

        # 添加y=0的直线
    x = np.arange(len(error_array[650:]))
    y = np.zeros(len(error_array[650:]))
    plt.plot(x, y, label='Truth', linestyle='--', color='black', alpha=line_alpha, linewidth=line_width)


    # 设置标题和标签
    plt.title('Error Lines for Different Methods')
    plt.xlabel('Data Point')
    plt.ylabel('Error Value')
    if Dataset == 'Oxford':
        plt.ylim(0, 0.01)
    # 添加图例
    plt.legend(loc='best', ncol=2)

    # 创建 "plot" 文件夹
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # 保存图像为PDF格式，文件名包含数据集名称
    pdf_filename = f'plot/error_lines_{Dataset}.pdf'
    plt.savefig(pdf_filename, format='pdf')
    plt.show()


def plot_error_ridge(Dataset, line_alpha=0.3, line_width=2, smooth=False):
    # 获取误差数据
    error_data = calculate_error_data(Dataset)

    # 创建 "plot" 文件夹
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # 设置绘图参数
    plt.figure(figsize=(8, 6))

    # fig, ax = plt.subplots(figsize=(8, 6))

    # 设置调色板
    cmap = sns.color_palette("husl", n_colors=len(error_data))

    # 绘制每个Method的误差山脊图
    for i, (Method, error_array) in enumerate(error_data.items()):
        color = cmap[i]

        if smooth:
            smooth_error = smooth_curve(error_array[:, -1], 20)
            # sns.kdeplot(smooth_error[650:], label=f'{Method} (Smoothed)', color=color, alpha=line_alpha, linewidth=line_width, cumulative=True)
            if Dataset == 'MIT':
                sns.kdeplot(smooth_error[650:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)
            elif Dataset == 'nasa':
                sns.kdeplot(smooth_error[70:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)
            elif Dataset == 'Oxford':
                sns.kdeplot(smooth_error[20:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)

            else:
                sns.kdeplot(smooth_error, label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)

            # sns.kdeplot(smooth_error[650:], label=f'{Method} (Smoothed)', color=color, alpha=line_alpha,
            #             linewidth=line_width, ax=ax)

            # 使用 fill_between 函数填充区域
            # plt.fill_between(np.arange(len(smooth_error[650:])), np.zeros(len(smooth_error[650:])),smooth_error[650:], color=color,alpha=0.2)
        else:
            if Dataset == 'MIT':
                sns.kdeplot(error_array[:, -1][650:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)
            elif Dataset == 'nasa':
                sns.kdeplot(error_array[:, -1][70:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)
            elif Dataset == 'Oxford':
                sns.kdeplot(error_array[:, -1][10:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)
            elif Dataset == 'RESU':
                sns.kdeplot(error_array[:, -1][0:], label=f'{Method}', color=color, alpha=line_alpha, linewidth=line_width, shade=True)

    # 设置标题和标签
    plt.title(Dataset)
    plt.xlabel('Error Value')
    plt.ylabel('Density')

    # if Dataset == 'Oxford':
    #     plt.ylim(0, 0.01)

    # 添加图例
    plt.legend(loc='best', ncol=2)

    # 保存图像为PDF格式，文件名包含数据集名称
    pdf_filename = f'plot/ridge_plot_{Dataset}.pdf'
    plt.savefig(pdf_filename, format='pdf')
    plt.show()


def plot_error_ridge_3d(Dataset, line_alpha=0.5, line_width=2, smooth=True):
    # 获取误差数据
    error_data = calculate_error_data(Dataset)


    # Create an empty list to store the data
    data = []

    # Iterate through error_data and append each Method and corresponding values to the data list
    for Method, error_array in error_data.items():
        for value in error_array[650:]:
            data.append({'Method': Method, 'value': value[0]})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Define the list of Methods you want to filter
    Method_list = ['VIT', 'LSTM', 'MLP', 'CNN_LSTM', 'CNN']

    # Filter the DataFrame based on the Method list
    df_filtered = df[df['Method'].isin(Method_list)]
    # print(df_filtered['value'][789:799])
    # exit()

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 2})
    palette = sns.color_palette("husl", n_colors=len(error_data))
    g = sns.FacetGrid(df_filtered, palette=palette, row="Method", hue="Method", aspect=9, height=1.2)

    g.map_dataframe(sns.kdeplot, x="value", alpha=line_alpha, linewidth=line_width, shade=True)


    def label(x, color, label):
        ax = plt.gca()
        # ax.text(0, .2, label, color='black', fontsize=13,
        #         ha="left", va="center", transform=ax.transAxes)

    g.map(label, "Method")
    g.fig.subplots_adjust(hspace=-0.8)
    g.set_titles("")
    g.set(yticks=[], xlabel="Error Value",ylabel="",xlim=(-0.005, 0.015))
    g.despine(left=True)
    plt.suptitle('Netflix Originals - IMDB Scores by Language', y=0.98)


    plt.show()


def plot_ablation_error_lines(Dataset, line_alpha=0.3, line_width=2, smooth=True):
    # 获取误差数据
    error_data = calculate_ablation_error_data(Dataset)

    # 设置绘图参数
    plt.figure(figsize=(10, 6))

    # cmap = get_cmap('RdYlBu')
    # cmap = get_cmap('RdYlGn')
    # cmap = get_cmap('coolwarm_r')
    cmap = get_cmap('winter')

    i = 0
    # 绘制每个Method的误差直线
    for Method, error_array in error_data.items():
        color = cmap(i / len(error_data))
        # plt.plot(error_array, label=Method, alpha=line_alpha, linewidth=line_width, color=color)
        plt.plot(error_array[650:], alpha=line_alpha, linewidth=line_width, color=color)

        if smooth:

            smooth_error = smooth_curve(error_array[:, -1], 20)
            # plt.plot(x, smooth_error, label=f'{Method} (Smoothed)', alpha=1, linewidth=line_width)
            plt.plot(np.arange(len(error_array[650:])), smooth_error[650:], label=Method, alpha=1, linewidth=line_width, color=color)

        i += 1

        # 添加y=0的直线
    x = np.arange(len(error_array[650:]))
    y = np.zeros(len(error_array[650:]))
    plt.plot(x, y, label='Truth', linestyle='--', color='black', alpha=line_alpha, linewidth=line_width)


    # 设置标题和标签
    plt.title('Error Lines for Different Methods')
    plt.xlabel('Data Point')
    plt.ylabel('Error Value')
    if Dataset == 'Oxford':
        plt.ylim(0, 0.13)
    # 添加图例
    plt.legend(loc='best', ncol=2)

    # 创建 "plot" 文件夹
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # 保存图像为PDF格式，文件名包含数据集名称
    pdf_filename = f'plot/ablation_lines_{Dataset}.pdf'
    plt.savefig(pdf_filename, format='pdf')
    plt.show()

def plot_real_lines(Dataset, line_alpha=0.3, line_width=2, smooth=True):
    # 获取误差数据
    all_data, label_data = get_data(Dataset)
    # 设置绘图参数
    plt.figure(figsize=(10, 6))

    # cmap = get_cmap('RdYlBu')
    # cmap = get_cmap('RdYlGn')
    # cmap = get_cmap('coolwarm_r')
    cmap = get_cmap('winter')

    i = 0
    # 绘制每个Method的误差直线
    for Method, real_array in all_data.items():
        color = cmap(i / len(all_data))
        # plt.plot(error_array, label=Method, alpha=line_alpha, linewidth=line_width, color=color)
        # plt.plot(real_array, label=Method, linewidth=line_width, color=color)
        if Method == 'MIT':
            plt.plot(real_array[650:], label=Method, linewidth=line_width, color=color)
        else:
            plt.plot(real_array, label=Method, linewidth=line_width, color=color)

        i += 1

        # 添加y=0的直线


    if Method == 'MIT':
        plt.plot(label_data[650:], label='Truth', linestyle='--', color='black', alpha=line_alpha, linewidth=line_width)
    else:
        plt.plot(label_data, label='Truth', linestyle='--', color='black', alpha=line_alpha, linewidth=line_width)


    # 设置标题和标签
    plt.title('Error Lines for Different Methods')
    plt.xlabel('Data Point')
    plt.ylabel('Error Value')
    # if Dataset == 'Oxford':
    #     plt.ylim(0, 0.01)
    # 添加图例
    plt.legend(loc='best', ncol=2)

    # 创建 "plot" 文件夹
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # 保存图像为PDF格式，文件名包含数据集名称
    pdf_filename = f'plot/real_lines_{Dataset}.pdf'
    # plt.savefig(pdf_filename, format='pdf')
    plt.show()

def read_ablation_data(Dataset):
    # Store all loaded data in a dictionary

    if Dataset not in ['MIT', 'nasa', 'Oxford','RESU']:
        raise ValueError("Dataset必须是'MIT', 'nasa','RESU'或'Oxford'中的一个")
        # Construct the file path
    data_folder = os.path.join('..', Dataset, 'plot_data')
    # Initialize a sub-dictionary for each directory
    dir_data = {}

    # Load 'saved_x_before.pt'
    x_before_file_path = os.path.join(data_folder, 'saved_x_before.pt')
    if os.path.exists(x_before_file_path):
        x_before_data = torch.load(x_before_file_path)
        dir_data['saved_x_before'] = x_before_data

    # Load 'saved_x_after.pt'
    x_after_file_path = os.path.join(data_folder, 'saved_x_after.pt')
    if os.path.exists(x_after_file_path):
        x_after_data = torch.load(x_after_file_path)
        print(x_after_data)
        dir_data['saved_x_after'] = x_after_data

    return dir_data


def plot_vec_line(Dataset):

    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]

    # 创建一个新的图形
    plt.figure(figsize=(15, 3))

    # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
    for i in range(num_subplots):
        current_data = saved_x_before[i].detach().cpu().view(-1)

        # 转换为NumPy数组
        data_array = current_data.numpy()
        print(data_array)
        print("="*20)
        # 获取时间步和特征的数量
        num_time_steps = len(data_array)
        time_steps = np.arange(num_time_steps)
        # 绘制折线图，使用不同颜色
        colors = ['blue', 'green', 'orange', 'red', 'black']
        plt.plot(time_steps, data_array, color=colors[i], label=f'Subplot {i + 1}', marker='o', linestyle='-')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Combined Line Charts')
    plt.ylim(-1,1)
    plt.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_vec_attn_line(Dataset):

    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]
    saved_x_after = dir_data['saved_x_after'][-1]

    num_subplots = saved_x_before.shape[0]

    # 创建一个新的图形
    plt.figure(figsize=(15, 3))

    # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
    for i in range(num_subplots):
        current_data = (saved_x_before*saved_x_after)[i].detach().cpu().view(-1)

        # 转换为NumPy数组
        data_array = current_data.numpy()
        print(data_array)
        print("="*20)
        # 获取时间步和特征的数量
        num_time_steps = len(data_array)
        time_steps = np.arange(num_time_steps)
        # 绘制折线图，使用不同颜色
        colors = ['blue', 'green', 'orange', 'red', 'black']
        plt.plot(time_steps, data_array, color=colors[i], label=f'Subplot {i + 1}', marker='o', linestyle='-')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Combined Line Charts')
    plt.ylim(-1,1)
    plt.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_vec_bar(Dataset, vmin=0, vmax=16):

    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]

    # 创建一个新的图形
    plt.figure(figsize=(15, 3))

    # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
    for i in range(num_subplots):
        current_data = saved_x_before[i].detach().cpu().view(-1)

        # 转换为NumPy数组
        data_array = current_data.numpy()

        # 获取时间步和特征的数量
        num_time_steps = len(data_array)
        time_steps = np.arange(num_time_steps)
        print(data_array)
        print("="*10)
        # 绘制柱状图，使用不同颜色
        colors = ['blue', 'green', 'orange', 'red', 'black']
        plt.bar(time_steps + i * 0.2, data_array, color=colors[i], alpha=0.7, label=f'Subplot {i + 1}')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Combined Bar Charts')
    plt.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_heatmaps(Dataset, vmin=0, vmax=16):

    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]
    print(num_subplots)
    # 创建一个新的图形
    plt.figure(figsize=(15, 3))

    for i in range(num_subplots):
        plt.subplot(1, num_subplots, i + 1)

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = saved_x_before[i].detach().cpu().view(1,-1)

        # 转换为NumPy数组
        data_array = current_data.numpy()
        print(max(max(data_array)),data_array.shape)
        # 绘制当前子图的热力图
        plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Value')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.title(f'Subplot {i + 1}')


    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_sqrt_heatmaps(Dataset, vmin=0, vmax=16):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]
    print(num_subplots)

    # 创建一个新的图形
    plt.figure(figsize=(5*num_subplots, 3))

    for i in range(num_subplots):
        plt.subplot(1, num_subplots, i + 1)

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = saved_x_before[i].detach().cpu().view(1, -1)

        # 转换为NumPy数组
        data_array = current_data.numpy()

        reshape_size = int(np.sqrt(data_array.shape[1]))
        data_array = data_array[:,:reshape_size**2].reshape((reshape_size, reshape_size))

        # 绘制当前子图的热力图
        plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Value')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.title(f'Subplot {i + 1}')

    # 调整子图之间的间距
    plt.tight_layout()

    # # 添加直方图
    # plt.figure(figsize=(5, 3))
    # plt.hist(data_array.flatten(), bins=10, color='blue', alpha=0.7)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of Subplot {i + 1}')

    # 显示图形
    plt.show()

def plot_sqrt_attn_heatmaps(Dataset, vmin=0, vmax=0.01):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]
    saved_x_after = dir_data['saved_x_after'][-1]
    print(saved_x_after)
    num_subplots = saved_x_before.shape[0]


    # 创建一个新的图形
    plt.figure(figsize=(5*num_subplots, 3))

    for i in range(num_subplots):
        plt.subplot(1, num_subplots, i + 1)

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = (saved_x_before*saved_x_after)[i].detach().cpu().view(1, -1)

        # 转换为NumPy数组
        data_array = current_data.numpy()
        reshape_size = int(np.sqrt(data_array.shape[1]))
        data_array = data_array[:,:reshape_size**2].reshape((reshape_size, reshape_size))

        # 绘制当前子图的热力图
        plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Value')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.title(f'Subplot {i + 1}')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_sqrt_3D(Dataset):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]
    print(num_subplots)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 创建一个新的图形
    plt.figure(figsize=(5*num_subplots, 3))
    color_name = ['red','yellow','blue','green','black']
    for i in range(num_subplots):

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = saved_x_before[i].detach().cpu().view(1, -1)

        # 转换为NumPy数组
        data_array = current_data.numpy()

        reshape_size = int(np.sqrt(data_array.shape[1]))
        data_array = data_array[:,:reshape_size**2].reshape((reshape_size, reshape_size))
        # Create meshgrid for X and Y
        x = range(reshape_size)
        y = range(reshape_size)
        X, Y = np.meshgrid(x, y)

        # Plot the 3D contour
        # ax.plot_surface(X, Y, data_array, cmap='coolwarm')
        # ax.contourf(X, Y, data_array, cmap='coolwarm')
        ax.contourf(X, Y, data_array,colors = color_name[i])

    # Set the view perspective to look down along the Z-axis
    ax.view_init(elev=90, azim=0)
    # Set labels
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_zlabel('Value')
    ax.set_title('3D Contour Plot')

    plt.show()

def plot_sqrt_attn_3D(Dataset):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]
    saved_x_after = dir_data['saved_x_after'][-1]
    num_subplots = saved_x_before.shape[0]

    print(num_subplots)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 创建一个新的图形
    plt.figure(figsize=(5*num_subplots, 3))
    color_name = ['red','yellow','blue','green','black']
    for i in range(num_subplots):

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = (saved_x_before[i]*saved_x_after[i]/saved_x_after[0]).detach().cpu().view(1, -1)

        # 转换为NumPy数组
        data_array = current_data.numpy()

        reshape_size = int(np.sqrt(data_array.shape[1]))
        data_array = data_array[:,:reshape_size**2].reshape((reshape_size, reshape_size))
        # Create meshgrid for X and Y
        x = range(reshape_size)
        y = range(reshape_size)
        X, Y = np.meshgrid(x, y)

        # Plot the 3D contour
        # ax.plot_surface(X, Y, data_array, cmap='coolwarm')
        # ax.contourf(X, Y, data_array, cmap='coolwarm')
        ax.contourf(X, Y, data_array,colors = color_name[i])

    # Set the view perspective to look down along the Z-axis
    ax.view_init(elev=90, azim=0)
    # Set labels
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_zlabel('Value')
    ax.set_title('3D Contour Plot')

    plt.show()

def plot_error_sqrt_heatmaps(Dataset, vmin=-0.08, vmax=0.08):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]
    print(num_subplots)

    # 创建一个新的图形
    plt.figure(figsize=(5, 3*num_subplots))

    for i in range(num_subplots):
        current_data = saved_x_before[i].detach().cpu().view(1, -1)

        if i > 0:
            all_data = current_data + all_data
        else:
            all_data = current_data
    mean_data = all_data/num_subplots

    for i in range(num_subplots):
        plt.subplot(num_subplots, 1, i + 1)

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = saved_x_before[i].detach().cpu().view(1, -1)


        diff_data = current_data - mean_data


        # 转换为NumPy数组
        data_array = diff_data.numpy()

        reshape_size = int(np.sqrt(data_array.shape[1]))
        data_array = data_array[:, :reshape_size**2].reshape((reshape_size, reshape_size))

        # 绘制当前子图的热力图
        # plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        # plt.imshow(data_array, cmap='coolwarm', aspect='auto')

        plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)



        plt.colorbar(label='Value')

        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.title(f'Subplot {i + 1}')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_error_sqrt_attn_heatmaps(Dataset, vmin=-0.08, vmax=0.08):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]
    saved_x_after = dir_data['saved_x_after'][-1]
    num_subplots = saved_x_before.shape[0]

    # 创建一个新的图形
    plt.figure(figsize=(5, 3*num_subplots))
    for i in range(num_subplots):
        current_data = (saved_x_before[i]*saved_x_after[i]).detach().cpu().view(1, -1)

        if i > 0:
            all_data = current_data + all_data
        else:
            all_data = current_data
    mean_data = all_data/num_subplots

    for i in range(num_subplots):
        plt.subplot(num_subplots, 1, i + 1)

        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = (saved_x_before[i]*saved_x_after[i]).detach().cpu().view(1, -1)

        diff_data = current_data - mean_data


        # 转换为NumPy数组
        data_array = diff_data.numpy()

        reshape_size = int(np.sqrt(data_array.shape[1]))
        data_array = data_array[:, :reshape_size**2].reshape((reshape_size, reshape_size))

        # 绘制当前子图的热力图
        # plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        # plt.imshow(data_array, cmap='coolwarm', aspect='auto')

        plt.imshow(data_array, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)



        plt.colorbar(label='Value')

        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.title(f'Subplot {i + 1}')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_sqrt_maxcolor(Dataset):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]

    num_subplots = saved_x_before.shape[0]
    print(num_subplots)

    # 创建一个新的图形
    plt.figure(figsize=(4, 3))

    result_array = np.empty((num_subplots, saved_x_before.shape[-1]))

    for i in range(num_subplots):
        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = saved_x_before[i].detach().cpu().view(1, -1)

        # 转换为NumPy数组
        data_array = current_data.numpy()
        result_array[i] = data_array
# 在每列中找到最大值的索引
    max_indices = np.argmax(result_array, axis=0)

    # 创建新的数组，形状为（1，y），其中（1，i）的数值为result_array[:,i]中最大值的索引
    new_array = np.zeros((1, result_array.shape[1]))
    for i in range(result_array.shape[1]):
        new_array[0, i] = max_indices[i]
    print(new_array.shape)
    reshape_size = int(np.sqrt(new_array.shape[1]))
    new_array = new_array[:, :reshape_size**2].reshape((reshape_size, reshape_size))

    # 绘制plot_array的热力图
    plt.imshow(new_array, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Iteration')
    plt.xlabel('Time Step')
    plt.ylabel('Feature')
    plt.title('Maximum Value Iteration')
    plt.show()

def plot_sqrt_attn_maxcolor(Dataset):
    dir_data = read_ablation_data(Dataset)

    # 获取保存的数据
    saved_x_before = dir_data['saved_x_before'][-1]
    saved_x_after = dir_data['saved_x_after'][-1]
    num_subplots = saved_x_before.shape[0]

    # 创建一个新的图形
    plt.figure(figsize=(4, 3))

    result_array = np.empty((num_subplots, saved_x_before.shape[-1]))

    for i in range(num_subplots):
        # 取出当前位置的数据并将其从GPU复制到CPU并去掉梯度信息
        current_data = (saved_x_before[i]*saved_x_after[i]).detach().cpu().view(1, -1)

        # 转换为NumPy数组
        data_array = current_data.numpy()
        result_array[i] = data_array
    # 在每列中找到最大值的索引
    max_indices = np.argmax(result_array, axis=0)

    # 创建新的数组，形状为（1，y），其中（1，i）的数值为result_array[:,i]中最大值的索引
    new_array = np.zeros((1, result_array.shape[1]))
    for i in range(result_array.shape[1]):
        new_array[0, i] = max_indices[i]
    print(new_array.shape)
    reshape_size = int(np.sqrt(new_array.shape[1]))
    new_array = new_array[:, :reshape_size ** 2].reshape((reshape_size, reshape_size))

    # 绘制plot_array的热力图
    plt.imshow(new_array, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Iteration')
    plt.xlabel('Time Step')
    plt.ylabel('Feature')
    plt.title('Maximum Value Iteration')

    # 显示图形
    plt.show()

def plot_dual_axis_par():
    # 创建图形和轴

    attn_ratio = np.array([0.5964, 0.5713, 0.5991, 0.4931, 0.5953])
    true_I = np.array([4.950511265, 4.942469328, 4.949353361, 4.936464926, 4.943018974])
    # 计算皮尔逊相关系数
    correlation_coefficient = np.corrcoef(1-attn_ratio, true_I)[0, 1]
    print("Pearson Correlation Coefficient:", correlation_coefficient)

    attn_ratio_nor = (attn_ratio-min(attn_ratio))/(max(attn_ratio)-min(attn_ratio))
    true_I_nor = (true_I-min(true_I))/(max(true_I)-min(true_I))
    print(attn_ratio_nor-true_I_nor)
    # print((max(attn_ratio)-min(attn_ratio)),(max(true_I)-min(true_I)))
    #
    # attn_ratio = (attn_ratio-min(attn_ratio))/(max(attn_ratio)-min(attn_ratio))
    # true_I = (true_I-min(true_I))/(max(true_I)-min(true_I))
    fig, ax1 = plt.subplots()

    # 绘制第一个纵坐标的柱状图
    color = 'tab:blue'
    ax1.set_xlabel('Data Points')
    ax1.set_ylabel('Attention Ratio', color=color)
    ax1.bar([0,2,4,6,8], attn_ratio, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.48, 0.605)  # 设置第一个纵坐标轴的范围

    # 创建第二个纵坐标的轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('True I', color=color)
    ax2.bar([1,3,5,7,9], true_I, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(4.935, 4.951)  # 设置第二个纵坐标轴的范围
    pdf_filename = f'plot/ratio-par.pdf'
    plt.savefig(pdf_filename, format='pdf')

    # 显示图形
    plt.show()


def plot_dual_axis_ser():
    # 创建图形和轴

    # attn_ratio = np.array([0.4549,0.4766,0.522,0.5481,0.4369,0.4898,0.4289,0.5903])
    attn_ratio = np.array([0.5451,0.5234,0.478,0.4519,0.5631,0.5102,0.5711,0.4097])


    true_I = np.array([4.206,4.207,4.191,4.195,4.203,4.195,4.203,4.195])

    # 计算皮尔逊相关系数
    correlation_coefficient = np.corrcoef(attn_ratio, true_I)[0, 1]
    print("Pearson Correlation Coefficient:", correlation_coefficient)

    attn_ratio_nor = (attn_ratio-min(attn_ratio))/(max(attn_ratio)-min(attn_ratio))
    true_I_nor = (true_I-min(true_I))/(max(true_I)-min(true_I))
    print(attn_ratio_nor-true_I_nor)
    # print((max(attn_ratio)-min(attn_ratio)),(max(true_I)-min(true_I)))
    #
    # attn_ratio = (attn_ratio-min(attn_ratio))/(max(attn_ratio)-min(attn_ratio))
    # true_I = (true_I-min(true_I))/(max(true_I)-min(true_I))
    fig, ax1 = plt.subplots()

    # 绘制第一个纵坐标的柱状图
    color = 'tab:blue'
    ax1.set_xlabel('Data Points')
    ax1.set_ylabel('Attention Ratio', color=color)
    ax1.bar([0,2,4,6,8,10,12,14], attn_ratio, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.405, 0.576)  # 设置第一个纵坐标轴的范围

    # 创建第二个纵坐标的轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('True V', color=color)
    ax2.bar([1,3,5,7,9,11,13,15], true_I, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(4.19, 4.208)  # 设置第二个纵坐标轴的范围
    pdf_filename = f'plot/ratio-ser.pdf'
    plt.savefig(pdf_filename, format='pdf')
    # 显示图形
    plt.show()

# plot_vec_bar('nasa')


# plot_vec_line('Oxford')
# plot_vec_attn_line('nasa')
# plot_error_sqrt_heatmaps('MIT',vmin=-0.06,vmax=0.06)
# plot_error_sqrt_attn_heatmaps('MIT',vmin=-0.03,vmax=0.03)
#
# plot_error_sqrt_heatmaps('nasa',vmin=-0.06,vmax=0.06)
# plot_error_sqrt_attn_heatmaps('nasa',vmin=-0.03,vmax=0.03)
#
# plot_error_sqrt_heatmaps('Oxford',vmin=-0.06,vmax=0.06)
# plot_error_sqrt_attn_heatmaps('Oxford',vmin=-0.03,vmax=0.03)

# plot_sqrt_3D('nasa')
# plot_sqrt_attn_3D('nasa')


# plot_sqrt_maxcolor('nasa')
# plot_sqrt_attn_maxcolor('nasa')
# for Dataset in ['MIT']:

for Dataset in ['MIT','nasa','Oxford','RESU']:
    # plot_error_lines(Dataset)
    # plot_real_lines(Dataset)
    # plot_ablation_error_lines(Dataset)
    # plot_normal_distribution(Dataset)
    # read_ablation_data(Dataset)
    plot_error_ridge(Dataset)
plot_dual_axis_par()
plot_dual_axis_ser()
# plot_waterfall_bar()
# plot_error_ridge_3d('MIT')