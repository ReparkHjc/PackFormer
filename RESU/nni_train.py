
search_space = {
    'batch_size': {'_type': 'choice', '_value': [2,4,6, 8, 10,12]},
    'decoderFc_size': {'_type': 'choice', '_value': [200,300,400, 500,600, 700, 800]},
    'depth': {'_type': 'choice', '_value': [2, 4, 6, 8]},
    'num_heads': {'_type': 'choice', '_value': [2, 5, 10, 20]},
    'embedding': {'_type': 'choice', '_value': [400, 600,800,1000, 1200]},
}

# 定义实验对象 选择local本地训练
from nni.experiment import Experiment
experiment = Experiment('local')


# 定义实验trial 命令 即运行model.py文件
experiment.config.trial_command = 'python auto_grid.py'
experiment.config.trial_code_directory = '.'

# 设置搜索空间
experiment.config.search_space = search_space

# 设置搜索算法
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# 设置搜索次数以及并行次数
experiment.config.max_trial_number = 10000
experiment.config.trial_concurrency = 1

# 运行实验 并在8090端口进行web展示
experiment.run(8090)


