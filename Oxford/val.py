import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgbb as xgb
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error


plt.rcParams['font.sans-serif'] = 'SimHei'  #显示中文
plt.rcParams['axes.unicode_minus'] = False  #显示负号
plt.rcParams['figure.dpi'] = 200  # 图像分辨率
plt.rcParams['text.color'] = 'black'  # 文字颜色
plt.style.use('ggplot')
print(plt.style.available)  # 可选的plt绘图风格
'''
['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
'''

import glob

csv_files = glob.glob('PRSA_data_*.csv')

df = pd.read_csv(csv_files[0],
                 index_col='No',
                 parse_dates={'datetime': [1,2,3,4]},
                 date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d %H')
                )

df.set_index('datetime',inplace=True)
df.head()

df.dropna(axis=0, how='any', inplace=True)
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 31815 entries, 2013-03-01 00:00:00 to 2017-02-28 23:00:00
Data columns (total 13 columns):
PM2.5      31815 non-null float64
PM10       31815 non-null float64
SO2        31815 non-null float64
NO2        31815 non-null float64
CO         31815 non-null float64
O3         31815 non-null float64
TEMP       31815 non-null float64
PRES       31815 non-null float64
DEWP       31815 non-null float64
RAIN       31815 non-null float64
wd         31815 non-null object
WSPM       31815 non-null float64
station    31815 non-null object
dtypes: float64(11), object(2)
memory usage: 3.4+ MB
'''
df.describe()
