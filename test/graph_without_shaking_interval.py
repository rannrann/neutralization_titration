import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


def drawGraph(x, y):
    x_max = max(x) + 10
    y_max = max(y) + 5 
    x_min = min(x) - 1
    y_min = min(y) - 1
    # 绘制线图
    plt.plot(x, y)

    # 添加一条y=0的水平参考线
    plt.axhline(y=0, color='r', linestyle='--', label='y = 0')

    # 设置 x 轴和 y 轴的范围
    plt.xlim(x_min, x_max)  
    plt.ylim(y_min, y_max)  

    # 通过设置适当的刻度范围
    y_ticks = np.arange(y_min, y_max, 1)  
    x_ticks = np.arange(x_min, x_max, 10)  
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)

    # 显示图形
    plt.show()
# 数据
df = pd.read_csv('files/一回目.csv', header = None)
filtered_df = df.iloc[1:, :5]
data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]

with open('dataset/stored_data_first.json', 'r') as file:
    anomalies_indexes = json.load(file)

data = np.array(data)
anomalies_indexes = np.array(anomalies_indexes)
anomalies_indexes = np.sort(anomalies_indexes)
data_series = np.delete(data, anomalies_indexes)
x = np.array([i for i in range(data_series.size)])
y = data_series
drawGraph(x, y)
