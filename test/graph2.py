import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np

def graph_with_original_index(data_file, anomalies_indexes_file):
    # 读取数据
    df = pd.read_csv(data_file, header=None)
    filtered_df = df.iloc[1:, :5]
    data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]

    # 读取异常点的索引
    with open(anomalies_indexes_file, 'r') as file:
        anomalies_indexes = list(dict.fromkeys(json.load(file)))
    
    # 用 None 代替异常值，这样异常值的位置不会显示任何点
    cleaned_data = [data[i] if i not in anomalies_indexes else None for i in range(len(data))]

    # 初始化水平线的起点和终点
    horizontal_lines = []
    prev_y = None  # 保存上一个非异常点的y值
    start_x = None  # 保存水平线的起始x点

    # 遍历数据，查找需要画水平线的区间
    for i in range(len(data)):
        if i in anomalies_indexes:
            # 如果遇到异常点，准备画水平线
            if prev_y is not None and start_x is None:
                start_x = i - 1  # 水平线从前一个非异常点的x值开始
        else:
            # 如果是正常点
            if start_x is not None:
                # 将水平线段添加到 horizontal_lines 中
                horizontal_lines.append((start_x, i, prev_y))
                start_x = None  # 重置水平线起点
            prev_y = data[i]  # 更新上一个非异常点的y值

    # 使用 Plotly 绘图
    fig = go.Figure()

    # 添加清理后的数据，保留原始索引
    fig.add_trace(go.Scatter(x=list(range(len(data))), y=cleaned_data,
                             mode='lines+markers',  # 显示点和线
                             marker=dict(size=6),  # 设置点的大小
                             name='Cleaned Data'))

    # 添加红色水平线
    for start, end, y in horizontal_lines:
        fig.add_trace(go.Scatter(x=[start, end], y=[y, y],
                                 mode='lines',
                                 line=dict(color='red', width=2),  # 红色水平线
                                 showlegend=False))  # 不显示图例

    # 设置布局
    fig.update_layout(
        title='Cleaned Data with Original Index and Red Horizontal Lines',
        xaxis_title='Original Index',
        yaxis_title='Data Value',
        template='plotly_white'
    )

    # 显示图表
    fig.show()

# 使用示例
data_file = 'files/一回目.csv'
anomalies_indexes_file = 'dataset/stored_data_first.json'
graph_with_original_index(data_file, anomalies_indexes_file)
