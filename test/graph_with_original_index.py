import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np

def graph_with_original_index(data_file, anomalies_indexes_file = None):
    # 读取数据
    df = pd.read_csv(data_file, header=None)
    filtered_df = df.iloc[1:, :5]
    data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]

    # 转换为 pandas Series
    data_series = pd.Series(data)

    # 读取异常点的索引
    if anomalies_indexes_file != None:
        with open(anomalies_indexes_file, 'r') as file:
            anomalies_indexes = list(dict.fromkeys(json.load(file)))
    
        # 准备清理后的数据
        cleaned_data = [data[i] if i not in anomalies_indexes else None for i in range(len(data))]
        
        # 准备异常点的数据
        anomalies_x = [i for i in anomalies_indexes]
        anomalies_y = [0] * len(anomalies_indexes)  # 异常点的 y 值为 0
    else:
        cleaned_data = data


    # 使用 Plotly 绘图
    fig = go.Figure()

    # 添加清理后的数据，保留原始索引
    fig.add_trace(go.Scatter(x=list(range(len(data))), y=cleaned_data,
                             mode='markers+lines',  # 显示点和线
                             marker=dict(size=6),  # 设置点的大小
                             name='Weight Data'))  

    # 添加异常点
    if anomalies_indexes_file != None:
        fig.add_trace(go.Scatter(x=anomalies_x, y=anomalies_y,
                                mode='markers',  # 只显示点
                                marker=dict(size=8, color='red'),  # 设置点为红色
                                name='Shaking Data'))  

    # 设置布局
    fig.update_layout(
        title=dict(
            text='Weight Data Combined with Shaking Data',  # 标题内容
            font=dict(size=30)  # 标题字体大小
        ),
        xaxis=dict(
            title=dict(
                text='Time/s',  # X轴标题
                font=dict(size=25)  # X轴标题字体大小
            ),
            tickfont=dict(size=25) 
        ),
        yaxis=dict(
            title=dict(
                text='Weight/g',  # Y轴标题
                font=dict(size=25)  # Y轴标题字体大小
            ),
            tickfont=dict(size=25) 
        ),
        legend=dict(
            x=0.01,  # 图例左对齐到图表的 1% 位置
            y=0.99,  # 图例顶部对齐到图表的 99% 位置
            bgcolor="rgba(255,255,255,0.5)",  # 图例背景半透明
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=20)  # 图例字体大小
        ),
        template='plotly_white'
    )

    # 显示图表
    fig.show()

# 示例文件路径
data_file = 'files/sample2.csv'
anomalies_indexes_file = 'dataset/stored_data2.json'
graph_with_original_index(data_file, anomalies_indexes_file)
# data_file = 'files/sample48.csv'
# graph_with_original_index(data_file)