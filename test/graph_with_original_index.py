import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np

def graph_with_original_index(data_file, anomalies_indexes_file):
    # 读取数据
    df = pd.read_csv(data_file, header=None)
    filtered_df = df.iloc[1:, :5]
    data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]

    # 转换为 pandas Series
    data_series = pd.Series(data)

    # 读取异常点的索引
    with open(anomalies_indexes_file, 'r') as file:
        anomalies_indexes = list(dict.fromkeys(json.load(file)))
    
    # 用 None 代替异常值，这样异常值的位置不会显示任何点
    cleaned_data = [data[i] if i not in anomalies_indexes else None for i in range(len(data))]

    # 使用 Plotly 绘图
    fig = go.Figure()

    # 添加清理后的数据，保留原始索引
    fig.add_trace(go.Scatter(x=list(range(len(data))), y=cleaned_data,
                             mode='markers+lines',  # 显示点和线
                             marker=dict(size=6),  # 设置点的大小
                             name='Cleaned Data'))

    # 设置布局
    fig.update_layout(
        title='Cleaned Data with Original Index',
        xaxis_title='Original Index',
        yaxis_title='Data Value',
        template='plotly_white'
    )

    # 显示图表
    fig.show()

data_file = 'files/一回目_revised.csv'
anomalies_indexes_file = 'dataset/stored_data_first_revised.json'
graph_with_original_index(data_file, anomalies_indexes_file)



