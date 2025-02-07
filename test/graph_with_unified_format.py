import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np


class graph_with_unified_format():
    """
    A class to display the data of the water experiment and the powder experient in unified format.

    Attributes:
    type: 0 = the water experiment; 1 = the powder experiment
    data_file: the location of raw data 
    """
    def __init__(self, type, data_file):
        df = pd.read_csv(data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
        self.data = pd.Series(data)
        self.type = type
        self.output_dir = "graph/united_format/"
        self.output_filename = data_file[6:-4] + ".jpg"
    
    def draw_graph(self):
        if self.type == 0:
            x_range = [0,33]
            x_dick = 1
            y_range = [0,11]
        if self.type == 1:
            x_range = [0, 150]
            x_dick = 10
            y_range = [0, 6]


        # 使用 Plotly 绘图
        fig = go.Figure()

        # 添加清理后的数据，保留原始索引
        fig.add_trace(go.Scatter(x=list(range(len(self.data))), y=self.data,
                                mode='markers+lines',  # 显示点和线
                                marker=dict(size=6),  # 设置点的大小
                                name='Weight Data'))  

        # 设置布局
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text='Time/s',  # X轴标题
                    font=dict(size=15)  # X轴标题字体大小
                ),
                tickfont=dict(size=15),
                range=x_range,  # 设置X轴范围 0~33
                tickmode="linear",  # 线性刻度
                dtick=x_dick  # X轴刻度间隔为1
            ),
            yaxis=dict(
                title=dict(
                    text='Weight/g',  # Y轴标题
                    font=dict(size=15)  # Y轴标题字体大小
                ),
                tickfont=dict(size=15),
                range=y_range,  # 设置Y轴范围 0~11
                tickmode="linear",  # 线性刻度
                dtick=1  # Y轴刻度间隔为1
            ),
            legend=dict(
                x=0.01,  # 图例左对齐到图表的 1% 位置
                y=0.99,  # 图例顶部对齐到图表的 99% 位置
                bgcolor="rgba(255,255,255,0.5)",  # 图例背景半透明
                bordercolor="Black",
                borderwidth=1,
                font=dict(size=15)  # 图例字体大小
            ),
            template='plotly_white'
        )

        # 显示图表
        fig.show()
        save_path = f"{self.output_dir}{self.output_filename}"
        fig.write_image(save_path)
        print(f"Graph saved as {save_path}")
        
        


# 示例文件路径
for i in range(45, 65):
    data_file = f'files/sample{i}.csv'
    g = graph_with_unified_format(1, data_file)
    g.draw_graph()
# data_file = 'files/sample48.csv'
# graph_with_original_index(data_file)