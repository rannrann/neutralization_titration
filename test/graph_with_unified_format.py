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
    def __init__(self, type, interval, x_dtick, y_range, data_file):
        df = pd.read_csv(data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
        self.data = pd.Series(data)
        self.type = type
        self.output_dir = "graph/united_format/"
        self.interval = interval
        self.x_dtick = x_dtick
        self.y_range = y_range
        self.gradient = np.gradient(self.data, self.interval)
        # print("len(self.data) = ", len(self.data))
        # print("len(self.gradient) = ", len(self.gradient))

        
    
    def draw_graph(self):
        self.output_filename = data_file[6:-4] + ".jpg"
        time_intervals = np.arange(0, len(self.data) * self.interval, self.interval)
     



        # 使用 Plotly 绘图
        fig = go.Figure()

        # 添加清理后的数据，保留原始索引
        fig.add_trace(go.Scatter(x=time_intervals, y=self.data,
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
                range=[0, max(time_intervals)],  # 设置X轴范围 0~33
                tickmode="linear",  # 线性刻度
                dtick=self.x_dtick  # X轴刻度间隔为1
            ),
            yaxis=dict(
                title=dict(
                    text='Weight/g',  # Y轴标题
                    font=dict(size=15)  # Y轴标题字体大小
                ),
                tickfont=dict(size=15),
                range=self.y_range,  # 设置Y轴范围 0~11
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
        
    def draw_gradient_graph(self):
        self.output_filename = data_file[6:-4] + "_gradient.jpg"
        time_intervals = np.arange(0, len(self.gradient) * self.interval, self.interval)


        # 使用 Plotly 绘图
        fig = go.Figure()

        # 添加清理后的数据，保留原始索引
        fig.add_trace(go.Scatter(x=time_intervals, y=self.gradient,
                                mode='markers+lines',  # 显示点和线
                                marker=dict(size=6),  # 设置点的大小
                                name='Speed Data'))  

        # 设置布局
        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text='Time/s',  # X轴标题
                    font=dict(size=15)  # X轴标题字体大小
                ),
                tickfont=dict(size=15),
                range=[0, max(time_intervals)],  # 设置X轴范围 0~33
                tickmode="linear",  # 线性刻度
                dtick=self.x_dtick  # X轴刻度间隔为1
            ),
            yaxis=dict(
                title=dict(
                    text='speed g/s',  # Y轴标题
                    font=dict(size=15)  # Y轴标题字体大小
                ),
                tickfont=dict(size=15),
                range=self.y_range,  # 设置Y轴范围 0~11
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
for i in range(175, 195):
    data_file = f'files/sample{i}.csv'
    interval = 0.05
    x_dtick = 1
    y_range = [0, 13]
    #y_range = [-10, 10]
    g = graph_with_unified_format(0, interval, x_dtick, y_range, data_file)
    '''
    sample25~44, type = 0
        interval = 1
        x_dtick = 1
        y_range = [0,11]

    sample45~65, type = 1
        interval = 1
        x_dtick = 10
        y_range = [0, 2]
        

    sample75~94, type=0
        interval = 0.2
        x_dtick = 1
        y_range = [0, 11]
    '''
    #g.draw_graph()
    g.draw_gradient_graph()
# data_file = 'files/sample48.csv'
# graph_with_original_index(data_file)