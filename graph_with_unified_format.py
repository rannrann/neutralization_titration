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
        self.gradient[0] = 0
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
#195 196 197 198 199 200 201
#for i in range(195, 214): 
gradient_range = {195: [-3, 3], 196: [-3, 6], 197: [-9, 10], 198: [-1, 7], 199: [-5, 7], 200: [-3, 4],
        201: [-13, 15], 202: [-3, 5], 203:[-1, 7], 204: [-1, 4], 205: [-7, 8], 206: [-1, 4],
        207: [-5, 5], 208: [-3, 8], 209: [-1, 5], 210: [-1, 7], 211: [-11, 11], 212: [0, 4], 
        213: [0, 4]}
for key, value in gradient_range.items():
    data_file = f'files/sample{key}.csv'
    interval = 0.1
    x_dtick = 0
    y_range = value
    g = graph_with_unified_format(0, interval, x_dtick, y_range, data_file)
    #g.draw_graph()
    g.draw_gradient_graph()
# da#ta_file = 'files/sample48.csv'
# graph_with_original_index(data_file)