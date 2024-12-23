import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import re
class graph_with_shaking_interval():
    def __init__(self, data_file, anomalies_indexes_file, shaking_interval_file):
        '''
        data_file: 重量数据文件
        anomalies_indexes_file: 摇晃锥形瓶期间的数据
        shaking_interval_file: 拿起和放下锥形瓶的时间点
        '''
        self.data_file = data_file
        df = pd.read_csv(self.data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        self.data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
        self.anomalies_indexes_file = anomalies_indexes_file
        self.anomalies_indexes = None
        self.interval_indexes = None
        self.horizontal_lines = []

        with open(self.anomalies_indexes_file, 'r') as file:
            self.anomalies_indexes = list(dict.fromkeys(json.load(file)))

        with open(shaking_interval_file, 'r') as file:
            self.interval_indexes = list(dict.fromkeys(json.load(file)))

        self.data_file, self.anomalies_indexes = self.revise_abnormous_data()

        with open(self.anomalies_indexes_file, 'r') as file:
            self.anomalies_indexes = list(dict.fromkeys(json.load(file)))
        df = pd.read_csv(self.data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        self.data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
        self.graph_with_shaking_interval2()
    

    def revise_abnormous_data(self):
        '''
        目的：处理既是间隔点又是异常点的数据; 为摇晃锥形瓶期间赋值假的重量数据，该数据为最近的锥形瓶重量
        既是间隔点又是异常点的数据将会被赋值为就近的非异常数据
        同时该异常点会被删除
        更新异常点的记录文件为stored_data_xxx_revised.json

        为摇晃锥形瓶期间赋值假的重量数据，该数据为最近的锥形瓶重量
        更新重量数据的记录文件为xxx_revised.csv
        '''
        index_loc = {}
        for i in range(len(self.interval_indexes)):
            e = self.interval_indexes[i]
            if e in self.anomalies_indexes and self.data[e] > -10:
                index_loc[e] = 0 if i % 2 == 0 else 1
        for index, loc in index_loc.items():
            if loc ==  0:
                direction = -1
                start_index = index
                end_index = 0
            if loc == 1:
                direction = 1
                start_index = index
                end_index = len(self.data)
            for revised_index in range(start_index + direction, end_index, direction):
                if revised_index not in self.anomalies_indexes:
                    self.anomalies_indexes.remove(index)
                    self.data[index] = self.data[revised_index]
                    break
        

        new_anomalies_indexes_file = self.anomalies_indexes_file[:-5] + '_revised' + self.anomalies_indexes_file[-5:]
        with open(new_anomalies_indexes_file, 'w') as file:
            json.dump(self.anomalies_indexes, file)

            

        #解决数据递减的问题 
        indices_to_adjust = []
        for i in range(0, len(self.interval_indexes), 2):
            for j in range(self.interval_indexes[i]+1, self.interval_indexes[i+1]+1):
                indices_to_adjust.append(j)
        prev_y = -1
        for i in range(len(self.data)):
            if i not in indices_to_adjust:
                if prev_y < self.data[i]:
                    prev_y = self.data[i]
                if prev_y > self.data[i]:
                    self.data[i]  = prev_y
            else:
                continue
        # pad the data within the interval
        for i in range(0, len(self.interval_indexes), 2):
            start_x = self.interval_indexes[i]
            end_x = self.interval_indexes[i + 1] if i + 1 < len(self.interval_indexes) else len(self.data) - 1
            prev_y = self.data[start_x] 

            # Set all points within the interval to prev_y
            for j in range(start_x, end_x + 1):
                self.data[j] = prev_y  # Fill interval with prev_y

        
        df = pd.read_csv(self.data_file)
        df.iloc[:1+len(self.data), 4] = self.data
        new_data_file = self.data_file[:-4] + '_revised' + self.data_file[-4:]
        df.to_csv(new_data_file, index=False)
        df.reset_index(drop=True, inplace=True)  # 重置索引
        return new_data_file, new_anomalies_indexes_file

    def graph_with_shaking_interval2(self):
        # 用 None 代替异常值，这样异常值的位置不会显示任何点
        cleaned_data = [self.data[i] if i not in self.anomalies_indexes else None for i in range(len(self.data))]

        # 用于存储蓝色线条的坐标
        blue_lines = []
        previous_end = (0, 0)  # 初始点设为 (0, 0)

        # 遍历 interval_indexes 中的偶数和下一个奇数索引，来绘制水平线
        for i in range(0, len(self.interval_indexes), 2):
            if i + 1 < len(self.interval_indexes):
                start_x = self.interval_indexes[i]  # 起点
                end_x = self.interval_indexes[i + 1]  # 终点
                prev_y = self.data[start_x]  # 使用起点的 y 值作为水平线的 y 值

                # 绘制水平红线
                self.horizontal_lines.append((start_x, end_x, prev_y))

                # 将前一根红线的终点与这一根红线的起点相连
                blue_lines.append((previous_end, (start_x, prev_y)))
                # 更新 previous_end 为当前红线的终点
                previous_end = (end_x, prev_y)

        # 使用 Plotly 绘图
        fig = go.Figure()

        # 添加红色水平线
        for start, end, y in self.horizontal_lines:
            fig.add_trace(go.Scatter(x=[start, end], y=[y, y],
                                     mode='lines',
                                     line=dict(color='red', width=3),  # 红色水平线
                                     showlegend=False))  # 不显示图例

        # 添加蓝色连接线
        for (x1, y1), (x2, y2) in blue_lines:
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2],
                                     mode='lines',
                                     line=dict(color='blue', width=2),  # 蓝色线条
                                     showlegend=False))  # 不显示图例

        # 设置布局
        fig.update_layout(
            title='Cleaned Data with Red Horizontal Lines and Blue Connecting Lines',
            xaxis_title='Original Index',
            yaxis_title='Data Value',
            template='plotly_white'
        )
        file_name_pattern = r'sample[0-9]*'
        file_name = re.findall(file_name_pattern, self.data_file)
        file_name = file_name[0]
        fig.write_html('graph/' + file_name + '.html')
        # 显示图表
        fig.show()


# 使用示例

# for i in range(15, 25):
#     data_file = 'files/sample' + str(i) + '.csv'
#     anomalies_indexes_file = 'dataset/stored_data' + str(i) + '.json'
#     shaking_interval_file = 'dataset/shaking_interval'+ str(i) + '.json'
#     g = graph_with_shaking_interval(data_file, anomalies_indexes_file, shaking_interval_file)
data_file = 'files/sample15.csv'
anomalies_indexes_file = 'dataset/stored_data15.json'
shaking_interval_file = 'dataset/shaking_interval15.json'
g = graph_with_shaking_interval(data_file, anomalies_indexes_file, shaking_interval_file)
