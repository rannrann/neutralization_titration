import numpy as np
import plotly.graph_objects as go

# 原始数据
x = np.array([0, 10, 20, 30, 40])  # 原始 x 轴数据
y = np.array([100, 150, 130, 180, 160])  # 对应的 y 轴数据

# n 是每对相邻点之间要插入的新点数
n = 5

# 新的 x 和 y 数组，用来存储插值后的数据
new_x = []
new_y = []

# 遍历所有相邻点，插入 n 个点并进行插值
for i in range(len(x) - 1):
    # 取相邻两个点
    x_start, x_end = x[i], x[i + 1]
    y_start, y_end = y[i], y[i + 1]

    # 在 x_start 和 x_end 之间插入 n 个点
    interpolated_x = np.linspace(x_start, x_end, n + 2)  # 包括起点和终点
    interpolated_y = np.interp(interpolated_x, [x_start, x_end], [y_start, y_end])  # 线性插值

    # 将新生成的点添加到新数组中
    new_x.extend(interpolated_x)
    new_y.extend(interpolated_y)

# 将原始数据和插值数据绘制出来
fig = go.Figure()

# 绘制原始数据点
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Original Data', marker=dict(color='blue', size=10)))

# 绘制插值后的数据，连接红线
fig.add_trace(go.Scatter(x=new_x, y=new_y, mode='lines', name='Interpolated Data', line=dict(color='red', width=2)))

# 显示图表
fig.show()
