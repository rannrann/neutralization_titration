import plotly.graph_objects as go
import numpy as np

# 生成模拟重量数据（30 个数据点）
weight_data = np.array([ i for i in range(0, 11) ])
print(weight_data)
# 生成时间序列，每 0.2 秒间隔
time_intervals = np.arange(0, len(weight_data) * 0.2, 0.2)

# 创建 Plotly 图表
fig = go.Figure()

# 添加数据点（散点+线条）
fig.add_trace(go.Scatter(
    x=time_intervals, y=weight_data,
    mode='markers+lines',  # 显示点和线
    marker=dict(size=8, color='blue'),  # 点的样式
    line=dict(width=2),  # 线的样式
    name="Weight Data"
))

# 设置布局
fig.update_layout(
    title=dict(
        text="Time vs. Weight Data", 
        font=dict(size=25)  # 标题字体大小
    ),
    xaxis=dict(
        title="Time (s)",  
        tickmode="linear",  # 线性刻度
        dtick=1,  # X 轴每 1 秒显示一个刻度
        range=[0, max(time_intervals)],  # 设置 X 轴范围
        tickfont=dict(size=18)  # X 轴字体大小
    ),
    yaxis=dict(
        title="Weight (g)",  
        tickmode="linear",  
        dtick=1,  # Y 轴每 1g 显示一个刻度
        range=[0, 11],  # 设置 Y 轴范围
        tickfont=dict(size=18)  # Y 轴字体大小
    ),
    legend=dict(
        font=dict(size=15)  # 图例字体大小
    ),
    template="plotly_white"
)

# 显示图表
fig.show()
