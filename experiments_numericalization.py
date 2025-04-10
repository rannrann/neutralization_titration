import pandas as pd
import plotly.graph_objects as go
import numpy as np

add_solution_times  = [6, 3, 1, 2, 3, 2, 2, 1, 3, 2, 4, 4, 1, 5, 3, 4, 3, 2, 3]
subjective_results = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
add_solution_durations = [[1.1, 0.6, 1.2, 1.6, 4.5, 0.5],
                          [5.5, 0.9, 0.9],
                          [6.3],
                          [3.1, 0.9],
                          [8.3, 0.8, 2],
                          [5.3, 1.8],
                          [6.9, 0.9],
                          [5.4],
                          [1.9, 2.6, 0.4],
                          [7.3, 1.1],
                          [9.1, 1.2, 1.4, 1.5],
                          [1.9, 4.8, 0.7, 1.3],
                          [6.5],
                          [3.5, 3.2, 0.6, 1.1, 0.4],
                          [3.4, 1.4, 0.4],
                          [3.3, 0.9, 0.5, 0.5],
                          [2.8, 1.5, 0.4],
                          [6.6, 0.4],
                          [4.5, 2.5, 0.5]]
x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
def draw_add_times_and_results():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = [i for i in x_data], 
                            y = add_solution_times, mode='markers+lines', 
                            marker=dict(size=6),
                            name = '溶液の入れ回数',
                            line=dict(color='royalblue', width=2),
                            yaxis = 'y1'))
    fig.add_trace(go.Bar(x = [i for i in x_data], 
                        y =  subjective_results,
                        marker=dict(color='rgba(255,140,0,0.4)'),
                        name = '結果',
                        width=0.4,
                        opacity=0.6, 
                        yaxis = 'y2'))

    fig.update_layout(
        title='水の秤量実験',
        xaxis=dict(title='実験の回数', dtick=1),
        yaxis=dict(title='溶液の入れ回数', side='left'),
        yaxis2=dict(title='結果', overlaying='y', side='right'),
        legend=dict(
            x=1,           # 靠右（0是最左，1是最右）
            y=1,           # 靠上（0是最下，1是最上）
            xanchor='right',  # x轴的对齐方式：以右边为锚点
            yanchor='top',    # y轴的对齐方式：以上边为锚点
        )
    )
    fig.show()