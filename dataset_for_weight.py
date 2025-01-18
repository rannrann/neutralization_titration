import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import json

class DashApp:
    '''
    选择标准：一定要把明显是摇晃阶段的数据选上
    样本间明显高于或是低于周围数值的数据选上
    '''
    def __init__(self, x, y, file_path):
        # 初始化 Dash 应用
        self.app = dash.Dash(__name__)

        # 定义数据
        self.x = x
        self.y = y

        # 初始化布局
        self._initialize_layout()

        # 设置回调函数
        self._setup_callbacks(file_path)

    def _initialize_layout(self):
        """初始化布局"""
        # 初始图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x, y=self.y, mode='markers', name='Normal Data'))
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(color='red', size=10), name='Anomalies'))

        # 设置布局，启用缩放功能
        fig.update_layout(
            xaxis=dict(fixedrange=False),  # 允许 x 轴缩放
            yaxis=dict(fixedrange=False)   # 允许 y 轴缩放
        )

        # 设置布局
        self.app.layout = html.Div([
            dcc.Graph(id='interactive-graph', figure=fig),
            dcc.Store(id='stored-data', data=[]),  # 用于存储全局状态的数据
            html.Div([
                html.Label('Y-axis range:'),
                dcc.Input(id='y-axis-min', type='number', placeholder='Min Y'),
                dcc.Input(id='y-axis-max', type='number', placeholder='Max Y'),
                html.Label('Y-axis tick interval:'),
                dcc.Input(id='y-axis-interval', type='number', placeholder='Y Interval', value=10),  # 控制y轴间隔

                html.Label('X-axis range:'),
                dcc.Input(id='x-axis-min', type='number', placeholder='Min X'),
                dcc.Input(id='x-axis-max', type='number', placeholder='Max X'),
                html.Label('X-axis tick interval:'),
                dcc.Input(id='x-axis-interval', type='number', placeholder='X Interval', value=100),  # 控制x轴间隔

                html.Button('Update Axis', id='update-axis', n_clicks=0)
            ]),
            html.Div(id='output')
        ])

    def _setup_callbacks(self, file_path):
        """设置回调函数"""

        @self.app.callback(
            [Output('interactive-graph', 'figure'),
             Output('output', 'children'),
             Output('stored-data', 'data')],
            [Input('interactive-graph', 'clickData'),
             Input('update-axis', 'n_clicks')],
            [State('stored-data', 'data'), 
             State('y-axis-min', 'value'), 
             State('y-axis-max', 'value'),
             State('y-axis-interval', 'value'),  # 获取y轴间隔
             State('x-axis-min', 'value'), 
             State('x-axis-max', 'value'),
             State('x-axis-interval', 'value')]  # 获取x轴间隔
        )
        def display_click_data(clickData, n_clicks, stored_data, y_min, y_max, y_interval, x_min, x_max, x_interval):
            """点击事件的回调函数"""
            fig = go.Figure()

            # 确保 stored_data 是字典
            if not isinstance(stored_data, dict):
                stored_data = {'x': [], 'y': []}

            # 重新创建图表
            fig.add_trace(go.Scatter(x=self.x, y=self.y, mode='markers', name='Normal Data'))

            # 如果点击事件触发
            if clickData:
                # 获取点击的 x 和 y 坐标
                clicked_x = clickData['points'][0]['x']
                clicked_y = clickData['points'][0]['y']

                # 将点击的异常点追加到 stored_data
                stored_data['x'].append(clicked_x)
                stored_data['y'].append(clicked_y)

                # 将 stored_data 保存到 JSON 文件中
                with open(file_path, 'w') as f:
                    json.dump(stored_data['x'], f)
                
                # 更新图表，标记点击的点
                fig.add_trace(go.Scatter(x=stored_data['x'], y=stored_data['y'], mode='markers', marker=dict(color='red', size=10), name='Anomalies'))

            # 更新图表范围：确保 x_min 和 x_max 存在，且设置了有效的 y 轴范围
            if x_min is not None and x_max is not None:
                fig.update_layout(xaxis=dict(range=[x_min, x_max], tickmode='linear', dtick=x_interval))  # 使用 dtick 控制 x轴间隔
            else:
                fig.update_layout(xaxis=dict(range=[min(self.x)-10, max(self.x)+10], tickmode='linear', dtick=x_interval))  # 默认保持初始范围

            if y_min is not None and y_max is not None:
                fig.update_layout(yaxis=dict(range=[y_min, y_max], tickmode='linear', dtick=y_interval))  # 使用 dtick 控制 y轴间隔
            else:
                fig.update_layout(yaxis=dict(range=[min(self.y)-10, max(self.y)+10], tickmode='linear', dtick=y_interval))  # 默认保持初始范围

            return fig, f"Marked point at: x={clicked_x}, y={clicked_y}" if clickData else "Click a point on the graph", stored_data


    def run(self):
        """运行应用"""
        self.app.run_server(debug=True)


# 创建 DashApp 实例并运行
if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('files/sample2.csv', header = None)
    filtered_df = df.iloc[1:, :5]
    y = np.array([float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]) 
    x = np.array([i for i in range(len(y))])
    output_file_path = "dataset/stored_data2.json"
    dash_app = DashApp(x, y, output_file_path)  # 创建类实例
    dash_app.run()
