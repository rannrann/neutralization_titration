import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import json

class DashApp:
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

        # 设置布局
        self.app.layout = html.Div([
            dcc.Graph(id='interactive-graph', figure=fig),
            dcc.Store(id='stored-data', data=[]),  # 用于存储全局状态的数据
            html.Div(id='output')
        ])

    def _setup_callbacks(self, file_path):
        """设置回调函数"""

        @self.app.callback(
            [Output('interactive-graph', 'figure'),
             Output('output', 'children'),
             Output('stored-data', 'data')],
            [Input('interactive-graph', 'clickData')],
            [State('stored-data', 'data')]  # 获取当前存储的数据
        )
        def display_click_data(clickData, stored_data):
            """点击事件的回调函数"""
            fig = go.Figure()

            # 重新创建图表
            fig.add_trace(go.Scatter(x=self.x, y=self.y, mode='markers', name='Normal Data'))
            
            if clickData:
                # 获取点击的 x 和 y 坐标
                clicked_x = clickData['points'][0]['x']
                clicked_y = clickData['points'][0]['y']

                # 将点击的 y 坐标追加到 stored_data
                stored_data.append(clicked_y)

                # 将 stored_data 保存到 JSON 文件中
                with open(file_path, 'w') as f:
                    json.dump(stored_data, f)
                
                # 打印当前存储的数据
                print(f"Y-axis values stored: {stored_data}")
                
                # 更新图表，标记点击的点
                fig.add_trace(go.Scatter(x=[clicked_x], y=[clicked_y], mode='markers', marker=dict(color='red', size=10), name='Anomalies'))

                # 返回更新后的图表、输出信息以及存储的 y 轴数据
                return fig, f"Marked point at: x={clicked_x}, y={clicked_y}", stored_data
            
            # 如果没有点击，返回原始图表
            return fig, "Click a point on the graph to mark it as an anomaly", stored_data

    def run(self):
        """运行应用"""
        self.app.run_server(debug=True)


# 创建 DashApp 实例并运行
if __name__ == '__main__':
    dash_app = DashApp()  # 创建类实例
    dash_app.run()        # 运行应用
