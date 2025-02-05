import sys

import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.offsetbox import HPacker, TextArea, VPacker, AnnotationBbox
import pandas as pd
import json
import matplotlib.pyplot as plt
class MatplotlibWidget(FigureCanvas):
    def __init__(self, data, anomalies_indexes, data_series,parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        self.lef_mouse_pressed = False  

        self.data = np.array(data)
        self.anomalies_indexes = anomalies_indexes
        self.data_series = data_series
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.connect_event()

        self.init_annotation()
    


    def connect_event(self):
        self.mpl_connect("scroll_event", self.on_mouse_wheel)
        self.mpl_connect("button_press_event", self.on_button_press)
        self.mpl_connect("button_release_event", self.on_button_release)
        self.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.mpl_connect("motion_notify_event", self.hover)

    def compute_initial_figure(self):
        self.axes.plot(self.data, label='original datas', marker='o', linestyle='-', alpha=0.6)
        self.axes.plot(self.data_series.index, self.data_series, label='cleaned datas', marker='o', linestyle='-')
        self.axes.scatter(self.anomalies_indexes, self.data[self.anomalies_indexes], color='red', label='anormal datas', marker='x')
        # 设置 x 轴和 y 轴的标签，单位为秒和克
        self.axes.set_xlabel('Time (seconds)')  # 设置 x 轴为秒
        self.axes.set_ylabel('Weight (g)')  # 设置 y 轴为克
        self.axes.legend()  # 显示右上角标签
    def init_annotation(self):
        # 初始化光标线和注释
        self.vertline, = self.axes.plot([], [], 'c-', lw=2)
        hPackerList = [HPacker(children=[TextArea("", textprops=dict(size=10))])]  # 预置一个空文本显示横坐标值
        for line in self.axes.get_lines():
            if line == self.vertline:  # 跳过光标线
                continue
            line_color = line.get_color()   # 获取每条曲线的颜色
            text_area = TextArea(line.get_label(), textprops=dict(size=10, color=line_color))   #根据曲线颜色设置文字颜色
            hPacker = HPacker(children=[text_area])
            hPackerList.append(hPacker)
        self.text_box = VPacker(children=hPackerList, pad=1, sep=3) # 竖值布局，设置padding和文字之间上下的间距
        self.annotation_bbox = AnnotationBbox(self.text_box, (0, 0),
                                              xybox=(100, 0),
                                              xycoords='data',
                                              boxcoords="offset points")
        if self.axes is not None:
            self.axes.add_artist(self.annotation_bbox)

    def on_mouse_wheel(self, event):
        if self.axes is not None:
            x_min, x_max = self.axes.get_xlim()
            x_delta = (x_max - x_min) / 10
            y_min, y_max = self.axes.get_ylim()
            y_delta = (y_max - y_min) / 10
            if event.button == "up":
                self.axes.set(xlim=(x_min + x_delta, x_max - x_delta))
                self.axes.set(ylim=(y_min + y_delta, y_max - y_delta))
            elif event.button == "down":
                self.axes.set(xlim=(x_min - x_delta, x_max + x_delta))
                self.axes.set(ylim=(y_min - y_delta, y_max + y_delta))

            self.draw_idle()

    def on_button_press(self, event):
        if event.inaxes is not None:  # 判断是否在坐标轴内
            if event.button == 1:
                self.lef_mouse_pressed = True
                self.pre_x = event.xdata
                self.pre_y = event.ydata

    def on_button_release(self, event):
        self.lef_mouse_pressed = False

    def on_mouse_move(self, event):
        if event.inaxes is not None and event.button == 1:
            if self.lef_mouse_pressed:
                x_delta = event.xdata - self.pre_x
                y_delta = event.ydata - self.pre_y
                # 获取当前原点和最大点的4个位置
                x_min, x_max = self.axes.get_xlim()
                y_min, y_max = self.axes.get_ylim()

                x_min = x_min - x_delta
                x_max = x_max - x_delta
                y_min = y_min - y_delta
                y_max = y_max - y_delta

                self.axes.set_xlim(x_min, x_max)
                self.axes.set_ylim(y_min, y_max)
                self.draw_idle()

    def hover(self, event):
        if event.inaxes == self.axes:
            x = event.xdata
            if x is not None:
                text = f"x: {x}" #显示横坐标值
                hPacker_list = self.text_box.get_children()
                time_hPacker = hPacker_list[0]
                time_text_area: TextArea = time_hPacker.get_children()[0]
                time_text_area.set_text(text)   # 更新横坐标值
                for index,line in enumerate(self.axes.get_lines()):
                    if line == self.vertline:  # 跳过光标线
                        continue
                    # x_data = line.get_xdata()
                    # y_data = line.get_ydata()
                    # y = np.interp(x, x_data, y_data)

                    if index == 1:
                        if int(x) in self.data_series:
                            y = self.data_series[int(x)]
                        else:
                            y = None
                    elif index == 2:
                        if int(x) in self.anomalies_indexes:
                            y = self.anomalies[int(x)]
                        else:
                            y = None
                    else:
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        y = np.interp(x, x_data, y_data) 


                    # 更新光标线的位置
                    self.vertline.set_xdata([x, x])
                    self.vertline.set_ydata([self.axes.get_ylim()[0], self.axes.get_ylim()[1]])

                    if y is not None:
                        line_text = f"{line.get_label()}: {y:.3f}"
                    else:
                        line_text = f"{line.get_label()}: No Data"

                    # 显示每条曲线的详细信息
                    hPacker = hPacker_list[index+1] # 因为横坐标值放在了第一个hPacker中，所以从第二个开始
                    text_area: TextArea = hPacker.get_children()[0]
                    text_area.set_text(line_text)

                # 更新AnnotationBbox的位置
                self.annotation_bbox.xy = (x, event.ydata)
                self.annotation_bbox.set_visible(True)
                self.draw_idle()
            else:
                # 隐藏AnnotationBbox和光标线
                self.annotation_bbox.set_visible(False)
                self.vertline.set_xdata([])
                self.vertline.set_ydata([])
                self.draw_idle()

class MainWindow(QMainWindow):
    def __init__(self, data, anomalies_indexes):
        super().__init__()

        self.anomalies_indexes = np.sort(anomalies_indexes)
        self.data_series = pd.Series(data)
        self.data_series = self.data_series.drop(anomalies_indexes)

        self.widget = QWidget()
        self.setMinimumHeight(600)
        self.setMinimumWidth(800)
        self.showMaximized() # 设置全屏
        self.setCentralWidget(self.widget)

        layout = QVBoxLayout(self.widget)
        self.mpl_widget = MatplotlibWidget(data, self.anomalies_indexes, self.data_series,self.widget, width=5, height=4, dpi=100)
        layout.addWidget(self.mpl_widget)

        self.show()
    def non_interactive_graph(self):
        x = self.data_series.size
        y = self.data_series
        plt.plot(x, y)
        '''plt.axhline function in Matplotlib is used to add a horizontal line across the entire axis at a specified y-coordinate. 
        This can be useful for highlighting specific y-values, such as baselines or thresholds.'''
        plt.axhline(y=0, color='r', linestyle='--', label='y = 0') 
        y_ticks = np.arange(min(y) - 0.01, max(y) + 0.01, 0.01)
        x_ticks = np.arange(min(x) - 0.01, max(x) + 0.01, 0.01)
        plt.yticks(y_ticks)
        plt.xticks(x_ticks)
        plt.show()

        

if __name__ == "__main__":
    # 数据
    df = pd.read_csv('files/sample4.csv', header = None)
    filtered_df = df.iloc[1:, :5]
    data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]

    with open('dataset/stored_data4.json', 'r') as file:
        anomalies_indexes = json.load(file)


    app = QApplication(sys.argv)
    mainWin = MainWindow(data, anomalies_indexes)
    sys.exit(app.exec_())
