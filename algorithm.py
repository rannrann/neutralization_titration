import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 数据
data = [0.483, 1.153, 1.839, 2.614, 3.266, 3.906, 4.58, 5.233, 5.988, 6.587, 7.196, 7.799, 8.519, 9.109, 9.679, 10.236, 
        10.418, 35.609, 10.418, 10.403, 10.433, 10.828, 22.834, 10.821, 10.828, 10.838, 10.843, 10.829, 10.837, 10.849, 
        10.87, 10.896, 10.851, 10.878, 10.948, 13.069, 10.959, 10.958, 10.966, 11.011, 11.018, 10.982, 11.028, 11.111, 
        11.116, 11.108, 11.096, 11.093, 11.221, 3.405, 11.412, 11.579, 11.9, 11.628, 11.645, 11.667, 11.71, 11.737, 
        11.728, 11.733, 11.75, 11.749, 11.767, 11.785, 11.795, 11.827, 11.781, 11.827, 11.866, 11.875, 11.903, 11.984, 
        11.884, 11.981, 13.814, 11.989, 11.98, 11.989, 11.971, 11.682, 11.995, 12.008, 12.032, 12.044, 12.019, 12.053, 
        12.059, 12.062, 12.076, 12.019, 12.009, 12.056, 12.053, 12.093]

# 转换为pandas Series
data_series = pd.Series(data)

# 设置移动平均线的窗口大小
window_size = 10

# 计算移动平均线和标准差
rolling_mean = data_series.rolling(window=window_size).mean()
rolling_std = data_series.rolling(window=window_size).std()

# 遍历不同的阈值倍数，从1到5倍标准差
best_threshold = 1
min_anomalies_count = len(data_series)
for threshold in np.arange(1, 6, 0.1):
    # 检测异常
    anomalies = data_series[(data_series - rolling_mean).abs() > threshold * rolling_std]
    # 计算异常点的数量
    anomalies_count = len(anomalies)
    
    # 判断是否为最合适的阈值
    if anomalies_count > 0 and anomalies_count < min_anomalies_count:
        min_anomalies_count = anomalies_count
        best_threshold = threshold

print(f"最合适的阈值倍数是：{best_threshold:.2f} 标准差")

# 使用找到的最佳阈值删除异常值
anomalies = data_series[(data_series - rolling_mean).abs() > best_threshold * rolling_std]
cleaned_data = data_series.drop(anomalies.index)

# 输出清洗后的数据
print("清洗后的数据:")
print(cleaned_data.tolist())


# ------------------------------LOF----------------------------------



# 转换为2D数组，因为LOF要求输入数据是二维的
data_2d = np.array(cleaned_data).reshape(-1, 1)

# 使用LOF进行异常检测
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outliers = lof.fit_predict(data_2d)

# 标记异常值
anomalies = cleaned_data[outliers == -1]

# 删除异常值
cleaned_cleaned_data = cleaned_data.drop(anomalies.index)

# 输出清洗后的数据
print("清洗后的数据:")
print(cleaned_cleaned_data.tolist())

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(data_series, label='原始数据', marker='o', linestyle='-', alpha=0.6)
plt.plot(cleaned_cleaned_data.index, cleaned_cleaned_data, label='清洗后的数据', marker='o', linestyle='-')
plt.scatter(anomalies.index, anomalies, color='red', label='检测出的异常值', marker='x')
plt.legend()
plt.title('基于LOF的异常检测与清理')
plt.show()
