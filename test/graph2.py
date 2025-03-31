import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# 已知数据点 (x, y)
x = np.array([0, 1, 2, 3, 4])
y = np.array([5.009, 5.011, 5.013, 5.014, 5.015])

# 创建分段线性插值函数
linear_interpolator = interpolate.interp1d(x, y, kind='linear')

# 生成更细的 x 数据点进行插值
x_new = np.linspace(0, 4, 9)  # 生成 100 个点用于绘图
y_new = linear_interpolator(x_new)

# 绘图：已知数据点和插值结果
plt.plot(x, y, 'o', label='原始数据点')  # 原始数据点
plt.plot(x_new, y_new, 'o', label='分段线性插值曲线')  # 插值曲线
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('分段线性插值')
plt.show()
