import matplotlib.pyplot as plt
import numpy as np

# 示例数据
A = list(range(1, 23))
B = [0.007, -0.002, 0.016, 0.008, 0.021, 0.025, 0.006, -0.01, 0.029, -0.019, 0.008, 0.012, 0.005, 0.004, 0.041, -0.002, 0.03, 0.006, 0.011, 0.028, 0.046, -0.001]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(A, B, marker='o', linestyle='-', color='b', label='A vs B')

# 绘制 y=0 的线
plt.axhline(y=0, color='r', linestyle='--', label='y = 0')

# 添加标题和标签
plt.title('Line Chart of A vs B')
plt.xlabel('A (x-axis)')
plt.ylabel('B (y-axis)')

# 设置 y 轴刻度
y_ticks = np.arange(min(B) - 0.01, max(B) + 0.01, 0.01)
plt.yticks(y_ticks)

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
