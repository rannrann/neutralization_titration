import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

df = pd.read_csv("files/sample196.csv", header=None)
filtered_df = df.iloc[1:, :5]
data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
y_data = pd.Series(data)
# 仮のデータ（ここに実験データを入れる）
x_data = np.array([i*0.1 for i in range(1, 207)])
print(x_data)

# 対数関数の定義
def log_func(x, a, b, c, d):
    return a * np.log(b * x + c) + d

# 初期値（手動で設定してもOK）
initial_guess = [1.0, 1.0, 1.0, 0.0]

# フィッティング
params, covariance = curve_fit(log_func, x_data, y_data, p0=initial_guess)

# フィッティング結果の描画
x_fit = np.linspace(min(x_data), max(x_data), 300)
y_fit = log_func(x_fit, *params)

plt.plot(x_data, y_data, 'bo', label='実験データ')
plt.plot(x_fit, y_fit, 'r-', label='対数フィッティング')
plt.xlabel('時間')
plt.ylabel('フラスコの重さ')
plt.legend()
plt.title('行動曲線の対数フィッティング')
plt.show()

# フィッティングしたパラメータの表示
print("フィッティングパラメータ: a=%.3f, b=%.3f, c=%.3f, d=%.3f" % tuple(params))
