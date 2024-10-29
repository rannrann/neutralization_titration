import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# 示例数据
data = np.random.normal(0, 1, (100, 2))  # 生成100个正常点
outliers = np.random.uniform(low=-6, high=6, size=(10, 2))  # 生成10个异常点
X = np.vstack([data, outliers])

# 假设我们有真实的标签：0表示正常，1表示异常
y_true = np.array([0] * 100 + [1] * 10)

# 定义LOF模型
lof = LocalOutlierFactor()

# 定义网格搜索的参数范围
param_grid = {
    'n_neighbors': [5, 10, 20, 30, 40],  # 搜索邻居数量
    'contamination': [0.05, 0.1, 0.15]  # 搜索污染比例
}

# 自定义打分函数，以f1_score为例
def lof_scorer(estimator, X, y_true):
    # 进行预测
    y_pred = estimator.fit_predict(X)
    # LOF的输出是-1表示异常，1表示正常，所以需要转换成0和1
    y_pred = np.where(y_pred == -1, 1, 0)
    return f1_score(y_true, y_pred)

# 设置评分器
scorer = make_scorer(lof_scorer, greater_is_better=True)

# 使用GridSearchCV进行参数优化
grid_search = GridSearchCV(lof, param_grid, scoring=scorer, cv=3)
grid_search.fit(X, y_true)

# 输出最佳参数组合和对应的评分
print("最佳参数组合：", grid_search.best_params_)
print("对应的F1分数：", grid_search.best_score_)
