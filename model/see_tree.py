import joblib

# 加载模型
model = joblib.load("model/model/trained_model_11.joblib")

trees = model.estimators_


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 可视化第一棵决策树
plt.figure(figsize=(20, 10))
plot_tree(trees[0], feature_names=model.feature_names_in_, filled=True)
plt.show()