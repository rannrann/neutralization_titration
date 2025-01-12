import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 加载训练好的决策树模型
model = joblib.load("model/model/trained_model_decisiontree_4_parameters_11_samples.joblib")  # 这是一个单一决策树模型

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=model.feature_names_in_, class_names=["label_0", "label_1"], filled=True, rounded=True)
plt.title("Visualization of Decision Tree")
plt.show()
