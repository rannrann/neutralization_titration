import pandas as pd
import numpy as np
import os
import joblib
from sklearn.tree import export_text, export_graphviz
from sklearn.metrics import accuracy_score, classification_report


from sklearn.metrics import classification_report, accuracy_score
import os

def test_model(model_name, test_df, X_test, y_test):
    # Make predictions
    model = joblib.load(model_name)
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    # 打印结果
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # 定义保存路径
    output_path = "model/model/test_performance/"
    os.makedirs(output_path, exist_ok=True)  # 确保目录存在

    # 文件名
    output_file = os.path.join(output_path, os.path.basename(model_name).replace(".joblib", ".txt"))
    
    # 保存结果到文件
    with open(output_file, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Test results saved to: {output_file}")


def output_trees(model_name, output_dir, X_test, y_test):
    model = joblib.load(model_name)
    os.makedirs(output_dir, exist_ok=True)
    # 存储预测正确的子模型
    correct_trees = []

    # 遍历随机森林中的所有子模型（决策树）
    for i, tree in enumerate(model.estimators_):
        # 使用子模型对测试集进行预测
        tree_predictions = tree.predict(X_test)
        
        # 检查该子模型是否对所有样本预测正确
        if (tree_predictions == y_test).all():
            print(f"Tree {i} predicts all test samples correctly!")
            correct_trees.append(tree)

            # 输出该树的规则（文本格式）
            tree_rules = export_text(tree, feature_names=list(X_test.columns))
            print(tree_rules)

            # 可视化（保存为 .dot 文件）
            dot_file = os.path.join(output_dir, f"correct_tree_{i}.dot")
            export_graphviz(
                tree,
                out_file=dot_file,
                feature_names=list(X_test.columns),
                class_names=["label_0", "label_1"],
                filled=True,
                rounded=True,
            )
            print(f"Tree {i} visualization saved to {dot_file}")

    # 检查预测正确的子模型数量
    print(f"Total number of correctly predicting trees: {len(correct_trees)}")


    
# Define directories and files
data_dir = "model/features/"
files = [f"aggregated_feature_{i}.csv" for i in range(15, 25)]  # File names for test samples
labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # Corresponding y_test values

# Load and combine test samples
dfs = [pd.read_csv(os.path.join(data_dir, file)) for file in files]
test_df = pd.concat(dfs, ignore_index=True)

# Ensure test data aligns with training features
significant_features = ['mean_of_mean_gradient', 'skewness', 'kurtosis', 'mean_gradient_max']
X_test = test_df[significant_features]  # Select only the significant features
y_test = np.array(labels)  # Actual labels


# Load the trained model
model_name = "model/model/trained_model_decisiontree_4_parameters_11_samples.joblib"

test_model(model_name, test_df, X_test, y_test,)
output_dir = "model/correct_trees/"
#output_trees(model_name, output_dir, X_test, y_test)

