import pandas as pd
import os
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib


def merge_data(data_dir, files_num, output_file):
    feature_files = [f"aggregated_feature_{i}.csv" for i in range(1, files_num + 1)]

    # Load and combine all feature files
    dataframes = [pd.read_csv(os.path.join(data_dir, file)) for file in feature_files]
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Add labels (customized to your data)
    combined_df['label'] = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]

    # Replace NaN values (example: replace NaN with 0)
    combined_df.fillna(0, inplace=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"Combined DataFrame saved to: {output_file}")
    print("Preview of combined DataFrame:")
    print(combined_df.head())

    return combined_df

def statistics_analysis(combined_df):
    print("----------------------------------statistics analysis----------------------------------------")
    # 获取特征列和标签
    feature_columns = combined_df.columns[:-1]  # 去掉标签列
    group1 = combined_df[combined_df['label'] == 1]  # 知道中和点的组
    group0 = combined_df[combined_df['label'] == 0]  # 不知道中和点的组

    # 对每个特征进行t检验
    for feature in feature_columns:
        t_stat, p_value = ttest_ind(group1[feature], group0[feature])
        print(f"Feature: {feature}, t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

        if p_value < 0.05:
            print(f"--> {feature} shows a significant difference between the two groups！")


def randomtree_model_training(significant_features, combined_df, output_route, output_file_name):
    print("----------------------------------model training----------------------------------------")
   
    X = combined_df[significant_features]
    y = combined_df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # 网格搜索
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数和最佳得分
    best_rf = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    print("----------------------------------matrix----------------------------------------")

    # Evaluate performance
    train_accuracy = accuracy_score(y_train, best_rf.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
    generalization_gap = train_accuracy - test_accuracy

    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(f"Generalization Gap: {generalization_gap:.2f}")

    if generalization_gap > 0.1:  # Threshold for gap, adjust based on your domain
        overfitting_message = "Potential overfitting detected."
        print(overfitting_message)
    else:
        overfitting_message = "Model generalizes well."
        print(overfitting_message)

    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    mean_cv_accuracy = scores.mean()
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.2f}")

    # Save the model to a file
    output_file_with_samples = output_route+ output_file_name + str(len(y_train)) + "_samples.joblib"
    joblib.dump(best_rf, output_file_with_samples)
    print("Model saved to ", output_file_with_samples)

    # 保存测试结果到指定路径
    log_file = output_route + "trainning_performance/" + output_file_name + str(len(y_train)) + "_samples.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 确保目录存在
    with open(log_file, "w") as f:
        f.write("----------------------------------Model Evaluation----------------------------------------\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best Accuracy from Grid Search: {grid_search.best_score_:.2f}\n\n")
        f.write(f"Training Accuracy: {train_accuracy:.2f}\n")
        f.write(f"Testing Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Generalization Gap: {generalization_gap:.2f}\n")
        f.write(f"{overfitting_message}\n\n")
        f.write(f"Cross-Validation Scores: {scores.tolist()}\n")
        f.write(f"Mean CV Accuracy: {mean_cv_accuracy:.2f}\n")
        f.write(f"Model saved to: {output_file_with_samples}\n")

    return model, len(y_train)



def decisiontree_model_training(significant_features, combined_df, output_route, output_file_name):
    print("----------------------------------model training----------------------------------------")
   
    X = combined_df[significant_features]
    y = combined_df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用决策树替代随机森林
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 5, 10, 20],            # 控制树的最大深度
        'min_samples_split': [2, 5, 10],          # 内部节点再分裂所需的最小样本数
        'min_samples_leaf': [1, 2, 4],            # 叶子节点的最小样本数
        'criterion': ['gini', 'entropy']          # 分裂标准
    }
    # 网格搜索
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数和最佳得分
    best_tree = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    print("----------------------------------matrix----------------------------------------")

    # Evaluate performance
    train_accuracy = accuracy_score(y_train, best_tree.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_tree.predict(X_test))
    generalization_gap = train_accuracy - test_accuracy

    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(f"Generalization Gap: {generalization_gap:.2f}")

    if generalization_gap > 0.1:  # Threshold for gap, adjust based on your domain
        overfitting_message = "Potential overfitting detected."
        print(overfitting_message)
    else:
        overfitting_message = "Model generalizes well."
        print(overfitting_message)

    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    mean_cv_accuracy = scores.mean()
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.2f}")

    # Save the model to a file
    output_file_with_samples = output_route + output_file_name + str(len(y_train)) + "_samples.joblib"
    joblib.dump(best_tree, output_file_with_samples)
    print("Model saved to ", output_file_with_samples)

    # 保存测试结果到指定路径
    log_file = output_route + "trainning_performance/" + output_file_name + str(len(y_train)) + "_samples.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 确保目录存在
    with open(log_file, "w") as f:
        f.write("----------------------------------Model Evaluation----------------------------------------\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best Accuracy from Grid Search: {grid_search.best_score_:.2f}\n\n")
        f.write(f"Training Accuracy: {train_accuracy:.2f}\n")
        f.write(f"Testing Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Generalization Gap: {generalization_gap:.2f}\n")
        f.write(f"{overfitting_message}\n\n")
        f.write(f"Cross-Validation Scores: {scores.tolist()}\n")
        f.write(f"Mean CV Accuracy: {mean_cv_accuracy:.2f}\n")
        f.write(f"Model saved to: {output_file_with_samples}\n")

    return model, len(y_train)
data_dir = "model_for_neutralization/features/"
files_num = 14
output_file = "model_for_neutralization/features/combined_features_14.csv"
combined_df = merge_data(data_dir, files_num, output_file)

statistics_analysis(combined_df)
significant_features = ['mean_of_mean_gradient', 'skewness', 'kurtosis', 'mean_gradient_max']
output_route = "model_for_neutralization/model/"
output_file_name = "trained_model_decisiontree_4_parameters_"
#model, training_sample_count = randomtree_model_training(significant_features, combined_df, output_route, output_file_name)
model, training_sample_count = decisiontree_model_training(significant_features, combined_df, output_route, output_file_name)





