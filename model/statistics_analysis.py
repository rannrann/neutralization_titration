import pandas as pd
import os
from scipy.stats import ttest_ind

def merge_data():
    data_dir = "model/features/"
    feature_files = [f"aggregated_feature_{i}.csv" for i in range(1, 15)]

    dataframes = [pd.read_csv(os.path.join(data_dir, file)) for file in feature_files]
    combined_df = pd.concat(dataframes, ignore_index=True) #Merge all DataFrames into one large one.ignore_index=True： Regenerate the index to ensure that the merged data index is continuous (starting from 0).

    combined_df['label'] = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    # 替换 NaN 值（示例：用0替换所有NaN）
    combined_df.fillna(0, inplace=True)

    # 查看修改后的数据
    print("After replacing NaN values:")
    print(combined_df.head())

    return combined_df


def statistics_analysis(combined_df):
    # 获取特征列和标签
    feature_columns = combined_df.columns[:-1]  # 去掉标签列
    group1 = combined_df[combined_df['label'] == 1]  # 知道中和点的组
    group0 = combined_df[combined_df['label'] == 0]  # 不知道中和点的组

    # 对每个特征进行t检验
    for feature in feature_columns:
        t_stat, p_value = ttest_ind(group1[feature], group0[feature])
        print(f"Feature: {feature}, t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

        if p_value < 0.05:
            print(f"--> {feature} 在两组之间有显著差异！")

combined_df = merge_data()
print("--------------------------------------------------------------------------")
statistics_analysis(combined_df)
print("--------------------------------------------------------------------------")
significant_features = ['last_stop', 'total_duration', 'mean_of_mean_gradient', 
                        'total_weight_change', 'skewness', 'kurtosis', 
                        'mean_gradient_mean', 'mean_gradient_std', 
                        'mean_gradient_max', 'max_gradient_mean', 'weight_change_max']

X = combined_df[significant_features]
y = combined_df['label']

# 训练简单的分类模型
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(len(y_test))

print("Accuracy:", accuracy_score(y_test, y_pred))
