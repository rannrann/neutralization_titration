import pandas as pd
import os
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
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


def model_training(significant_features, combined_df, output_file):
    print("----------------------------------model training----------------------------------------")
   
    X = combined_df[significant_features]
    y = combined_df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("----------------------------------matrix----------------------------------------")

    # Evaluate performance
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))



    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    generalization_gap = train_accuracy - test_accuracy
    print(f"Generalization Gap: {generalization_gap:.2f}")

    if generalization_gap > 0.1:  # Threshold for gap, adjust based on your domain
        print("Potential overfitting detected.")
    else:
        print("Model generalizes well.")

    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.2f}")

    # Save the model to a file
    output_file = output_file + str(len(y_train)) + ".joblib"
    joblib.dump(model, output_file)
    print("Model saved to ", output_file)

    return model, len(y_train)


data_dir = "model/features/"
files_num = 14
output_file = "model/features/combined_features_14.csv"
combined_df = merge_data(data_dir, files_num, output_file)

statistics_analysis(combined_df)
significant_features = ['last_stop', 'total_duration', 'mean_of_mean_gradient', 
                            'total_weight_change', 'skewness', 'kurtosis', 
                            'mean_gradient_mean', 'mean_gradient_std', 
                            'mean_gradient_max', 'max_gradient_mean', 'weight_change_max']

output_file = "model/model/trained_model_"
model, training_sample_count = model_training(significant_features, combined_df, output_file)






