import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Define directories and files
data_dir = "model/features/"
files = [f"aggregated_feature_{i}.csv" for i in range(15, 25)]  # File names for test samples
labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # Corresponding y_test values

# Load and combine test samples
dfs = [pd.read_csv(os.path.join(data_dir, file)) for file in files]
test_df = pd.concat(dfs, ignore_index=True)

# Ensure test data aligns with training features
significant_features = ['last_stop', 'total_duration', 'mean_of_mean_gradient', 
                        'total_weight_change', 'skewness', 'kurtosis', 
                        'mean_gradient_mean', 'mean_gradient_std', 
                        'mean_gradient_max', 'max_gradient_mean', 'weight_change_max']
X_test = test_df[significant_features]  # Select only the significant features
y_test = np.array(labels)  # Actual labels

# Load the trained model
model = joblib.load("model/model/trained_model_11.joblib")

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Save results to a file
test_df["predictions"] = predictions
test_df["true_label"] = y_test
test_df.to_csv("test_results.csv", index=False)
print("Test results saved to test_results.csv.")
