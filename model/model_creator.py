import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def combine_features_to_X(folder, output_file):
    """
    Combine multiple aggregated features files into a single matrix X.

    Parameters:
    - folder: str, the path to the folder containing aggregated features files.
    - output_file: str, the name of the output CSV file for the combined features.
    
    Returns:
    - X: pandas DataFrame, the combined features.
    """
    # Initialize an empty list to store DataFrames
    dataframes = []
    
    # Loop through all the files in the specified folder
    for filename in os.listdir(folder):
        if filename.startswith("aggregated_feature") and filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)
            
            # Read the file and append it to the list
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    # Concatenate all DataFrames into one
    X = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined DataFrame to a CSV file
    X.to_csv(output_file, index=False)
    print(f"Combined features saved to {output_file}")
    return X

# Combine all aggregated features
X = combine_features_to_X("model/features", "model/combined_features.csv")
print(X)

# Convert to numpy array
X = X.to_numpy()

# Handle NaN values in the numpy array
# Replace NaN with the column-wise mean
nan_indices = np.isnan(X)
col_means = np.nanmean(X, axis=0)
X[nan_indices] = np.take(col_means, np.where(nan_indices)[1])

# Labels
y = np.array([0, 1, 1, 0, 1, 1, 1])

# Stratified split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

X_train = X[0:3]
X_test = X[3:-1]
y_train = y[0:3]
y_test = y[3:-1]

# Classifier with balanced class weights
clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Predictions and metrics
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("--------------------------------------------------------------------------------------")


from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(class_weight="balanced", random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(scores)} ± {np.std(scores)}")
print("--------------------------------------------------------------------------------------")



import matplotlib.pyplot as plt

clf.fit(X, y)
feature_importances = clf.feature_importances_
plt.barh(range(len(feature_importances)), feature_importances)
plt.xlabel('Importance')
plt.ylabel('Feature Index')
plt.show()

print("--------------------------------------------------------------------------------------")

from sklearn.metrics import roc_curve

y_probs = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold:", optimal_threshold)