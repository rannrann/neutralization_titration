import pandas as pd
import numpy as np
import json

def aggregated_features(file_name, index):
    # Load the JSON data
    with open(file_name, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Define the columns for aggregation
    segment_features = ['duration', 'shaking_duration', 'mean_gradient', 'max_gradient', 'weight_change']

    # Aggregate the segment features
    aggregated_features = df[segment_features].agg(['mean', 'std', 'min', 'max']).T
    aggregated_features.columns = ['mean', 'std', 'min', 'max']

    # Flatten the aggregated features into a single row
    flattened_aggregated = aggregated_features.values.flatten()

    # Add global features
    global_features = [
        df['first_stop'].iloc[0],
        df['last_stop'].iloc[0],
        df['total_duration'].iloc[0],
        df['mean_of_mean_gradient'].iloc[0],
        df['max_of_max_gradient'].iloc[0],
        df['total_weight_change'].iloc[0],
        df['skewness'].iloc[0],  # Ensure skewness exists in the input JSON
        df['kurtosis'].iloc[0]   # Ensure kurtosis exists in the input JSON
    ]

    # Combine global and aggregated features
    final_features = np.concatenate([global_features, flattened_aggregated])

    # Convert to a DataFrame for a consistent dataset
    feature_columns = ['first_stop', 'last_stop', 'total_duration', 'mean_of_mean_gradient',
                       'max_of_max_gradient', 'total_weight_change', 'skewness', 'kurtosis']
    feature_columns += [f'{col}_{stat}' for col in segment_features for stat in ['mean', 'std', 'min', 'max']]

    final_df = pd.DataFrame([final_features], columns=feature_columns)

    # Save the aggregated features
    output_file_name = f'model/features/aggregated_feature_{index}.csv'
    final_df.to_csv(output_file_name, index=False)
    print(f"Aggregated features saved to {output_file_name}")
    print(final_df)


for i in range(1, 15):
    file_name = 'model/features/extracted_features' + str(i) + '.json'
    aggregated_features(file_name, i)
