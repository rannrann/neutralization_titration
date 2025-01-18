import pandas as pd
import json
from datetime import datetime

class FeatureExtractor:
    def __init__(self, file_list, experiment_name):
        """
        Initialize the FeatureExtractor with a list of sample file paths.

        :param file_list: List of file paths for the samples.
        :param experiment_name: analyze which experiment
            -"water" : Water Weighing Experiment
            -"powder" : Powder Weighing Experiment
        """
        self.file_list = file_list
        self.all_sample_features = []  # To store features from all samples
        self.experiment_name = experiment_name

    def extract_features(self):
        """
        Extract both global and local features for each sample and save them as JSON files.
        """
        for i, file_path in enumerate(self.file_list, start=1):
            data = pd.read_csv(file_path)

            # Ensure 'Data' column is numeric
            '''
            In Pandas, the 'object' type is a general data type that is commonly used to store text data (strings). 
            Here, text data is extracted and converted into numeric values. The str.replace() method is used to remove strings like "123+".
            errors='coerce' -> If it encounters unconvertible values, those values will be replaced with NaN (missing values).
            '''
            if data['Data'].dtype == 'object':
                data['Data'] = data['Data'].str.replace('+', '')
            data['Data'] = pd.to_numeric(data['Data'], errors='coerce')

            # Convert Date and Time to a single Timestamp column
            data['Timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

            # Global Features
            global_features = self._extract_global_features(data)

            # Local Features
            local_features = self._extract_local_features(data)

            # Combine Features
            features = {
                "global_features": global_features,
                "local_features": local_features
            }

            # Save individual sample features
            self.all_sample_features.append(global_features)

            # Save to JSON file
            output_file = f"model_for_other_experiments/features/extracted_feature_for_{self.experiment_name}{i}.json"
            with open(output_file, 'w') as f:
                json.dump(features, f, indent=4, default=str)

            print(f"Features for sample {i} saved to {output_file}")

        # Aggregate global features across all samples
        aggregated_global_features = self._extract_aggregated_global_features()

        # Save aggregated features for all samples
        aggregated_output = {
            "aggregated_global_features": aggregated_global_features,
            "samples": self.all_sample_features
        }

        with open(f"model_for_other_experiments/features/aggregated_features_{self.experiment_name}.json", 'w') as f:
            json.dump(aggregated_output, f, indent=4, default=str)

        print("Aggregated features saved to aggregated_features.json")

    def _extract_global_features(self, data):
        """
        Extract global features from the dataset.

        :param data: Pandas DataFrame containing the sample data.
        :return: Dictionary of global features.
        """
        max_weight = data['Data'].max()
        min_weight = data['Data'].min()
        mean_weight = data['Data'].mean()
        std_weight = data['Data'].std()
        total_time = (data['Timestamp'].iloc[-1] - data['Timestamp'].iloc[0]).total_seconds()

        start_weight = data['Data'].iloc[0]
        end_weight = data['Data'].iloc[-1]
        average_growth_rate = (end_weight - start_weight) / total_time if total_time > 0 else 0

        state_counts = data['Header'].value_counts()
        state_ratios = (state_counts / len(data)).to_dict()

        state_switches = (data['Header'] != data['Header'].shift()).sum() - 1
        print(state_switches)
        return {
            "max_weight": max_weight,
            "min_weight": min_weight,
            "mean_weight": mean_weight,
            "std_weight": std_weight,
            "total_time": total_time,
            "average_growth_rate": average_growth_rate,
            "state_ratios": state_ratios,
            "total_records": len(data),
            "state_switches": state_switches
        }

    def _extract_local_features(self, data):
        """
        Extract local features from the dataset.

        :param data: Pandas DataFrame containing the sample data.
        :return: List of dictionaries, each representing local features for a data point.
        """
        local_features = []
        for idx, row in data.iterrows():
            local_features.append({
                "No.": row["No."],
                "Timestamp": row["Timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                "weight": row["Data"],
                "state": row["Header"]
            })
        return local_features

    def _extract_aggregated_global_features(self):
        """
        Extract global features aggregated across all samples.

        :return: Dictionary of aggregated global features.
        """
        max_weights = [f["max_weight"] for f in self.all_sample_features]
        min_weights = [f["min_weight"] for f in self.all_sample_features]
        avg_growth_rates = [f["average_growth_rate"] for f in self.all_sample_features]
        total_records = sum(f["total_records"] for f in self.all_sample_features)
        total_state_switches = sum(f["state_switches"] for f in self.all_sample_features)

        aggregated_features = {
            "overall_max_weight": max(max_weights),
            "overall_min_weight": min(min_weights),
            "overall_average_growth_rate": sum(avg_growth_rates) / len(avg_growth_rates),
            "total_records": total_records,
            "total_state_switches": total_state_switches
        }

        return aggregated_features

# Example usage
file_path = "files/sample"
file_list = []
for i in range(45, 65):
    file_list.append(file_path + str(i) + ".csv")

experiment_name = "powder"

extractor = FeatureExtractor(file_list, experiment_name)
extractor.extract_features()
