import numpy as np
import pandas as pd
import json

class feature_data():
    def __init__(self, data_file, shaking_interval_file):
        #, anomalies_indexes_file, shaking_interval_file, required_training_experiment
        self.data_file = data_file
        df = pd.read_csv(self.data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        self.data = np.array([float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])])
        self.gradient = np.gradient(self.data)

        with open(shaking_interval_file, 'r') as file:
            self.interval_indexes = list(dict.fromkeys(json.load(file)))
        self.shake_times = len(self.interval_indexes) / 2
        
        self.shake_start_time = []
        self.shake_duration = []
        self.total_shake_duration = 0
        self.initial_stopping_time = None
        self.final_stopping_time = None
        for i in range(0, len(self.interval_indexes), 2):
            self.shake_start_time.append(self.interval_indexes[i])
            duration = self.interval_indexes[i+1] - self.interval_indexes[i]
            self.shake_duration.append(duration)
            self.total_shake_duration += duration
        print("len(shake_start_time) = ", len(self.shake_start_time), ", len(shake_duration) = ", len(self.shake_duration))

data_file = 'files/三回目_revised.csv'
shaking_interval_file = 'dataset/shaking_interval_third.json'
feature_data(data_file, shaking_interval_file)