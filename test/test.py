import numpy as np
import pandas as pd
import json

class feature_matrix():
    
    '''
    Each segment has the parameters below
    
    '''

    class local_parameters:
        def __init__(self, time_segment, shaking_or_not):
            '''
            Local Parameters:
                Average gradient
                Maximum gradient
                Minimum gradient
                Gradient variance
                Number of shakes
                Cumulative shaking time
                Time-weighted shaking time
            '''
            self.average_gradient = None
            self.maximum_gradient = None
            self.minimum_gradient = None
            self.gradient_variance = None
            self.number_of_shakes = None
            self.cumulative_shaking_time = None
            self.time_weighted_shaking_time = None
            
    
    def __init__(self, data_file, shaking_interval_file, trial_number_of_the_experiment, result_of_the_previous_experiment = 0):
        '''
        Global Parameters:
            Total shaking time
            Result of the previous experiment
            Trial number of the experiment
            Time when dripping stopped for the first time
            Time when dripping stopped for the last time
        '''
        self.data = None
        self.interval_indexes = None
        

        self.total_shaking_time = None
        self.result_of_the_previous_experiment = result_of_the_previous_experiment
        self.trial_number_of_the_experiment = trial_number_of_the_experiment
        self.time_when_dripping_stopped_for_the_first_time = None
        self.time_when_dripping_stopped_for_the_last_time = None


        df = pd.read_csv(data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        self.data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
        with open(shaking_interval_file, 'r') as file:
            self.interval_indexes = list(dict.fromkeys(json.load(file)))
        print(len(self.interval_indexes))

        self.total_shaking_time = 0
        for i in range(0, len(self.interval_indexes), 2):
            if i + 1 < len(self.interval_indexes):
                self.total_shaking_time += (self.interval_indexes[i+1] - self.interval_indexes[i])
        self.time_when_dripping_stopped_for_the_first_time = self.interval_indexes[0]

        index = len(self.interval_indexes) - 2
        self.time_when_dripping_stopped_for_the_last_time = self.interval_indexes[index]

data_file = 'files/一回目_revised.csv'
shaking_interval_file = 'dataset/shaking_interval_first.json'
fm = feature_matrix(data_file, shaking_interval_file, 1)