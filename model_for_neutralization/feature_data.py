import numpy as np
import pandas as pd
import json
from scipy.stats import skew, kurtosis

class feature_matrix():
    
    '''
    Target: figure out the differentiation between the neutralization point being known or not from shaking pattern
    Each segment has the parameters below
    
    '''

    class local_parameters:
        def __init__(self, data, segment):
            '''
            Features for Each Segment:
                Data
                Start Time(deleted)
                End Time(deleted)
                Duration(deleted)
                Shaking Duration 
                Shaking Percent(deleted)
                Mean Gradient
                Max Gradient
                Weight Change(deleted)
            '''
            self.data = None

            # self.start_time = None
            # self.end_time = None
            # self.duration = None
            self.shaking_duration = None
            #self.shaking_percent = None
            self.mean_gradient = None
            self.max_gradient = None
            #self.weight_change = None

            self.data = data
            #self.start_time = segment[0]
            #self.end_time = segment[1]
            #self.duration = segment[1] - segment[0] + 1
            self.shaking_duration = segment[2] - segment[0] + 1
            # print("duration: ", self.duration)
            # print("shaking duration: ", self.shaking_duration)

            #self.shaking_percent = self.shaking_duration / self.duration
            self.mean_gradient = np.mean(np.gradient(self.data))
            self.max_gradient = max(np.gradient(self.data))
            #self.weight_change = self.data[len(self.data) - 1] - self.data[0]
            # print("weight change: ", self.weight_change)


            
    
    def __init__(self, data_file, shaking_interval_file, nth):
        '''
        Global Parameters:
            First Stop
            Last Stop
            Total Duration
            Total shaking Duration 
            Absolute First Stop Point = First Stop / Total Duration
            absolute Last Stop Point = Last stop / Total Duration
            Absolute change = Total Shaking Duration / Total Duration 
            Mean of Mean Gradient 
            Max of Max Gradient
            Total Weight Change (deleted)
            Skewness
            Kurtosis
        

        For local paramenters:
            Segments: The range for splitting the whole event into multiple events based on shaking
            Segment Objs: Save the location parameters of each segment

        '''
        self.data = None
        self.interval_indexes = None
        self.first_stop = None
        self.last_stop = None
        self.total_duration = 0
        self.total_shaking_duration = 0

        self.absolute_first_stop_point = None
        self.absolute_last_stop_point = None
        self.absolute_change = None

        self.mean_of_mean_gradient = None
        self.max_of_max_gradient = None
        #self.total_weight_change = 0
        self.skewness = 0
        self.kurtosis = 0

        self.segments = [] 
        self.segment_objs = []
        self.global_parameters = []
        self.segment_parameters = []
        self.features = []
   
        self.feature_names = [
            "absolute_first_stop_point", "absolute_last_stop_point", "absolute_change", "mean_of_mean_gradient", 
            "max_of_max_gradient", "skewness", "kurtosis", "mean_gradient", "max_gradient"
        ]
        self.nth = nth

        '''
        The range for splitting the whole event into multiple events based on shaking
        Use these timestamps to define segments:
        Segment 1: From the first dripping stop to the second dripping stop.
        Segment 2: From the second dripping stop to the third dripping stop.
        ...
        Final Segment: From the last dripping stop to the experiment’s conclusion.

        each element be like, for example:
        interval_indexs:[1, 7, 10, 15 ]
        [a, b, c]
        a: the first dripping stop 1
        b: the second dripping stop  10
        c: the first returning stop 7 # It is used to calculate the local parameter, Shaking Duration.
        
        '''
        
        
        # get origin weight data
        df = pd.read_csv(data_file, header=None)
        filtered_df = df.iloc[1:, :5]
        self.data = np.array([float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])])

        #get interval range
        with open(shaking_interval_file, 'r') as file:
            self.interval_indexes = list(dict.fromkeys(json.load(file)))


        #Compute the total shaking duration
        for i in range(0, len(self.interval_indexes), 2):
            if i + 1 < len(self.interval_indexes):
                self.total_shaking_duration += (self.interval_indexes[i+1] - self.interval_indexes[i])
        

        # split the event up into multiple shaking events
        #The basis of grouping is 'Dripping Event + Shaking Event'.
        for i in range(0, len(self.interval_indexes), 2):
            if i + 2 < len(self.interval_indexes):
                self.segments.append([self.interval_indexes[i], self.interval_indexes[i + 2], self.interval_indexes[i + 1]])
        self.segments.append([self.interval_indexes[-2], self.interval_indexes[-1], self.interval_indexes[-1]])
        # first_data = 0
        # for i in range(0, len(self.interval_indexes), 2):
        #     if i + 1 < len(self.interval_indexes):
        #         self.segments.append([first_data, self.interval_indexes[i + 1], self.interval_indexes[i]])
        #         first_data = self.interval_indexes[i + 1]
        

        # data normalization
        min_val = min(self.data)
        max_val = max(self.data)
        self.data = [(x - min_val) / (max_val - min_val) for x in self.data]


        # print("interval indexes: ", self.interval_indexes)
        # print("segments: ", self.segments)
        self.create_segments()
        for segment_obj in self.segment_objs:
            self.segment_parameters.append([segment_obj.mean_gradient, segment_obj.max_gradient])
            self.total_shaking_duration += segment_obj.shaking_duration

        self.last_stop = self.interval_indexes[-2]


        #self.total_weight_change = self.data[self.last_stop] - self.data[self.first_stop]

        
        self.skewness = skew(self.data)
        self.kurtosis = kurtosis(self.data, fisher=True) 

        self.first_stop = self.interval_indexes[0]
        self.last_stop = self.interval_indexes[-2]
        self.total_duration = len(self.data)
        self.absolute_last_first_point = self.first_stop / self.total_duration
        self.absolute_last_stop_point = self.last_stop / self.total_duration
        self.absolute_change = self.total_shaking_duration / self.total_duration

        
        self.global_parameters = np.array([self.absolute_first_stop_point, self.absolute_last_stop_point, 
                                           self.absolute_change, self.mean_of_mean_gradient, self.max_of_max_gradient, 
                                           self.skewness, self.kurtosis])
  
        for segment_parameter in self.segment_parameters:
            segment_parameter_array = np.array(segment_parameter)  # Ensure it's a numpy array
            
            # Concatenate global and segment parameters
            combined_parameters = np.concatenate((self.global_parameters, segment_parameter_array))
            self.features.append(combined_parameters)
        self.features = np.array(self.features)
        df = pd.DataFrame(self.features, columns = self.feature_names)
        output_file_name = "model_for_neutralization/features/extracted_features" + str(self.nth) +".json"
        df.to_json(output_file_name, orient="records", indent=4)


        
    def create_segments(self):
        mean_gradient = []
        max_gradient = []
        
        for segment_obj in self.segments:
            data_for_segment = self.data[segment_obj[0]: segment_obj[1] + 1]
            segment_obj = self.local_parameters(data_for_segment, segment_obj)
            self.segment_objs.append(segment_obj)
            mean_gradient.append(segment_obj.mean_gradient)
            max_gradient.append(segment_obj.max_gradient)
        self.mean_of_mean_gradient = np.mean(mean_gradient)
        self.max_of_max_gradient = np.max(max_gradient)
  



for i in range(1,25):
    data_file = 'files/sample'+ str(i) +'_revised.csv'
    shaking_interval_file = 'dataset/shaking_interval' + str(i) + '.json'
    fm = feature_matrix(data_file, shaking_interval_file, i)
            
# data_file = 'files/sample1_revised.csv'
# shaking_interval_file = 'dataset/shaking_interval1.json'
# fm = feature_matrix(data_file, shaking_interval_file, 5)
