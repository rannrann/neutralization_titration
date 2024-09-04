import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor

class weight_extractor:
    def __init__(self, weight_array):
        '''
        the commended order as follows:
        1.__init__
        2.deleteShake ----> generate self.weight_array_filtered---->deltasCreator ----> generate self.deltas self.deltas_indexes
        4.isDeltasOverThreshold
        5.extractedWeightWithoutShaking ----> regenerate  self.weight_array_filtered
        '''
        if not isinstance(weight_array, np.ndarray):
            weight_array = np.array(weight_array)
        self.weight_array = weight_array # original array
        self.indexes = np.array([i for i in range(self.weight_array.size)]) # original array indexes
        self.shakeOrNot = np.zeros(self.weight_array.size)
        self.deltas = []
        self.deltas_indexes = []
        self.weight_array_over_zero = None
        self.weight_array_without_abnormal = None
        self.weight_test = None

    def __deltasUpdate(self):
        '''
        deleteShake: when the element from the deltas over the threshold, the associated indexes need to be updated 
        '''
        
        indexes_deltas_update = np.where(self.shakeOrNot == 0)[0]
        weight_array_filtered = self.weight_array[indexes_deltas_update]
        self.deltas = np.diff(weight_array_filtered)
        self.deltas_indexes = []
        for i in range(1, indexes_deltas_update.size):
            self.deltas_indexes.append([indexes_deltas_update[i-1], indexes_deltas_update[i]])

    def drawGraph(self, x, y):
        plt.plot(x, y)
        '''plt.axhline function in Matplotlib is used to add a horizontal line across the entire axis at a specified y-coordinate. 
        This can be useful for highlighting specific y-values, such as baselines or thresholds.'''
        plt.axhline(y=0, color='r', linestyle='--', label='y = 0') 
        y_ticks = np.arange(min(y) - 0.01, max(y) + 0.01, 0.01)
        x_ticks = np.arange(min(x) - 0.01, max(x) + 0.01, 0.01)
        plt.yticks(y_ticks)
        plt.xticks(x_ticks)
        plt.show()

    
    def delete_abnormal_data(self, type,threshold = -0.1, window_size = 10,  n_neighbors=20, contamination=0.05,LOF_times = 10):
        '''
        type 1: It's suitable for limited data volume. It can be used after the first time of the neutralization titration
        type 2: It's suitable for larger data volume. It can be used for the first time
        '''
        try:
            self.weight_array_over_zero = self.weight_array[self.weight_array > 0]
            indexes_less_than_zero = np.where(self.weight_array < 0 )[0]
            self.shakeOrNot[indexes_less_than_zero] = 1
        except Exception as e:
            print("Error in deleting the data less than 0: {e}")
        else:
            print("Successing in deleting the data less than 0")

        indexes = np.array([i for i in range(self.weight_array_over_zero.size)])
        self.drawGraph(indexes, self.weight_array_over_zero)

        # Rolling Mean & Rolling Standard Deviation
        if type == 1:
            try:
                data_series = pd.Series(self.weight_array_over_zero)
                rolling_mean = data_series.rolling(window=window_size).mean()
                rolling_std = data_series.rolling(window=window_size).std()

                best_threshold = 1
                min_anomalies_count = len(data_series)
                for threshold in np.arange(1, 6, 0.1):
                    anomalies = data_series[(data_series - rolling_mean).abs() > threshold * rolling_std]
                    anomalies_count = len(anomalies)

                    if anomalies_count > 0 and anomalies_count < min_anomalies_count:
                        min_anomalies_count = anomalies_count
                        best_threshold = threshold

                print(f"The most suitable threshold multiple is：{best_threshold:.2f} standard deviation")
                anomalies = data_series[(data_series - rolling_mean).abs() > best_threshold * rolling_std]
                self.weight_array_without_abnormal = data_series.drop(anomalies.index)

            except Exception as e:
                print("Error in Rolling mean & Rolling standard : {e}")
            else:
                print("Success in Rolling mean & Rolling standard")
            indexes = np.array([i for i in range(self.weight_array_without_abnormal.size)])
            self.drawGraph(indexes, self.weight_array_without_abnormal)
            # LOF

            try:
                for i in range(LOF_times):
                    data_2D = np.array(self.weight_array_without_abnormal).reshape(-1, 1)
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                    outliers = lof.fit_predict(data_2D)
                    anomalies = self.weight_array_without_abnormal[outliers == -1]

                    if len(anomalies) == 0:
                        print(f"There is no more anomalous data after the {i} times iteration")

                    self.weight_array_without_abnormal = self.weight_array_without_abnormal.drop(anomalies.index)
            except Exception as e:
                print(f"Error in LOF : {e}")
            else:
                print("Success in LOF")

            self.weight_array_without_abnormal = self.weight_array_without_abnormal.tolist()
            self.weight_array_without_abnormal = np.array(self.weight_array_without_abnormal)
        if type == 2:
            '''
            used to calculate the delta between the two adjacent elements.
            return deltas list and the elements indexes
            e.q. deltas:[3]
                deltas_indexes: [[0,1]]
                The delta is 3 between the first element(index 0) and the second element(index 1)
            '''
            try:
                self.deltas = np.diff(self.weight_array_over_zero)
                for i in range(1, self.weight_array_over_zero.size):
                    self.deltas_indexes.append([i-1, i])
                self.deltas_indexes = np.array(self.deltas_indexes)
            except Exception as e:
                print("Error in creating deltas: {e}")
            else:
                print("Successing in creating deltas")
            
            '''
            Check if all the deltas is less than threshold. 
            If any, there is the data closing to shake. the delta and the indexes need to be updated   
            '''
            try:
                while True:
                    indexes_to_remove = np.where(self.deltas <= threshold)[0] #let's say original array has n elements. the deltas will have n-1 elements

                    if len(indexes_to_remove) == 0:
                        break


                    temp = np.zeros(self.deltas.size)
                    temp[indexes_to_remove] = 1
                        
                    for i in range(1, self.deltas.size):
                        if temp[i] == 1:
                            if temp[i-1] == 1:
                                self.shakeOrNot[self.deltas_indexes[i][0]] = 1
                            self.shakeOrNot[self.deltas_indexes[i][1]] = 1
                    self.__deltasUpdate()
            except Exception as e:
                print("Error in deleting the data closing to shake: {e}")
            else:
                print("Success in deleting the data closing to shake")

            
            index_without_shaking = np.where(self.shakeOrNot == 0)[0]
            self.weight_array_without_abnormal = self.weight_array[index_without_shaking]

            
            

    def z_scores(self, threshold = 5):
        self.weight_test = self.weight_array_over_zero 
        while True:
            deltas = np.diff(self.weight_test)
            z_scores = zscore(deltas)
            oscillations = np.abs(z_scores) > threshold
            indices_to_remove = np.where(oscillations)[0] + 1
            former_size = self.weight_test.size
            self.weight_test  = np.delete(self.weight_test, indices_to_remove)
            latter_size = self.weight_test.size
            if former_size == latter_size:
                break

        
            


