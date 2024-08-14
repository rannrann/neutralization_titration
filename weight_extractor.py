import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class weight_extractor:
    def __init__(self, weight_array):
        '''
        the commended order as follows:
        1.__init__
        2.deleteShake ----> generate self.weight_array_filtered
        3.deltasCreator ----> generate self.deltas self.deltas_indexes
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
        self.weight_array_filtered = None #original array without shaking

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

    
    def deleteShake(self):
        '''
        Whenever the element is negative, the operator is definitely shaking the wide-mouth jar.
        '''
        self.weight_array_filtered = self.weight_array[self.weight_array > 0]
        indexes_less_than_zero = np.where(self.weight_array < 0 )[0]
        print(indexes_less_than_zero)
        for index in indexes_less_than_zero:
            self.shakeOrNot[index] = 1

    def deltasCreator(self):
        '''
        used to calculate the delta between the two adjacent elements.
        return deltas list and the elements indexes
        e.q. deltas:[3]
            deltas_indexes: [[0,1]]
            The delta is 3 between the first element(index 0) and the second element(index 1)
        '''
        assert self.weight_array_filtered is not None
        for i in range(1, self.weight_array_filtered.size):
            self.deltas.append(self.weight_array_filtered[i] - self.weight_array_filtered[i - 1])
            self.deltas_indexes.append([i-1, i])
        self.deltas = np.array(self.deltas)
        self.deltas_indexes = np.array(self.deltas_indexes)
        return self.deltas, self.deltas_indexes

    def __deltasUpdate(self, index):
        '''
        when the element from the deltas over the threshold, the associated indexes need to be updated 
        '''
        assert index < self.deltas.size
        assert self.deltas_indexes is not None
        for i in range(index, self.deltas.size - 1):
            if i != index:
                self.deltas_indexes[i] = [self.deltas_indexes[i][0] + 1, self.deltas_indexes[i][1] + 1]
            else:
                self.shakeOrNot[self.deltas_indexes[i][1]] = 1
                self.deltas_indexes[i] = [self.deltas_indexes[i][0], self.deltas_indexes[i][1] + 1]
                
            self.deltas[i] = self.weight_array_filtered[self.deltas_indexes[i][1]] - self.weight_array_filtered[self.deltas_indexes[i][0]]
        self.deltas = self.deltas[:-1]
        self.deltas_indexes = self.deltas_indexes[:-1, :]

        # return self.deltas, self.deltas_indexes


    def isDeltasOverThreshold(self, threshold = -0.1):
        '''
        Check if all the deltas is less than threshold. 
        If any, there is the data closing to shake. the delta and the indexes need to be updated
        '''
        assert self.deltas is not None
        closeToShake = 0
        threshold = -0.1
        while closeToShake < self.deltas.size:
            if self.deltas[closeToShake] < threshold : 
                self.__deltasUpdate(closeToShake)
            closeToShake += 1
        # return self.deltas, self.deltas_indexes
        

    def extractedWeightWithoutShaking(self):
        '''
        if you execute this function, it means you have executed.
        the function returns the indexes without shaking
        '''
        assert self.deltas is not None
        assert self.weight_array_filtered is not None
        filter_indexes = []
        for index in self.deltas_indexes:
            filter_indexes.append(index[0])
            filter_indexes.append(index[1])
        filter_indexes = set(filter_indexes)
        filter_indexes = np.array(list(filter_indexes))
        self.weight_array_filtered = self.weight_array_filtered[filter_indexes]
            


