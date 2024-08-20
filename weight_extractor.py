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
        self.weight_array_over_zero = None #original array without shaking
        self.weight_array_over_threshold = None

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
        self.weight_array_over_zero = self.weight_array[self.weight_array > 0]
        indexes_less_than_zero = np.where(self.weight_array < 0 )[0]
        self.shakeOrNot[indexes_less_than_zero] = 1

    def deltasCreator(self):
        '''
        need to be canceled
        used to calculate the delta between the two adjacent elements.
        return deltas list and the elements indexes
        e.q. deltas:[3]
            deltas_indexes: [[0,1]]
            The delta is 3 between the first element(index 0) and the second element(index 1)
        '''
        assert self.weight_array_over_zero is not None
        self.deltas = np.diff(self.weight_array_over_zero)
        for i in range(1, self.weight_array_over_zero.size):
            self.deltas_indexes.append([i-1, i])
        self.deltas_indexes = np.array(self.deltas_indexes)

    def __deltasUpdate(self):
        '''
        when the element from the deltas over the threshold, the associated indexes need to be updated 
        '''
        # print("index = ",index, ", deltas.size = ", self.deltas.size)

        # assert index <= self.deltas.size
        # assert self.deltas_indexes is not None
        # for i in range(index, self.deltas.size - 1):
        #     if i != index:
        #         self.deltas_indexes[i] = [self.deltas_indexes[i][0] + 1, self.deltas_indexes[i][1] + 1]
        #     else:
        #         self.shakeOrNot[self.deltas_indexes[i][1]] = 1
        #         self.deltas_indexes[i] = [self.deltas_indexes[i][0], self.deltas_indexes[i][1] + 1]
                
        #     self.deltas[i] = self.weight_array_filtered[self.deltas_indexes[i][1]] - self.weight_array_filtered[self.deltas_indexes[i][0]]
        # self.deltas = self.deltas[:-1]
        # self.deltas_indexes = self.deltas_indexes[:-1, :]
        
        indexes_deltas_update = np.where(self.shakeOrNot == 0)[0]
        weight_array_filtered = self.weight_array[indexes_deltas_update]
        self.deltas = np.diff(weight_array_filtered)
        self.deltas_indexes = []
        for i in range(1, indexes_deltas_update.size):
            self.deltas_indexes.append([indexes_deltas_update[i-1], indexes_deltas_update[i]])



        
    def deleteTheElementWithDiffUnderThreshold(self, threshold = -0.1):
        '''
        replace to the function isDeltasOverThreshold and __deltasUpdate
        Check if all the deltas is less than threshold. 
        If any, there is the data closing to shake. the delta and the indexes need to be updated   
        '''
        assert self.deltas is not None
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
        
            # self.deltas = np.delete(self.deltas, indexes_to_remove)
            # self.deltas_indexes = np.delete(self.deltas_indexes, indexes_to_remove)
            
            self.__deltasUpdate()


    def extractedWeightWithoutShaking(self):
        '''
        if you execute this function, it means you have executed.
        the function returns the indexes without shaking
        '''
        assert self.deltas is not None
        assert self.weight_array_over_zero is not None
        index_without_shaking = np.where(self.shakeOrNot == 0)[0]
        self.weight_array_over_threshold  = self.weight_array[index_without_shaking]

            


