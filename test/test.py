import pandas as pd

data = [i for i in range(50)]
interval = [5, 10, 23, 24, 44, 50]
drip_index = []

for i in range(0, len(interval), 2):
    #print("i = ", i, ", i+1 = ", i+1 )
    for j in range(interval[i]+1, interval[i+1] + 1):
        drip_index.append(j)
print(drip_index)