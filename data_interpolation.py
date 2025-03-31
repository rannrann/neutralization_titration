import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import shutil
def data_interpolation(data_file):
    backup_file = data_file[:-4] + "_copy.csv"
    shutil.copy(data_file, backup_file)  
    df = pd.read_csv(backup_file, nrows = 0)
    headers = list(df.columns)
    df = pd.read_csv(backup_file, header=None)
    filtered_df = df.iloc[1:, :5]
    y = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
    y = pd.Series(y)
    x = [i for i in range(1, len(y) + 1)]

    linear_interpolator = interpolate.interp1d(x, y, kind='linear')
    x_new = np.linspace(1, len(y), len(y) * 2 -1)
    y_new = linear_interpolator(x_new)


    new_rows = []
    for i, val in enumerate(y_new):
        # print(i+1, ":", val)
        new_row = [i+1] + [np.nan] * 3 + [val] + [np.nan]
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows, columns=headers)
    new_df.to_csv(data_file, index=False, header=True)

for i in [177, 178]:
    data_file = f"files/sample{i}.csv"
    #print(data_file)
    data_interpolation(data_file)