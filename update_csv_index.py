import pandas as pd

def update_csv_index(file_path):
    df = pd.read_csv(file_path)
    df.iloc[:, 0] = range(1,len(df) + 1)
    df.to_csv(file_path, index=False)

for i in range(45, 75):
    file_path = "files/sample" + str(i) + ".csv"
    update_csv_index(file_path)
