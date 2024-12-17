import pandas as pd

file_path = "files/十四回目.csv"
df = pd.read_csv(file_path)
df.iloc[:, 0] = range(1,len(df) + 1)

output_file = 'files/十四回目.csv'
df.to_csv(output_file, index=False)
