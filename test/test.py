import pandas as pd

# 创建一个示例 DataFrame
data = {'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3]}

df = pd.DataFrame(data)

# 确保索引是默认的整数索引
print(df.index)

# 保存为 CSV 文件，不写入索引
df.to_csv('output.csv', index=False)
