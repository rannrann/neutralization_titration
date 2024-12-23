import re

data_file = 'files/sample14.csv'

file_name_pattern = r'sample[0-9]*'
file_name = re.findall(file_name_pattern, data_file)
print(file_name[0])