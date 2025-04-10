import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv("files/sample213.csv", header=None)
filtered_df = df.iloc[1:, :5]
data = [float(filtered_df.iloc[i, 4]) for i in range(filtered_df.shape[0])]
data = pd.Series(data)
is_collision = [0] * len(data)



end_index = None
start_point = None
collision_times = 0
collision_durations = []
for index in range(len(is_collision) - 1, 0, -1):
    if end_index == None and data[index - 1] - data[index] >= 0.003 :
        end_index = index
        collision_times += 1
        is_collision[index] = 1
    elif end_index != None:
        is_collision[index] = 1
        if data[index] <= data[end_index]:
            collision_durations.insert(0, (end_index - index) * 0.1)
            end_index = None

if end_index != None and data[end_index] >= data[0]:
    is_collision[0] = 1
    collision_durations.insert(0, end_index * 0.1)

time_intervals = np.arange(0, len(data) * 0.1, 0.1)
fig = go.Figure()
fig.add_trace(go.Scatter(x = time_intervals, y = data, mode = "markers+lines", 
                         marker= dict(size = 6, color = "blue"), name = 'Weight Data'))

x_collision_1 = [x for x, c in zip(time_intervals, is_collision) if c == 1]
y_collision_1 = [y for y, c in zip(data, is_collision) if c == 1]

fig.add_trace(go.Scatter(
    x = x_collision_1,
    y = y_collision_1,
    mode = 'markers',
    name = 'is_collision = 1',
    marker = dict(color = 'red'))
)

fig.update_layout(
    xaxis_title='X-axis',
    yaxis_title='Y-axis',
    showlegend=True
)

fig.show()

print("collision times = ",collision_times)
print("collision_durations = ", collision_durations)