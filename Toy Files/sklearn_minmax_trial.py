# Python code to Rescale data (between 0 and 1)
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data1 = [[5],[7],[10],[6],[3],[4]]
data2 = [[105],[107],[110],[106],[103],[104]]
data2_norm = [[105/10],[107/10],[110/10],[106/10],[103/10],[104/10]]
scaler = MinMaxScaler()
print(f"data2_norm = {data2_norm}")
print(f"data 2 is {data2}")
print(f"scaler fit data1 is {scaler.transform(data1)}")
print(f"scaler fit  data2 is {scaler.transform(data2)}")


print(scaler.data_max_)

print(scaler.transform(data1))
