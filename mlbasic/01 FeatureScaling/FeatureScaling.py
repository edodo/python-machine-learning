import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

X = np.arange(-3,6).astype("float32").reshape(-1,1)
X

#MinMaxScale
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)
print(X_minmax)

#Standard Scaler
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
print(X_standard)