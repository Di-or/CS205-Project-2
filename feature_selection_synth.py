import numpy as np

data = np.loadtxt('CS170_Small_DataSet__7.txt')
labels = data[:, 0].astype(int)
features = data[:, 1:]


means = features.mean(axis=0)
stds = features.std(axis=0)
features = (features - means) / stds