import numpy as np

def load_data(name):
    data = np.loadtxt(name)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    features = (features - means) / stds
    return labels, features

if __name__ == "__main__":
    name = input("Enter the name of the data file: ")
    labels, features = load_data(name)
    print(features.shape)