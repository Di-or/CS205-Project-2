import numpy as np

def load_data(name):
    data = np.loadtxt(name)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    features = (features - means) / stds
    return labels, features

def nearest_neighbor(labels, features, selected_feats):
    n_data = len(labels)
    data = features[:, list(selected_feats)]
    correct_pred = 0

    for i in range(len(labels)):
        distances = np.sqrt(np.sum((data - data[i]) ** 2, axis=1))
        distances[i] = float('inf')
        nearest = np.argmin(distances)
        if labels[nearest] == labels [i]:
            correct_pred = correct_pred + 1
    
    return correct_pred / len(labels)


# Used a built in scikit learn KNN classifier on the
# small dataset to check if my NN implementation was correct
# and it was as both got 79.1% accuracy




if __name__ == "__main__":
    name = input("Enter the name of the data file: ")
    labels, features = load_data(name)
    selected_features = set(range(features.shape[1]))
    print(nearest_neighbor(labels, features, selected_features))