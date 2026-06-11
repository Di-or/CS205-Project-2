import numpy as np

#Our code handles both our synthetic and real datasets as the only change needed is how the data is loaded

# loads data and z-normalizes it 
def load_data(name):
    if name.endswith('.csv'):
        data = np.loadtxt(name, delimiter=',', skiprows=1)
    else:
        data = np.loadtxt(name)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    features = (features - means) / stds
    return labels, features

# performs nearest neighbor algorithm with leave one out 
def nearest_neighbor(labels, features, selected_feats):
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
# and it was as both got the same accuracy

#performs forward selection algorithm to find the best subset of features with the NN classifier
def forward_selection(labels, features):
    curr_feats = set()
    best_acc = 0
    best_feats = set()
 
    print("\nBeginning search.\n")
 
    for i in range(features.shape[1]):
        best_curr_acc = 0
        best_feature = None
 
        for f in range(features.shape[1]):
            if f in curr_feats:
                continue
            test_feats = set(curr_feats)
            test_feats.add(f)
            curr_acc = nearest_neighbor(labels, features, test_feats)
            print("Using feature(s)", sorted(test_feats), "accuracy is", curr_acc * 100, "%")
 
            if curr_acc > best_curr_acc:
                best_curr_acc = curr_acc
                best_feature = f
 
        curr_feats.add(best_feature)
 
        if best_curr_acc < best_acc:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
 
        print("Feature set", sorted(curr_feats), "was best, accuracy is", best_curr_acc * 100, "%\n")
 
        if best_curr_acc > best_acc:
            best_acc = best_curr_acc
            best_feats = set(curr_feats)
 
    print("Finished search!! The best feature subset is", sorted(best_feats), "which has an accuracy of", best_acc * 100, "%")


#performs the backward elimination algorithm to find the best subset of features with the NN classifier
def backward_elimination(labels, features):
    curr_feats = set(range(features.shape[1]))
    best_acc = nearest_neighbor(labels, features, curr_feats)
    best_feats = set(curr_feats)

    print("\nRunning nearest neighbor using all features achieves an accuracy of", best_acc * 100, "%")
    print("\nBeginning search.\n")

    for i in range(features.shape[1] - 1):
        best_curr_acc = 0
        worst_feature = None

        for f in sorted(curr_feats):
            test_feats = set(curr_feats)
            test_feats.remove(f)
            curr_acc = nearest_neighbor(labels, features, test_feats)
            print("Using feature(s)", sorted(test_feats), "accuracy is", curr_acc * 100, "%")

            if curr_acc > best_curr_acc:
                best_curr_acc = curr_acc
                worst_feature = f

        curr_feats.remove(worst_feature)

        if best_curr_acc < best_acc:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        print("Feature set", sorted(curr_feats), "was best, accuracy is", best_curr_acc * 100, "%\n")

        if best_curr_acc > best_acc:
            best_acc = best_curr_acc
            best_feats = set(curr_feats)

    print("Finished search!! The best feature subset is", sorted(best_feats), "which has an accuracy of", best_acc * 100, "%")


if __name__ == "__main__":
    name = input("Enter the name of the data file: ")
    labels, features = load_data(name)
    selected_features = set(range(features.shape[1]))
    print(nearest_neighbor(labels, features, selected_features))
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    algo = input("\n")
    if algo == "1":
        forward_selection(labels, features)
    elif algo == "2":
        backward_elimination(labels, features)