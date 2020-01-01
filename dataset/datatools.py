import numpy as np

# retrieve dataset
glass_data = np.genfromtxt('dataset/glass_data.txt', delimiter=",")

def clean_dataset(dataset):
    '''
    Strip erroneous fields and further classify into 2 main classes [windowed glass = 1, non-windowed glass = 2]
    Returns cleaned numpy array
    '''
    # delete id column
    dataset = np.delete(dataset, 0, 1)

    # group into two classes [windowed = 1, non-windowed = 0]
    for x in dataset:
        if x[-1] <= 4.0:
            x[-1] = 1.0
        else:
            x[-1] = 0.0
    
    return dataset

def split_classes(dataset):
    '''
    Split dataset into two a dataset per class
    '''
    # classes: w1 = windowed, w2 = non-windowed
    w1 = np.empty((0, dataset.shape[1]))
    w2 = np.empty((0, dataset.shape[1]))

    # split into two classes
    for x in dataset:
        if x[-1] == 1.0:
            w1 = np.append(w1, [x], axis=0)
        else:
            w2 = np.append(w2, [x], axis=0)
    
    return w1, w2

def binarize(dataset):
    '''
    Binarize data by a mean threshold per feature
    feature > feature mean: 1, else: 0
    '''
    dataset = np.copy(dataset)
    for x in range(dataset.shape[1] - 1):
        mean = dataset[:,x].mean()
        for feature in range(len(dataset[:,x])):
            if dataset[:,x][feature] > mean:
                dataset[:,x][feature] = 1
            else:
                dataset[:,x][feature] = 0
    return dataset