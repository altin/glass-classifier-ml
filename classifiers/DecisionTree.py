import numpy as np
from dataset import datatools
from .DecisionTreeNode import Node
from sklearn.model_selection import KFold

def entropy(dataset):
    w1, w2 = datatools.split_classes(dataset)

    if len(dataset) == 0:
        return 0
    
    p_w1 = w1.size / dataset.size
    p_w2 = 1 - p_w1
    
    if p_w1 == 0: p_w1 = 1
    elif p_w2 == 0: p_w2 = 1

    return -p_w1 * np.log2(p_w1) - p_w2 * np.log2(p_w2)

def gain(dataset):
    # [attribute, information gain]
    max_info_gain = {'idx': None, 'gain': -np.inf}

    for x in range(dataset.shape[1] - 1):
        v1 = np.empty((0, 2))
        v2 = np.empty((0, 2))

        for feature, w in zip(dataset[:,x], dataset[:,-1]):
            if feature == 1:
                v1 = np.append(v1, np.array([[feature, w]]), axis=0)
            else:
                v2 = np.append(v2, np.array([[feature, w]]), axis=0)
        
        weight_v1 = entropy(v1) * v1.size / dataset.size
        weight_v2 = entropy(v2) * v2.size / dataset.size
        curr_gain = entropy(dataset) - (weight_v1 + weight_v2)

        if curr_gain > max_info_gain['gain']:
            max_info_gain['idx'] = x
            max_info_gain['gain'] = curr_gain

            if weight_v1 + weight_v2 == 0:
                max_info_gain['gain'] = np.inf
    
    # print("Max entropy: " + str(max_info_gain['idx']))
    return max_info_gain

def get_children(parent, depth=0):
    p_w1 = np.count_nonzero(parent.data['data'] == 1.0) / parent.data['data'].size
    p_w2 = 1 - p_w1

    if depth != 0 and parent.data['data'].shape[1] == 2 or parent.data['attr']['gain'] == np.inf:
        if parent.data['data'].shape[1] == 2:
            parent.data['decision'] = np.ceil(np.maximum(p_w1, p_w2))
        else:
            parent.data['decision'] = parent.data['data'][0][-1]
        return

    left = np.empty((0, parent.data['data'].shape[1]))
    right = np.empty((0, parent.data['data'].shape[1]))

    # create left and right subsets
    for x in parent.data['data']:
        if x[parent.data['attr']['idx']] == 0:
            left = np.append(left, [x], axis=0)
        else:
            right = np.append(right, [x], axis=0)
    
    # drop parent columns
    left = np.delete(left, parent.data['attr']['idx'], 1)
    right = np.delete(right, parent.data['attr']['idx'], 1)

    # edge case when all remaining 
    if len(left) == 0:
        parent.left = Node({'attr': None, 'data': left, 'decision': int(not np.ceil(np.maximum(p_w1, p_w2))) / 1.0})
    if len(right) == 0:
        parent.right = Node({'attr': None, 'data': right, 'decision': int(not np.ceil(np.maximum(p_w1, p_w2))) / 1.0})

    if left.size != 0:
        parent.left = Node({'attr': gain(left), 'data': left, 'decision': None})
        get_children(parent.left, depth + 1)
    if right.size != 0:
        parent.right = Node({'attr': gain(right), 'data': right, 'decision': None})
        get_children(parent.right, depth + 1)
    
def build_tree(dataset):
    root = Node({'attr': gain(dataset), 'data': dataset, 'decision': None})
    get_children(root)
    return root

def classify(x, tree):
    while tree != None:
        if tree.data['decision'] != None:
            return tree.data['decision']
        if x[tree.data['attr']['idx']] == 0:
            tree = tree.left
        elif x[tree.data['attr']['idx']] == 1:
            tree = tree.right

def train_test(dataset, display=False, folds=5):
    '''
    Train and test using k-fold cross validation
    Returns accuracy
    '''
    # 5-fold cross-validation
    kfold = KFold(folds, True, 1)
    accuracy = 0
    fold_count = 0

    for train, test in kfold.split(dataset):
        curr_predicted = 0
        fold_count += 1
        tree = build_tree(dataset[train])

        # test
        for x in dataset[test]:
            labelled = np.copy(x)
            unlabelled = np.delete(x, -1)
            k = classify(unlabelled, tree)
            if k == 1 and labelled[-1] == 1:
                curr_predicted += 1
            elif k == 0 and labelled[-1] == 0:
                curr_predicted += 1

        if display:
            tree.display()
        
        print("\t\tFold #" + str(fold_count) + ": " + str(curr_predicted) + "/" + str(len(test)) + " (" + str(curr_predicted/len(test)) + ")")
        accuracy += curr_predicted/len(test)

    return accuracy / 5

def naive_train_test(dataset):
    tree = build_tree(dataset)
    predicted = 0
    for x in dataset:
        k = classify(x, tree)
        if k == x[-1]:
            predicted = predicted + 1
    return predicted / len(dataset)