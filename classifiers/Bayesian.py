import numpy as np
from dataset import datatools
from sklearn.model_selection import KFold

def mean(sample):
    '''
    Compute sample mean
    '''
    M = []
    for x in range(sample.shape[1]):
        M.append(sample[:,x].mean())
    return np.array(M)

def covariance(sample, sample_mean):
    '''
    Compute sample covariance
    '''
    sigma = np.zeros((len(sample[0]), len(sample[0])))
    for x in sample:
        sigma += np.outer(x - sample_mean, x - sample_mean)
    return sigma / (len(sample) - 1)

def GaussianNB(x, M_w1, M_w2, sigma_w1, sigma_w2, p_w1, p_w2):
    '''
    Gaussian Naive Bayes Classifier
    '''
    # diagonalize covariance matricies
    sigma_w1 = np.diag(np.diag(sigma_w1))
    sigma_w2 = np.diag(np.diag(sigma_w2))

    # sigma det
    det_sigma_w1 = np.linalg.det(sigma_w1)
    det_sigma_w2 = np.linalg.det(sigma_w2)

    # mahalanobis distance w1, w2
    d_w1 = np.matmul((x - M_w1).T, np.matmul(np.linalg.inv(sigma_w1), (x - M_w1)))
    d_w2 = np.matmul((x - M_w2).T, np.matmul(np.linalg.inv(sigma_w2), (x - M_w2)))

    # discriminant
    return np.log(det_sigma_w2) - np.log(det_sigma_w1) + (d_w2 - d_w1)  + (np.log(p_w1) - np.log(p_w2))

def GaussianOB(x, M_w1, M_w2, sigma_w1, sigma_w2, p_w1, p_w2):
    '''
    Gaussian Optimal Bayes Classifier
    '''
    det_sigma_w1 = np.linalg.det(sigma_w1)
    det_sigma_w2 = np.linalg.det(sigma_w2)

    # mahalanobis distance w1, w2
    d_w1 = np.matmul((x - M_w1).T, np.matmul(np.linalg.inv(sigma_w1), (x - M_w1)))
    d_w2 = np.matmul((x - M_w2).T, np.matmul(np.linalg.inv(sigma_w2), (x - M_w2)))

    # discriminant
    return np.log(det_sigma_w2) - np.log(det_sigma_w1) + (d_w2 - d_w1) + (np.log(p_w1) - np.log(p_w2))

def train_test(dataset, classifier, folds=5):
    '''
    Train and test using k-fold cross validation
    Returns accuracy
    '''
    # 5-fold cross-validation
    kfold = KFold(folds, True, 1)
    accuracy = 0
    fold_count = 0

    if classifier == "GaussianNB":
        print("(Naive Bayes)")
    else:
        print("(Optimal Bayes)")
    for train, test in kfold.split(dataset):
        curr_predicted = 0
        fold_count += 1
        train_w1, train_w2 = datatools.split_classes(dataset[train])

        # apriori
        p_w1 = np.count_nonzero(train_w1[:,-1] == 1) / (len(train_w1) + len(train_w2))
        p_w2 = 1 - p_w1

        # unlabel training sets (to simplify the calculation for computing the gaussian params)
        train_w1 = np.delete(train_w1, -1, 1)
        train_w2 = np.delete(train_w2, -1, 1)

        # gaussian parameters
        mean_w1 = mean(train_w1)
        mean_w2 = mean(train_w2)
        cov_w1 = covariance(train_w1, mean_w1)
        cov_w2 = covariance(train_w2, mean_w2)

        # test
        for x in dataset[test]:
            labelled = np.copy(x)
            unlabelled = np.delete(x, -1)
            if classifier == "GaussianNB":
                k = GaussianNB(unlabelled, mean_w1, mean_w2, cov_w1, cov_w2, p_w1, p_w2)
            elif classifier == "GaussianOB":
                k = GaussianOB(unlabelled, mean_w1, mean_w2, cov_w1, cov_w2, p_w1, p_w2)
            if k > 0 and labelled[-1] == 1.0:
                curr_predicted += 1
            elif k < 0 and labelled[-1] == 0.0:
                curr_predicted += 1
            
        print("\t\tFold #" + str(fold_count) + ": " + str(curr_predicted) + "/" + str(len(test)) + " (" + str(curr_predicted/len(test)) + ")")
        accuracy += curr_predicted/len(test)

    return accuracy / 5