import numpy as np
from dataset import datatools
from classifiers import *

glass_data = datatools.clean_dataset(datatools.glass_data)
glass_binary = datatools.binarize(glass_data)

test = np.array([64,580,29,66,570,33,68,590,37,69,660,46,73,600,55]).reshape(5,3)

print("(Glass Data Results)")
# Decision Tree
print("(Decision Tree)")
accuracy = DecisionTree.train_test(glass_binary, True)
print("\tAvg. Decision Tree Accuracy (5-fold): " + str(accuracy))
print()

# Gaussian Naive Bayes
accuracy = Bayesian.train_test(glass_data, "GaussianNB")
print("\tAvg. Naive Bayes Accuracy (5-fold): " + str(accuracy))
print()

# Gaussian Optimal Bayes
accuracy = Bayesian.train_test(glass_data, "GaussianOB")
print("\tAvg. Optimal Bayes Accuracy (5-fold): " + str(accuracy))
print()