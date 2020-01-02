# Glass dataset classification
Altin Rexhepaj, 2020

## About
This program is an exercise in multivariate dataset classification using machine learning techniques. In particular, I implement Naive Bayes, Optimal Bayes, and Decision Tree classifiers<sup>[[1]](http://www.uoitc.edu.iq/images/documents/informatics-institute/Competitive_exam/Artificial_Intelligence.pdf)</sup> to classify between two types of glass: windowed, and non-windowed (binary class classification). Testing and training is done with k-fold cross validation.

## Dataset
The dataset I used for this exercise comes from the UCI Machine Learning Repository<sup>[[2]](https://archive.ics.uci.edu/ml/datasets/glass+identification)</sup>.  

#### Number of Samples
214

#### Number of Attributes
10 (including an Id#) plus the class attribute

#### Attribute Information
 1. Id number: 1 to 214
 2. RI: refractive index
 3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as 
                are attributes 4-10)
 4. Mg: Magnesium
 5. Al: Aluminum
 6. Si: Silicon
 7. K: Potassium
 8. Ca: Calcium
 9. Ba: Barium
10. Fe: Iron
11. Type of glass: (class attribute)
    * 1 building_windows_float_processed
    * 2 building_windows_non_float_processed
    * 3 vehicle_windows_float_processed
    * 4 vehicle_windows_non_float_processed (none in this database)
    * 5 containers
    * 6 tableware
    * 7 headlamps

Windowed glass is given by class attributes 1 to 4, and non-windowed glass is given by class-attributes 5 to 7.  

## Classifiers
### Decision Tree
#### Information gain
The information gain is calculated using the usual entropy measure for decision trees.

#### Discretisation function
The dataset attributes are discretised according to the mean of the attributes across all samples. If the attribute is greater than the mean, it will go to the right subtree, if the attribute is less than the mean, it will go to the left subtree. Windowed glass is given by the value `1.0` whereas non-windowed glass is given by the value `0.0`.

### Bayesian
I assume a gaussian distribution for both Bayesian classifiers. Naive Bayes assumes independent samples, thus I diagonalize the covariance matrix, whereas optimal Bayes assumes dependent samples, thus I do not diagonalize the covariance matrix.

## Accuracy
* *Based on 5-fold cross validation on 214 samples*
### Decision Tree
#####
78.4%
* *Future work: The accuracy can be greatly improved by using a better discretisation function, and overfitting can be limited by tree pruning.*

### Naive Bayes
90.6%

### Optimal Bayes 
93.0%

## How to run
1. `pip install -r requirements.txt`
2. `python classify.py`
3. Check console output

## Screenshots
### Bayesian Classifiers
![image](https://github.com/altin/glass-classifier-ml/blob/master/bayesexample.PNG)

### Decision Tree Classifier
![image](https://github.com/altin/glass-classifier-ml/blob/master/dtexample.PNG)
* The numbers (nodes) represent the position (starting from zero) of the attribute at that level of the decision tree. For example `2` is `magnesium` on level `0`, whereas `2` is `aluminum` on level `1` since `magnesium` is removed from the attribute list. The `D:` represents the decision to be made by the tree, where `1.0` is windowed glass, and `0.0` is non-windowed glass.  
* The accuracy per fold is given by the correctly labelled samples versus the total unlabelled in the fold.


