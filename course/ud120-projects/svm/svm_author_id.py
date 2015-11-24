#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

# idea: train on a small dataset first (1%), tune parameters, then train on a
# full dataset

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
# the following command showed that dim(x) = 3785
#print(len(features_train[1]))
# the following command showed that dim(y) = 1
#print(labels_train[1])

from sklearn.svm import SVC

c = 1e4
#for c in [0.1, 0.5, 1, 10, 1000, 10000, 1e6, 1e7]:
#    print("C = ", c)
clf = SVC(kernel="rbf", C = c)
clf.fit(features_train, labels_train)
print("Accuracy is: ", clf.score(features_test, labels_test))

#print(clf.predict([features_test[10], features_test[26], features_test[50]]))
print(sum(clf.predict(features_test)))
