#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print("Accuracy is: ", clf.score(features_test, labels_test))

pred = clf.predict(features_test)
print("Predicted " + str(sum(pred)) + " POIs")
print("Number of people in the test set: " + str(len(features_test)))
print("Number of truly POI people in the test set: " + str(sum(labels_test)))
print("Baseline accuracy: " + str(1.0  - sum(labels_test) / len(labels_test)))
print("Number of true positives: " + str(sum(labels_test * pred)))

from sklearn import metrics

precision = metrics.precision_score(labels_test, pred)
recall = metrics.recall_score(labels_test, pred)
f1 = metrics.f1_score(labels_test, pred)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

