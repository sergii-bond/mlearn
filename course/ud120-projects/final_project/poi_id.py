#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

print("Number of points in the dataset: " + str(len(data_dict)))

num_poi = 0
num_features = 0

for k, v in data_dict.iteritems():
    poi = v["poi"]

    num_poi += int(poi)

    if len(v) > num_features:
        num_features = len(v)
        feature_names = v.keys()

print("Number of points with 'poi = True': " + str(num_poi))

print("Number of original features: " + str(num_features))
print("List of original features: " + str(feature_names))



#for f in features:
#    print("Summary for feature " + str(f) + ":")
#    num_nans = 0
#    for k, v in data_dict.iteritems():
#        if v[f] == 'NaN':
#            num_nans += 1
#
#    print("\tPercent of NaNs: " + str(round(num_nans / float(len(data_dict)) * 100.0, 1)) + "%")

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset.pop("TOTAL", 0)

for k, v in my_dataset.iteritems():
    from_poi = v['from_poi_to_this_person']
    to_poi = v['from_this_person_to_poi']
    sent = v['to_messages']
    received = v['from_messages']

    if from_poi != 'NaN' and received != 'NaN':
        v['from_poi_rel'] = from_poi / float(received) ;
    else:
        v['from_poi_rel'] = 'NaN'

    if to_poi != 'NaN' and sent != 'NaN':
        v['to_poi_rel'] = to_poi / float(sent) ;
    else:
        v['to_poi_rel'] = 'NaN'

feature_names = my_dataset[my_dataset.keys()[0]].keys()
print("Modified list of features: " + str(feature_names))

feature_names.remove('poi')
feature_names.remove('email_address')
#feature_names.remove('salary')
#feature_names.remove('to_messages')
#feature_names.remove('deferral_payments')
#feature_names.remove('total_payments')
#feature_names.remove('exercised_stock_options')
#feature_names.remove('bonus')
#feature_names.remove('restricted_stock')
#feature_names.remove('shared_receipt_with_poi')
#feature_names.remove('restricted_stock_deferred')
#feature_names.remove('total_stock_value')
#feature_names.remove('expenses')
#feature_names.remove('loan_advances')
#feature_names.remove('from_messages')
#feature_names.remove('other')
#feature_names.remove('from_this_person_to_poi')
#feature_names.remove('director_fees')
#feature_names.remove('deferred_income')
#feature_names.remove('long_term_incentive')
#feature_names.remove('from_poi_to_this_person')

features_list = ['poi'] + feature_names
#features_list = ['poi', 'salary']

### Extract features and labels from dataset for local testing

# Before looking at data, separate training set, validation set and test set
# Test set will be only used to report the final evaluation metrics
#data = featureFormat(my_dataset, features_list, remove_NaN = False, sort_keys = True)
data = featureFormat(my_dataset, features_list, remove_NaN = True, sort_keys = True)
labels, features = targetFeatureSplit(data)

# The separation is done by labels to avoid skewness in resulting sets
data_with_label0 = data[labels == 0,:]
data_with_label1 = data[labels == 1,:]

# separate test set from cross-validation set (training + validation)
from sklearn.cross_validation import train_test_split

data_with_label1_cv, data_with_label1_test = train_test_split(data_with_label1, test_size=0.2)
data_with_label0_cv, data_with_label0_test = train_test_split(data_with_label0, test_size=0.2)
#print(len(data_with_label1), len(data_with_label1_cv), len(data_with_label1_test))
#print(len(data_with_label0), len(data_with_label0_cv), len(data_with_label0_test))
data_cv = np.vstack((data_with_label1_cv, data_with_label0_cv))
labels_cv, features_cv = targetFeatureSplit(data_cv)
print("Number of points in cross-validation set: " + str(len(labels_cv)))
print("Number of POIs in cross-validation set: " + str(sum(labels_cv)))

data_test = np.vstack((data_with_label1_test, data_with_label0_test))
labels_test, features_test = targetFeatureSplit(data_test)
print("Number of points in test set: " + str(len(labels_test)))
print("Number of POIs in test set: " + str(sum(labels_test)))

# separate validation set from the training set
# Validation set will be used only in the beginning, for feature selection and
# to get a sense of what algorithm is better suited for the application 
# later only the cross-validation set will be used to select final parameters of learning algorithms
data_with_label1_train, data_with_label1_valid = train_test_split(data_with_label1_cv, test_size=0.2)
data_with_label0_train, data_with_label0_valid = train_test_split(data_with_label0_cv, test_size=0.2)

data_train = np.vstack((data_with_label1_train, data_with_label0_train))
labels_train, features_train = targetFeatureSplit(data_train)
print("Number of points in training set: " + str(len(labels_train)))
print("Number of POIs in training set: " + str(sum(labels_train)))

data_valid = np.vstack((data_with_label1_valid, data_with_label0_valid))
labels_valid, features_valid = targetFeatureSplit(data_valid)
print("Number of points in validation set: " + str(len(labels_valid)))
print("Number of POIs in validation set: " + str(sum(labels_valid)))

print("Baseline accuracy: " + str(1.0 - sum(labels_valid) / float(len(labels_valid))))

#print features_valid
# Deal with NaNs
#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#imp.fit(features_train)
#features_train = imp.transform(features_train)
#features_valid = imp.transform(features_valid)
#features_test = imp.transform(features_test)

# normalize data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_train = min_max_scaler.fit_transform(features_train)
features_valid = min_max_scaler.transform(features_valid)
features_test = min_max_scaler.transform(features_test)

# Select K best features
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#
#kbest = SelectKBest(chi2, k = 7)
#kbest.fit(features_train, labels_train)
#features_train = kbest.transform(features_train)
#features_valid = kbest.transform(features_valid)
#features_test = kbest.transform(features_test)
  

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

from sklearn import svm
c = 10.0
g = 10
clf = svm.SVC(C = c, kernel = 'rbf', gamma = g)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier(min_samples_split=10)
#clf = tree.DecisionTreeClassifier()

clf.fit(features_train, labels_train)
pred_train = clf.predict(features_train)
pred_valid = clf.predict(features_valid)
#print(pred_train)
#print(pred_valid)

#print(sorted(zip(clf.feature_importances_, feature_names), key = lambda x : x[0]))
from sklearn import metrics

print("\tAccuracy on a training set: " + str(metrics.accuracy_score(labels_train, pred_train)))
print("\tPrecision on a training set: " + str(metrics.precision_score(labels_train, pred_train)))
print("\tRecall on a training set: " + str(metrics.recall_score(labels_train, pred_train)))

print("\tAccuracy on a validation set: " + str(metrics.accuracy_score(labels_valid, pred_valid)))
print("\tPrecision on a validation set: " + str(metrics.precision_score(labels_valid, pred_valid)))
print("\tRecall on a validation set: " + str(metrics.recall_score(labels_valid, pred_valid)))

import matplotlib.pyplot as plt
#plt.scatter(features_train[:, feature_names.index('salary')], color=test_color ) 
#f1_name = 'salary'

f1_name = 'from_poi_rel'
#f2_name = 'bonus'
f2_name = 'to_poi_rel'
id1 = feature_names.index(f1_name)
id2 = feature_names.index(f2_name)
colors = ["b", "r"]
for ii, pp in enumerate(labels_train):
    plt.scatter(features_train[ii,id1], features_train[ii,id2], color = colors[int(pp)])

for ii, pp in enumerate(labels_valid):
    plt.scatter(features_valid[ii,id1], features_valid[ii,id2], color =
            colors[int(pp)], marker = '*', s = 40)

plt.xlabel(f1_name)
plt.ylabel(f2_name)
plt.show() 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
