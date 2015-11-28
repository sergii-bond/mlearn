#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit

# Classifier evaluation function using cross-validation
def test_classifier(clf, dataset, feature_list, folds = 1000, scale = False):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    if scale == True:
        # normalize data if required by an algorithm
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        features = min_max_scaler.fit_transform(features)

    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on validation set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return (precision, recall, f1)
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
        return (0, 0, 0)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

print("Number of points in the dataset: " + str(len(data_dict)))

num_poi = 0

# find the available feature names
feature_names = data_dict[data_dict.keys()[0]].keys()
num_features = len(feature_names) 

# how many POIs ?
for k, v in data_dict.iteritems():
    poi = v["poi"]
    num_poi += int(poi)

print("Number of points with 'poi = True': " + str(num_poi))
print("Number of original features: " + str(num_features))
print("List of original features: " + str(feature_names))



# See how many NaNs for each feature
for f in feature_names:
    print("Summary for feature " + str(f) + ":")
    num_nans = 0
    for k, v in data_dict.iteritems():
        if v[f] == 'NaN':
            num_nans += 1

    print("\tPercent of NaNs: " + str(round(num_nans / float(len(data_dict)) * 100.0, 1)) + "%")

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Remove the outliers
my_dataset.pop("TOTAL", 0)
my_dataset.pop("TRAVEL AGENCY IN THE PARK", 0)

# create two new features
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

# regenerate the list of features after adding new features
feature_names = my_dataset[my_dataset.keys()[0]].keys()
print("Modified list of features: " + str(feature_names))

# 'poi' is a label, and 'email_address' is not used
feature_names.remove('poi')
feature_names.remove('email_address')

# this is a list of features that procedure featureFormat requires ('poi' must
# be included and go first)
features_list = ['poi'] + feature_names
feature_names = np.array(feature_names) 

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, remove_NaN = True, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Select percentile features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

kbest = SelectKBest(f_classif, k = 4)
kbest.fit(features, labels)
print("SelectKBest scores: " + str(sorted(zip(feature_names, [round(x, 2) for x in kbest.scores_]), key = lambda x : x[1])))
  

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import tree
from sklearn.svm import SVC

#fl = ['poi'] + list(feature_names[kbest.get_support()]) 

#for c in [0.1, 0.5, 1, 5, 10, 15, 20, 30, 40, 50, 100]:
#    for g in [0.1, 0.5, 1, 5, 10, 15, 20, 30, 40, 50, 100]:
#for c in [50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
#    for g in [100]:
#        clf = SVC(C = c, gamma = g)
#        (precision, recall, f1) = test_classifier(clf, my_dataset, fl, folds = 1000, scale = True)
#        print("SVM(" + str(c) + " " + str(g) + "): F1 score is " + str(f1) + " with precision " + str(precision) + " and recall " + str(recall))

#for m in [1, 5, 10, 15, 20]:
#    clf = tree.DecisionTreeClassifier(min_samples_split = m)
#    (precision, recall, f1) = test_classifier(clf, my_dataset, fl, folds = 1000)
#    print("Decision Tree" + str(m) + "): F1 score is " + str(f1) + " with precision " + str(precision) + " and recall " + str(recall))
#
#print("With new features: ")
#fl = ['poi'] + list(feature_names[kbest.get_support()]) + ['from_poi_rel', 'to_poi_rel']
#
#for m in [1, 5, 10, 15, 20]:
#    clf = tree.DecisionTreeClassifier(min_samples_split = m)
#    (precision, recall, f1) = test_classifier(clf, my_dataset, fl, folds = 1000)
#    print("Decision Tree" + str(m) + "): F1 score is " + str(f1) + " with precision " + str(precision) + " and recall " + str(recall))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#tune = False
tune = True

if tune == True:

    sel_p1 = 0 # parameter for SelectPercentile
    sel_p2 = 0 # parameter for min_samples_split
    prec = 0   # precision
    rec = 0    # recall
    max_f1 = 0 # f1 score
    clf = None # classifier

    # tune the parameters
    for p1 in np.arange(1, 21, 1):
        kbest = SelectKBest(f_classif, k = p1)
        kbest.fit(features, labels)
        fl = ['poi'] + list(feature_names[kbest.get_support()])

        for p2 in np.arange(1, 15, 1):
            sys.stdout.write(".")
            sys.stdout.flush()
            cur_clf = tree.DecisionTreeClassifier(min_samples_split = p2)
            (precision, recall, f1) = test_classifier(cur_clf, my_dataset, fl, folds = 1000)

            # here the best parameters/scores are remembered
            if f1 > max_f1:
                max_f1 = f1
                sel_p1 = p1
                sel_p2 = p2
                features_list = fl
                prec = precision
                rec = recall
                clf = cur_clf


    print('\n')
    print("Best number for features: " + str(sel_p1))
    print("Best features: " + str(features_list))
    print("Best min_samples_split: " + str(sel_p2))
    print("Best F1 score is " + str(max_f1) + " with precision " + str(prec) + " and recall " + str(rec))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
