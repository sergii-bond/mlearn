#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit

#PERF_FORMAT_STRING = "\
#\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
#Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
#RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

# Classifier evaluation function
def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
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
        #print clf
        #print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        #print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        #print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
    
    return (precision, recall, f1)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

print("Number of points in the dataset: " + str(len(data_dict)))

num_poi = 0

feature_names = data_dict[data_dict.keys()[0]].keys()
num_features = len(feature_names) 

for k, v in data_dict.iteritems():
    poi = v["poi"]
    num_poi += int(poi)

    #if len(v) > num_features:
    #    num_features = len(v)
    #    feature_names = v.keys()

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
#feature_names.remove('from_poi_rel')
#feature_names.remove('to_poi_rel')

features_list = ['poi'] + feature_names
#features_list = ['poi', 'salary']

### Extract features and labels from dataset for local testing

# Before looking at data, separate training set, validation set and test set
# Test set will be only used to report the final evaluation metrics
#data = featureFormat(my_dataset, features_list, remove_NaN = False, sort_keys = True)
data = featureFormat(my_dataset, features_list, remove_NaN = True, sort_keys = True)
labels, features = targetFeatureSplit(data)

# normalize data
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#features = min_max_scaler.fit_transform(features)

# Deal with NaNs
#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#imp.fit(features)
#features = imp.transform(features)

# Select percentile features
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

  

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#from sklearn import svm
#c = 10.0
#g = 10
#clf = svm.SVC(C = c, kernel = 'rbf', gamma = g)

from sklearn import tree

sel_p1 = 0
sel_p2 = 0
prec = 0
rec = 0
max_f1 = 0
feature_names = np.array(feature_names)
clf = None

for p1 in np.arange(5, 50, 5):
    kbest = SelectPercentile(f_classif, percentile = p1)
    kbest.fit(features, labels)
    fl = ['poi'] + list(feature_names[kbest.get_support()])

    for p2 in np.arange(1, 15, 1):
        sys.stdout.write(".")
        sys.stdout.flush()
        cur_clf = tree.DecisionTreeClassifier(min_samples_split = p2)
        (precision, recall, f1) = test_classifier(cur_clf, my_dataset, fl, folds = 1000)

        if f1 > max_f1:
            max_f1 = f1
            sel_p1 = p1
            sel_p2 = p2
            features_list = fl
            prec = precision
            rec = recall
            clf = cur_clf


print('\n')
print("Best percentile for features: " + str(sel_p1))
print("Best features: " + str(features_list))
print("Best min_samples_split: " + str(sel_p2))
print("Best F1 score is " + str(max_f1) + " with precision " + str(prec) + " and recall " + str(rec))

#print(sorted(zip(clf.feature_importances_, feature_names), key = lambda x : x[0]))
#from sklearn import metrics
#
#print("\tAccuracy on a training set: " + str(metrics.accuracy_score(labels_train, pred_train)))
#print("\tPrecision on a training set: " + str(metrics.precision_score(labels_train, pred_train)))
#print("\tRecall on a training set: " + str(metrics.recall_score(labels_train, pred_train)))
#
#print("\tAccuracy on a validation set: " + str(metrics.accuracy_score(labels_valid, pred_valid)))
#print("\tPrecision on a validation set: " + str(metrics.precision_score(labels_valid, pred_valid)))
#print("\tRecall on a validation set: " + str(metrics.recall_score(labels_valid, pred_valid)))

#import matplotlib.pyplot as plt
##plt.scatter(features_train[:, feature_names.index('salary')], color=test_color ) 
##f1_name = 'salary'
#
#f1_name = 'from_poi_rel'
##f2_name = 'bonus'
#f2_name = 'to_poi_rel'
#id1 = feature_names.index(f1_name)
#id2 = feature_names.index(f2_name)
#colors = ["b", "r"]
#for ii, pp in enumerate(labels_train):
#    plt.scatter(features_train[ii,id1], features_train[ii,id2], color = colors[int(pp)])
#
#for ii, pp in enumerate(labels_valid):
#    plt.scatter(features_valid[ii,id1], features_valid[ii,id2], color =
#            colors[int(pp)], marker = '*', s = 40)
#
#plt.xlabel(f1_name)
#plt.ylabel(f2_name)
#plt.show() 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Create a set of validation and training sets
# The separation is done by labels to avoid skewness in resulting sets

#cv = StratifiedShuffleSplit(labels, n_iter = 100, test_size = 0.1, random_state = 42)

#print("Baseline accuracy: " + str(1.0 - sum(labels_valid) / float(len(labels_valid))))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
