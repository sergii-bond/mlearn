#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print("Number of data points (people): " + str(len(enron_data)))

num_features = 0
num_poi = 0
max_total_payments = 0
total_folks_with_email_addr = 0
total_folks_with_salary = 0
num_folks_with_NaN_total_payments = 0
num_poi_with_NaN_total_payments = 0

for k,v in enron_data.iteritems():
    if k == "TOTAL":
        continue
    l = len(v)
    if l > num_features:
        num_features = l

    if v["poi"] == True:
        num_poi += 1

    total_payments = v["total_payments"]
    if total_payments == 'NaN':
        num_folks_with_NaN_total_payments += 1
        if v["poi"] == True:
            num_poi_with_NaN_total_payments += 1

    elif total_payments > max_total_payments:
    #if math.isnan(float(total_payments)) == False
        max_total_payments = total_payments
        person_with_max_total_payments = k

    if v["email_address"] != 'NaN':
        total_folks_with_email_addr += 1 

    if v["salary"] != 'NaN':
        total_folks_with_salary += 1 




print("Features available per person: " + str(num_features))
print("Number of people of interest: " + str(num_poi))

num_poi_in_txt_file = 0
import re
with open("../final_project/poi_names.txt", 'r') as f:
    for line in f:
        if re.match("\(y\)|\(n\)", line):
            num_poi_in_txt_file += 1

print("Total POIs in the text file: " + str(num_poi_in_txt_file))

print("Total value of the stock belonging to James Prentice: " +
        str(enron_data["PRENTICE JAMES"]["total_stock_value"]))

print("How many email messages do we have from Wesley Colwell to persons of interest? - " +
        str(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]))

#for k in enron_data.keys():
#    if re.match('^S', k):
#        print k

print("What's the value of stock options exercised by Jeffrey Skilling? - " + 
        str(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]))

print(person_with_max_total_payments + " took " + str(max_total_payments) + 
    " from the company, the largest amount")

print("Number of folks with email address: " + str(total_folks_with_email_addr))
print("Number of folks with salary: " + str(total_folks_with_salary))
#print(enron_data["TOTAL"]["salary"])

#print("Number of folks with NaN total payments: " + str(num_folks_with_NaN_total_payments))
print("Percent of folks with NaN total payments: " +
        str(float(num_folks_with_NaN_total_payments) / len(enron_data) * 100))
print("Percent of POIs with NaN total payments: " +
        str(float(num_poi_with_NaN_total_payments) / num_poi * 100))

print("Number of people + 10 = " + str(len(enron_data) + 10))
print("Number of folks with NaN total payments + 10: " +
        str(num_folks_with_NaN_total_payments + 10))
