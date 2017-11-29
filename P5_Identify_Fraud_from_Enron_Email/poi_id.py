#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
#DataSet and features  
def data_info (dic_data):
	pois = 0
	nonpoi = 0
	for v in dic_data.values():
		if v['poi']:
    			pois+= 1
    		else:
    			nonpoi+=1
#features with many missing values
	features_keys = data_dict['LAY KENNETH L'].keys() 
	missing_dict = dict()
	for feature in features_keys:
		missing_dict[feature] = 0
	for person in data_dict.keys():
		for feature in features_keys:
			if data_dict[person][feature] == 'NaN':
				missing_dict[feature] += 1

	for feature in features_keys:  
		print feature 
		print missing_dict[feature]
#====================================================================
#length of the dataset (how many people)
	print "Total Number of People in the Dataset"
	print len(dic_data)
#how many features
	print "Total Number of features "
	print len(dic_data["LAY KENNETH L"]) 
#how many poi
	print "Number of pois "
	print pois
#how many non-poi
	print "Number of non pois "
	print nonpoi
	print "CEO Total Payments"
	print dic_data['SKILLING JEFFREY K']['shared_receipt_with_poi']#CEO
	print "Chairman Total Payments"
	print dic_data['LAY KENNETH L']['total_payments'] #chairman
	print "CFO Total Payments"
	print dic_data['FASTOW ANDREW S']['total_payments']#CFO
#=====================================================================

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

all_features_list = [
	'poi',
    'salary',
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'total_payments',
    'total_stock_value',
    'from_messages',
    'to_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'fraction_from_poi',
    'fraction_to_poi'
]


features_list =['poi',
'salary', 
'total_payments',
'loan_advances',
'bonus',
'total_stock_value', 
'shared_receipt_with_poi', 
'exercised_stock_options',
'deferred_income',
'restricted_stock',
'long_term_incentive',
'fraction_to_poi'
]


features_list_without_created_feature =['poi',
'salary',
'total_payments',
'loan_advances',
'bonus',
'total_stock_value',
'shared_receipt_with_poi',
'exercised_stock_options',
'deferred_income',
'restricted_stock',
'long_term_incentive']

print features_list
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#=====================================================================
### Task 2: Remove outliers
# the function below is for the seek of visualizing outliers
def Plot(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
    	matplotlib.pyplot.scatter(x, y, color=color)
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()

#view it before removing
Plot(data_dict, 'salary', 'bonus') #we can see there's high point 
#remove outlier No#1 phenomenon of spreadsheet that calculate the total of each columns 
data_dict.pop('TOTAL',0) 
# Remove outlier No#2 person with 'NaN' values for all the feature.
data_dict.pop('LOCKHART EUGENE E')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

#=====================================================================
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#The two new features that calculate the fraction of email messages from/to person to/from POI     

my_dataset = data_dict

for person in my_dataset:
    data_point = my_dataset[person]
    data_point["fraction_from_poi"] = 0
    data_point["fraction_to_poi"] = 0
    if data_point["from_poi_to_this_person"] =='NaN' or data_point["to_messages"] =='NaN':
    	data_point["fraction_from_poi"] = 0
    else:
    	fraction_from_poi = float(data_point["from_poi_to_this_person"]) / float(data_point["to_messages"])
    	data_point["fraction_from_poi"] = fraction_from_poi

	if data_point["from_this_person_to_poi"] =='NaN' or data_point["from_messages"] =='NaN':
		data_point["fraction_to_poi"] = 0
	else:
		fraction_to_poi = float(data_point["from_this_person_to_poi"]) / float(data_point["from_messages"])
    	data_point["fraction_to_poi"] = fraction_to_poi

#=====================================================================

### Extract features and labels from dataset for local testing
#(featureFormat&targetFeatureSplit) take a list of feature names and the data dictionary, and return a numpy array. 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#features selection & scoring 
def best_features(num,featureList,features,labels):
	k_best = SelectKBest(k=num)
	k_best.fit(features, labels)
	scores = k_best.scores_
	print scores

	unsorted_pairs = zip(featureList[1:], scores)
	sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
	k_best_features = dict(sorted_pairs[:num])
	print featureList
	print "{0} best features: {1}\n".format(num, k_best_features.keys())
#========================================================================


#scalling all the features 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

#====================================================================
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)

# Support Vector Machine Classifier
from sklearn.svm import SVC
from sklearn import svm, grid_search, datasets
#note : remove auto from below class weight from some error 
svm_parameters = {'kernel':['rbf'], 'gamma': [0.0001],'C':[1,10,100,1000] ,'class_weight': ['auto'],'random_state': [42]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, svm_parameters) 

# Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
randomForest_parameters = {'criterion':['gini', 'entropy'], 'class_weight': ['balanced','auto'],'random_state': [42]}
rf = RandomForestClassifier()
rf_clf = grid_search.GridSearchCV(rf, randomForest_parameters) 

#DecisionTreeClassifier
from sklearn import tree
dt_parameters = { 'min_samples_split':[10,20,30,40], 'criterion' : ['gini', 'entropy'] ,'splitter': ['best', 'random'],'class_weight': ['balanced' ,'auto'],'random_state': [13, 20, 42]}
dt = tree.DecisionTreeClassifier()
dt_clf=grid_search.GridSearchCV(dt, dt_parameters)

#two validation is tried 
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
#peace of code from the tester.py to try StratifiedShuffleSplit validation 
from sklearn.cross_validation import StratifiedShuffleSplit

def get_test_data(features, labels, feature_list, folds = 10):
	cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
	features_train = []
	features_test  = []
	labels_train   = []
	labels_test    = []
	for train_idx, test_idx in cv: 
		for ii in train_idx:
			features_train.append( features[ii] )
			labels_train.append( labels[ii] )
		for jj in test_idx:
			features_test.append( features[jj] )
			labels_test.append( labels[jj] )    
	return features_train, features_test, labels_train, labels_test

features_train_s, features_test_s, labels_train_s, labels_test_s = get_test_data(features, labels, features_list)


#Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score
print clf

clf.fit(features_train,labels_train)
print "best paramters "
print clf.best_params_
clf = clf.best_estimator_
predictions = clf.predict(features_test)
print "validation#1 prediction is "
print predictions
print "validation#1 precision is "
print precision_score(labels_test, predictions)

print "validation#1 recalll is"
print recall_score(labels_test, predictions)
print "validation#1 Accuracy is "
print accuracy_score(labels_test, predictions)

clf.fit(features_train_s,labels_train_s)
predictions = clf.predict(features_test_s)
print "validation#2 prediction is "
print predictions
print "validation#2 precision is "
print precision_score(labels_test_s, predictions)

print "validation#2 recalll is"
print recall_score(labels_test_s, predictions)
print "validation#2 Accuracy is "
print accuracy_score(labels_test_s, predictions)


#dump and fininal test 
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from tester import dump_classifier_and_data, test_classifier
test_classifier(clf, data_dict, features_list)
dump_classifier_and_data(clf, my_dataset, features_list)

#========================================================
#MAIN class 
def main():
	data_info (data_dict)
	best_features(11,all_features_list,features,labels)
if __name__ == '__main__':
	main()

