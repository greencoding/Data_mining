#-*- coding:utf-8 -*-
from sklearn import tree
from sklearn.model_selection import KFold
from numpy import *
from sklearn import metrics


Sampleset = []
FeatureSet = []
Label = []
my_file = open('Iris.data', 'r')
for string in my_file.readlines():
	i = 0
	Sampleset = []
	string_list = string.split(',')
	Label.append(string_list[-1])
	while i < len(string_list) - 1 :
		Sampleset.append(string_list[i])
		i = i + 1
	FeatureSet.append(Sampleset)


FeatureSet = array(FeatureSet)
Label = array(Label)
kf = KFold(n_splits=10)
 
i=0
for train_index, test_index in kf.split(FeatureSet, Label):
    i = i + 1
    X_train, X_test = FeatureSet[train_index], FeatureSet[test_index]
    y_train, y_test = Label[train_index], Label[test_index]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    pre_labels = clf.predict(X_test)
    # Modeal Evaluation
    ACC = metrics.accuracy_score(y_test, pre_labels)
    print str(i) + ':  ACC   ' + str(ACC)


