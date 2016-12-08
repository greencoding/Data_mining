#-*- coding:utf-8 -*-
from sklearn import tree
from sklearn.cross_validation import train_test_split
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



X_train, X_test, y_train, y_test = train_test_split(
    FeatureSet, Label, random_state=1)  # 将数据随机分成训练集和测试集


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pre_labels = clf.predict(X_test)
# Modeal Evaluation
ACC = metrics.accuracy_score(y_test, pre_labels)
print 'ACC   ' + str(ACC)
print 'The graph has been created!'
tree.export_graphviz(clf,out_file='tree.dot')  

