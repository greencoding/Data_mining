from sklearn import cross_validation
from sklearn import tree
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


rs = cross_validation.ShuffleSplit(len(FeatureSet), 3, 0.25, random_state=0)
clf = tree.DecisionTreeClassifier()
for train_index, test_index in rs:
	X_train = []
	X_test = []
	y_train = []
	y_test = []
	for trainid in train_index.tolist():
		X_train.append(FeatureSet[trainid])
                y_train.append(Label[trainid])
	for testid in test_index.tolist():
                X_test.append(FeatureSet[testid])
                y_test.append(Label[testid])

	clf = clf.fit(X_train, y_train)
	pre_labels = clf.predict(X_test)
	# Modeal Evaluation
	ACC = metrics.accuracy_score(y_test, pre_labels)
	print 'ACC   ' + str(ACC)
