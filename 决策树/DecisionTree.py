#-*- coding:utf-8 -*-

# 定义节点的属性

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import cross_validation


class decisionnode:

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col   # col是待检验的判断条件所对应的列索引值
        self.value = value  # value对应于为了使结果为True，当前列必须匹配的值
        self.results = results  # 保存的是针对当前分支的结果，它是一个字典
        self.tb = tb  # desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb  # desision node,对应于结果为true时，树上相对于当前节点的子树上的节点

tree = ''


class DecisionTree():

    def __init__(self):
        self.FeatureSet = []
        self.Label = []
        self.threshold = 0.0
	self.testset = []
	self.property = []

    def load_data(self, fileName):
        my_file = open(fileName, 'r')
        for string in my_file.readlines():
            i = 0
            Sampleset = []
            new_string = string.strip()
            string_list = new_string.split(',')
            self.Label.append(string_list[-1])
            while i < len(string_list) - 1 :
                Sampleset.append(string_list[i])
                i = i + 1
	    self.testset.append(Sampleset)
	    i = 0
            Sampleset = []
            while i < len(string_list) :
                Sampleset.append(string_list[i])
                i = i + 1
            self.FeatureSet.append(Sampleset)

    # count the number
    def uniquecounts(self, rows):
        results = {}
        for row in rows:
            # 计数结果在最后一列
            r = row[len(row) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results  # return a dict including the number of the label

    # entropy
    def entropy(self, rows):
        from math import log
        log2 = lambda x: log(x) / log(2)
        results = self.uniquecounts(rows)  
        # 开始计算熵的值
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows) # the possiblility of the label
            ent = ent - p * log2(p)
        return ent

    # misclassification
    def misclassification(self, rows):
        from math import log
        results = self.uniquecounts(rows)  
        max_mis = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            if max_mis < p:
                max_mis = p
        return 1-max_mis

    # gini
    def giniimpurity(self, rows):
        total = len(rows)
        results = self.uniquecounts(rows)
        imp = 0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            imp += p * p
        return 1-imp



    # 预减枝
    def pre_pruning(self, mingain):
        self.threshold = mingain

    # 后减枝
    def post_pruning(self, tree, mingain):
        # 如果分支不是叶节点，则对其进行剪枝
        if tree.tb.results is None:
            self.post_pruning(tree.tb, mingain)
        if tree.fb.results is None:
            self.post_pruning(tree.fb, mingain)
        # 如果两个子分支都是叶节点，判断是否能够合并
        if tree.tb.results is not None and tree.fb.results is not None:
            # 构造合并后的数据集
            tb, fb = [], []
            for v, c in tree.tb.results.items():
                tb += [[v]] * c
            for v, c in tree.fb.results.items():
                fb += [[v]] * c
            # 检查熵的减少量
            delta = self.entropy(tb + fb) - \
                (self.entropy(tb) + self.entropy(fb) / 2)
            if delta < mingain:
                # 合并分支
                tree.tb, tree.fb = None, None
                tree.results = self.uniquecounts(tb + fb)

    # 在某一列上对数据集进行拆分。可应用于数值型或因子型变量
    def divideset(self, rows, column, value):
        # 定义一个函数，判断当前数据行属于第一组还是第二组
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value
        # 将数据集拆分成两个集合，并返回
        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]
        return(set1, set2)

    # 以递归方式构造树

    def buildtree(self, rows, scoref=giniimpurity):
        if len(rows) == 0:
            return decisionnode()
        current_score = scoref(self, rows)
        # 定义一些变量以记录最佳拆分条件
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        # 去掉最后一个类别标签
        column_count = len(rows[0]) - 1
        for col in range(0, column_count):
            # 在当前列中生成一个由不同值构成的序列
            column_values = {}
            
            for row in rows:
                column_values[row[col]] = 1  # 初始化 {"a":1}
            # 根据这一列中的每个值，尝试对数据集进行拆分
            for value in column_values.keys():
		if col not in self.property:
                	(set1, set2) = self.divideset(rows, col, value)
                	# 信息增益
                	p = float(len(set1)) / len(rows)
                	gain = current_score - p * \
                    		scoref(self, set1) - (1 - p) * scoref(self, set2)
                	if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    		best_gain = gain
                    		best_criteria = (col, value)
                    		best_sets = (set1, set2)

        # 创建子分支
        if best_gain > self.threshold:
	    self.property.append(best_criteria[0])
            trueBranch = self.buildtree(best_sets[0])  # 递归调用
            falseBranch = self.buildtree(best_sets[1])
            return decisionnode(col=best_criteria[0], value=best_criteria[1],
                                tb=trueBranch, fb=falseBranch)
        else:
            return decisionnode(results=self.uniquecounts(rows))


    # 决策树的显示
    def printtree(self, tree, indent=''):
        # 是否是叶节点
        if tree.results is not None:
            print str(tree.results)
        else:
            # 打印判断条件
            print 'X' + str(tree.col) + " >= " + str(tree.value) + "? "
            # 打印分支
            print indent + "T->",
            self.printtree(tree.tb, indent + " ")
            print indent + "F->",
            self.printtree(tree.fb, indent + " ")

    # 对新的观测数据进行分类

    def classify(self, observation, tree):
        if tree.results is not None:
            return tree.results
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)


    def predict(self, X_test, tree):
        pre_label = []
        for i in X_test:
            pre_label.append(d.classify(i, tree).keys()[0])
        return pre_label


if __name__ == '__main__':
    d = DecisionTree()
    d.load_data("Iris.data")
    #d.pre_pruning(0.1)
    tree = d.buildtree(d.FeatureSet)
    pre_label = d.predict(d.testset, tree)
    d.post_pruning(tree, 0.1)
    ACC = metrics.accuracy_score(d.Label, pre_label)
    print 'ACC: ' + str(ACC)
    d.printtree(tree=tree)
