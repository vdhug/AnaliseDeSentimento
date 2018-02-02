import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

test_index = [0, 50, 100]

train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis=0)

test_target = iris.target[test_index]
test_data = iris.data[test_index]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

#visualizaon code
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("iris.pdf")



#print (iris.feature_names)
#print (iris.target_names)
#print (iris.data[0])

