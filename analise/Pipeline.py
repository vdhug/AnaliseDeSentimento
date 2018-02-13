from sklearn import datasets

iris = datasets.load_iris()

X = iris.data #features
y = iris.target #labels
#Pensamos no classificador como uma função f(X) = y -> por isso temos que X são os atributos (features) e y são as classes (labels)
from sklearn.model_selection import train_test_split

#here we are pationating our data set
#We are taking the X's and Y's (remember that they are like an "array") and partionating in two sets
#X_trains are the features (atributos) for the set of training, and y_train are the labels for the set of training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

#create the classifier
from sklearn import tree
arvoreDecisao = tree.DecisionTreeClassifier()

arvoreDecisao.fit(X_train, y_train)

predictions = arvoreDecisao.predict(X_test)

print(predictions)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print("Arvore de Decisao Acurácia =", (accuracy_score(y_test, predictions))*100, "%")

#create the classifier KNeighbors
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()

knc.fit(X_train, y_train)

predictions = knc.predict(X_test)

print(predictions)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print("K Neighboors Acurácia =", (accuracy_score(y_test, predictions))*100, "%")