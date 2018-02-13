from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class BasicoKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]



iris = datasets.load_iris()

X = iris.data #features
y = iris.target #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
knc = BasicoKNN()

knc.fit(X_train, y_train)

predictions = knc.predict(X_test)

print(predictions)

from sklearn.metrics import accuracy_score
print("Nosso classificador  Acurácia =", (accuracy_score(y_test, predictions))*100, "%")