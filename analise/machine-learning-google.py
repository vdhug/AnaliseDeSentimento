
from sklearn import  tree

#Codigo criado a partir dos tutoriais Google Developers https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1

#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
#labels =["apple", "apple", "orange", "orange"]

#Convert to numbers
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()

#The training algorithm is included in the classifier object, and it's called fit (find patterns in data)
clf = clf.fit(features, labels)

previsto = clf.predict([[160, 0]])

print(previsto)




