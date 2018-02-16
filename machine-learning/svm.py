import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digitos = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

print(len(digitos.data))

x, y = digitos.data[:-1], digitos.target[:-1]
clf.fit(x, y)

print("Valor predito =", clf.predict(digitos.data[-1:]))

print("Valor real =", digitos.target[-1:])

plt.imshow(digitos.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
