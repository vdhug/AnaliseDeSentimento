from nltk.corpus import movie_reviews
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.metrics import ConfusionMatrix
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from collections import Counter



revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]


revisoes_negativas = revisoes[0:1000]
revisoes_positivas = revisoes[1000:2000]

baseBalanceada = []
i = 0

while i < 1000:
    baseBalanceada.append(revisoes_positivas[i])
    baseBalanceada.append(revisoes_negativas[i])
    i= i+1

maxP = 0
minP = len(baseBalanceada[0][0])
minR = []
media = len(baseBalanceada[0][0])

for review in baseBalanceada:
    if len(review[0]) > maxP:
        maxP = len(review[0])

    if len(review[0]) < minP:
        minP = len(review[0])
        minR = review[0]
    media = (media + len(review[0]))/2

print(maxP)
print(minP)
print(minR)
print(media)