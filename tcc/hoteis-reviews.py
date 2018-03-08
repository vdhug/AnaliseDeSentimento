from nltk.corpus import movie_reviews
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.metrics import ConfusionMatrix
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from collections import Counter
import json
from nltk.tokenize import sent_tokenize, word_tokenize


#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\260444.json')) 469 negativas Positivas = 2052

#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\774804.json')) Negativas = 118 Positivas = 2097

#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\2515499.json')) Negativas = 472 Positivas = 3930

#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\91703.json'))Negativas = 461 Positivas = 3802

data1 = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\260444.json'))
data2 = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\774804.json'))
data3 = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\2515499.json'))

neg = 0
pos = 0

reviews_pos = []
reviews_neg = []

for review in data1["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        objeto = word_tokenize(str(review["Content"])), 'neg'
        reviews_neg.append(objeto)
        neg +=1

    if float((review["Ratings"]["Overall"])) >=4:
        objeto = word_tokenize(str(review["Content"])), 'pos'
        reviews_pos.append(objeto)
        pos +=1

for review in data2["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        objeto = word_tokenize(str(review["Content"])), 'neg'
        reviews_neg.append(objeto)
        neg +=1
    if float((review["Ratings"]["Overall"])) >=4:
        objeto = word_tokenize(str(review["Content"])), 'pos'
        reviews_pos.append(objeto)
        pos +=1

for review in data3["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        objeto = word_tokenize(str(review["Content"])), 'neg'
        reviews_neg.append(objeto)
        neg +=1
    if float((review["Ratings"]["Overall"])) >=4:
        objeto = word_tokenize(str(review["Content"])), 'pos'
        reviews_pos.append(objeto)
        pos +=1

base_balanceada = []

for i in range(0, 1000):
    base_balanceada.append(reviews_pos[i])
    base_balanceada.append(reviews_neg[i])

maxP = 0
minP = len(base_balanceada[0][0])
minR = []
media = len(base_balanceada[0][0])

for review in base_balanceada:
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