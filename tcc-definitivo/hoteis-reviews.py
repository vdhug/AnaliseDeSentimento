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

mediaPalavrasPorFold = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
mediaRevisoesPositivas = len(reviews_pos[0][0])
mediaRevisoesNegativas = len(reviews_neg[0][0])

for i in range(1, 1000):
    mediaRevisoesPositivas = (mediaRevisoesPositivas  + len(reviews_pos[i][0]))/2
    mediaRevisoesNegativas = (mediaRevisoesPositivas + len(reviews_neg[i][0]))/2



print("Média do tamanho das revisões positivas =", mediaRevisoesPositivas)
print("Média do tamanho das revisões negativas =", mediaRevisoesNegativas)
print("Diferença entre as médias é =", mediaRevisoesPositivas-mediaRevisoesNegativas)


fold10 = len(reviews_neg[900][0])
for i in range(901, 1000):
    fold10 = (fold10+len(reviews_neg[i][0]))/2

print("Fold 10", fold10)
treino = len(reviews_neg[0][0])
for i in range(1, 900):
    treino = (treino + len(reviews_neg[i][0])) / 2
print("Treino =", treino)

for i in range(0, 1000):
    base_balanceada.append(reviews_pos[i])
    base_balanceada.append(reviews_neg[i])


todas_palavras  = []

"Começando a construir o classificador Naive Bayes, pegando todas as palavras e montando o algoritmo "
for review in base_balanceada:
    for palavra in review[0]:
        todas_palavras.append(palavra)


"Convertendo a lista de palavras para uma distribuição de frequencia do nltk"

todas_palavras = nltk.FreqDist(todas_palavras)

#Armazenando as palavras que mais aparecem  a partir da posicao 3000 em diante
palavra_atributo = list(todas_palavras.keys())[:3000]

"""
#Media de atributos
mediaNegFold1 = 0
for i in range(200, 400):
    if base_balanceada[i][1] == 'neg':
        for palavra in base_balanceada[i][0]:
            if palavra in palavra_atributo:
                mediaNegFold1 += 1

print("Media das negativas do fold 1 =", mediaNegFold1/100)

mediaNegFold10 = 0
for i in range(1800, 2000):
    if base_balanceada[i][1] == 'neg':
        for palavra in base_balanceada[i][0]:
            if palavra in palavra_atributo:
                mediaNegFold10 += 1

print("Media das negativas do fold 10 =", mediaNegFold10/100)

"""

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