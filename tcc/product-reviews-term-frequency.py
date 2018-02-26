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


todas_palavras  = []

"Começando a construir o classificador Naive Bayes, pegando todas as palavras e montando o algoritmo "
for review in base_balanceada:
    for palavra in review[0]:
        todas_palavras.append(palavra)


"Convertendo a lista de palavras para uma distribuição de frequencia do nltk"

todas_palavras = nltk.FreqDist(todas_palavras)

#Armazenando as palavras que mais aparecem  a partir da posicao 3000 em diante
palavra_atributo = list(todas_palavras.keys())[:3000]





#funcao que verifica se as palavras contidas em uma revisão estão entre as mais comuns, mapeia "Eu gostei muito do filme" -> 'Eu', false
#'gostei', true; 'muito', false.... E assim por diante
def term_frequency(revisao):
    counts = Counter(revisao)
    atributo = {}
    for p in palavra_atributo:
        if p in revisao:
            atributo[p] = counts[p]
        else:
            atributo[p] = 0
    return atributo


conjunto_atributos = [(term_frequency(rev), categoria) for (rev, categoria) in base_balanceada]


LinearSVC_classifier = SklearnClassifier(LinearSVC())
MNB_classifier = SklearnClassifier(MultinomialNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

for i in range(0, 10):
    print("Iteração =", i)
    teste = conjunto_atributos[0:200]
    treino = conjunto_atributos[200:]

    # --------------------------------------------------ALGORITMO MNB INÍCIO-----------------------------------------------------------------------------------------------
    MNB_classifier.train(treino)
    print("Acurácia MNB_classificador :", (nltk.classify.accuracy(MNB_classifier, teste)) * 100)

    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = MNB_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO MULTINOMINAL ALGORITHM-----------------------------")
    print(matrizDeConfusao)

    # -------------------------------ALGORITMO MULTINOMINAL ALGORITHM FINAL------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------ALGORITMO REGRESSÃO LOGISTICA INÍCIO-----------------------------------------------------------------------------------------------
    LogisticRegression_classifier.train(treino)
    print("Acurácia LogisticRegression_classificador :", (nltk.classify.accuracy(LogisticRegression_classifier, teste)) * 100)
    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = LogisticRegression_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO REGRESSÃO LOGISTICA-----------------------------")
    print(matrizDeConfusao)
# -------------------------------ALGORITMO REGRESSÃO LOGISTICA FINAL------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------ALGORITMO LINEAR SVC INÍCIO-----------------------------------------------------------------------------------------------
    LinearSVC_classifier.train(treino)
    print("Acurácia LinearSVC_classificador :", (nltk.classify.accuracy(LinearSVC_classifier, teste)) * 100)
    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = LinearSVC_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO LINEAR SVC-----------------------------")
    print(matrizDeConfusao)
# -------------------------------ALGORITMO LINEAR SVC FINAL---------------------------------------------------------------------------------------------------------------------------
    for rev in teste:
        conjunto_atributos.remove(rev)

    for rev in teste:
        conjunto_atributos.append(rev)


# -----------------------------------------FINAL DA VALIDAÇÃO CRUZADA----------------------------------------------------------------------------------------------------------------

