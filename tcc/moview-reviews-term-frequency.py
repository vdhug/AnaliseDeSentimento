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


todas_palavras  = []

"Começando a construir o classificador Naive Bayes, pegando todas as palavras e montando o algoritmo "
for p in movie_reviews.words():
    todas_palavras.append(p.lower())

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


conjunto_atributos = [(term_frequency(rev), categoria) for (rev, categoria) in baseBalanceada]


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

