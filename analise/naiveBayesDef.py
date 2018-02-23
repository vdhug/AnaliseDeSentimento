import nltk
from nltk.corpus import movie_reviews
from nltk.metrics import ConfusionMatrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC



revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]

print(revisoes[1])

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
def encontra_atributos (revisao):
    #In python, set() is an unordered collection with no duplicate elements
    palavras = set(revisao)
    atributo = {}
    for p in palavra_atributo:
        atributo[p] = (p in palavras)
    return atributo



conjunto_atributos = [(encontra_atributos(rev), categoria) for (rev, categoria) in baseBalanceada]
print(conjunto_atributos[0])
#ALGORITMOS SENDO UTILIZADOS

MNB_classifier = SklearnClassifier(MultinomialNB())
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SVC_classifier = SklearnClassifier(SVC())
LinearSVC_classifier = SklearnClassifier(LinearSVC())
NuSVC_classifier = SklearnClassifier(NuSVC())

todas = []
j = 0
#-----------------------------------------INÍCIO DA VALIDAÇÃO CRUZADA----------------------------------------------------------------------------------------------------------------
while j < 10:
    print("Iteração =", j)
    teste = conjunto_atributos[0:200]
    treino = conjunto_atributos[200:]



#--------------------------------------------------ALGORITMO NAIVE BAYES INÍCIO-----------------------------------------------------------------------------------------------
    classificador = nltk.NaiveBayesClassifier.train(treino)
    print("Acurácia do Naive :", (nltk.classify.accuracy(classificador, teste)) * 100, "%")


    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO NAIVE BAYES-----------------------------")
    print(matrizDeConfusao)
#-------------------------------ALGORITMO NAIVE BAYES FINAL------------------------------------------------------------------------------------------------------------------------



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



# --------------------------------------------------ALGORITMO BERNOULLI INÍCIO-----------------------------------------------------------------------------------------------
    BernoulliNB_classifier.train(treino)
    print("Acurácia BernoulliNB_classificador :", (nltk.classify.accuracy(BernoulliNB_classifier, teste)) * 100)
    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = BernoulliNB_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO BERNOULLI ALGORITHM-----------------------------")
    print(matrizDeConfusao)
# -------------------------------ALGORITMO BERNOULLI ALGORITHM FINAL------------------------------------------------------------------------------------------------------------------------



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




# --------------------------------------------------ALGORITMO SGDC INÍCIO-----------------------------------------------------------------------------------------------
    SGDClassifier_classifier.train(treino)
    print("Acurácia SGDClassifier_classificador :", (nltk.classify.accuracy(SGDClassifier_classifier, teste)) * 100)
    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = SGDClassifier_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO REGRESSÃO LOGISTICA-----------------------------")
    print(matrizDeConfusao)
# -------------------------------ALGORITMO SGDC FINAL------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------ALGORITMO SVC INÍCIO-----------------------------------------------------------------------------------------------
    SVC_classifier.train(treino)
    print("Acurácia SVC_classificador :", (nltk.classify.accuracy(SVC_classifier, teste)) * 100)
    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = SVC_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO SVC-----------------------------")
    print(matrizDeConfusao)
# -------------------------------ALGORITMO SVC FINAL------------------------------------------------------------------------------------------------------------------------




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


# --------------------------------------------------ALGORITMO LINEAR SVC INÍCIO-----------------------------------------------------------------------------------------------
    NuSVC_classifier.train(treino)
    print("Acurácia NuSVC_classificador :", (nltk.classify.accuracy(NuSVC_classifier, teste)) * 100)
    esperado = []
    previsto = []

    for (frase, classe) in teste:
        resultado = NuSVC_classifier.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO NuSVC -----------------------------")
    print(matrizDeConfusao)
# -------------------------------ALGORITMO LINEAR SVC FINAL---------------------------------------------------------------------------------------------------------------------------

    for rev in teste:
        conjunto_atributos.remove(rev)

    for rev in teste:
        conjunto_atributos.append(rev)

    j = j + 1
#-----------------------------------------FINAL DA VALIDAÇÃO CRUZADA----------------------------------------------------------------------------------------------------------------
