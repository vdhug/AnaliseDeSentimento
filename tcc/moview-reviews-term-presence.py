from nltk.corpus import movie_reviews
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.metrics import ConfusionMatrix
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier



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
def encontra_atributos (revisao):
    #In python, set() is an unordered collection with no duplicate elements
    palavras = set(revisao)
    atributo = {}
    for p in palavra_atributo:
        atributo[p] = (p in palavras)
    return atributo


conjunto_atributos = [(encontra_atributos(rev), categoria) for (rev, categoria) in baseBalanceada]


LinearSVC_classifier = SklearnClassifier(LinearSVC())
MNB_classifier = SklearnClassifier(MultinomialNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

from sklearn.metrics import recall_score, precision_score, f1_score



revocacaoNB = 0
revocacaoRL = 0
revocacaoSVM = 0

precisaoNB = 0
precisaoRL = 0
precisaoSVM = 0


f1NB = 0
f1RL = 0
f1SVM = 0

for i in range(0, 10):
    print("Iteração =", i)
    teste = conjunto_atributos[0:200]
    treino = conjunto_atributos[200:]

    # --------------------------------------------------ALGORITMO MNB INÍCIO-----------------------------------------------------------------------------------------------
    MNB_classifier.train(treino)

    y_pred = []
    y_true = []
    for (frase, classe) in teste:
        r = MNB_classifier.classify(frase)
        y_pred.append(r)
        y_true.append(classe)


    #neg, pos
    revocacaoNB = revocacaoNB + float(recall_score(y_true, y_pred, average=None)[1])
    precisaoNB = precisaoNB + float(precision_score(y_true, y_pred, average=None)[1])
    f1NB = f1NB + float(f1_score(y_true, y_pred, average=None)[1])


    # -------------------------------ALGORITMO MULTINOMINAL ALGORITHM FINAL------------------------------------------------------------------------------------------------------------------------


    # --------------------------------------------------ALGORITMO LogisticRegression_classifier INÍCIO-----------------------------------------------------------------------------------------------
    LogisticRegression_classifier.train(treino)

    y_pred = []
    y_true = []
    for (frase, classe) in teste:
        r = LogisticRegression_classifier.classify(frase)
        y_pred.append(r)
        y_true.append(classe)

    # neg, pos
    revocacaoRL = revocacaoRL + float(recall_score(y_true, y_pred, average=None)[1])
    precisaoRL = precisaoRL + float(precision_score(y_true, y_pred, average=None)[1])
    f1RL = f1RL + float(f1_score(y_true, y_pred, average=None)[1])

    # -------------------------------ALGORITMO MULTINOMINAL ALGORITHM FINAL------------------------------------------------------------------------------------------------------------------------





    # --------------------------------------------------ALGORITMO MNB INÍCIO-----------------------------------------------------------------------------------------------
    LinearSVC_classifier.train(treino)

    y_pred = []
    y_true = []
    for (frase, classe) in teste:
        r = LinearSVC_classifier.classify(frase)
        y_pred.append(r)
        y_true.append(classe)

    # neg, pos
    revocacaoSVM = revocacaoSVM + float(recall_score(y_true, y_pred, average=None)[1])
    precisaoSVM = precisaoSVM + float(precision_score(y_true, y_pred, average=None)[1])
    f1SVM = f1SVM + float(f1_score(y_true, y_pred, average=None)[1])

    # -------------------------------ALGORITMO MULTINOMINAL ALGORITHM FINAL------------------------------------------------------------------------------------------------------------------------





    for rev in teste:
        conjunto_atributos.remove(rev)

    for rev in teste:
        conjunto_atributos.append(rev)


# -----------------------------------------FINAL DA VALIDAÇÃO CRUZADA----------------------------------------------------------------------------------------------------------------

print("Revocações")
print("NB",revocacaoNB/10)
print("RL",revocacaoRL/10)
print("SVM",revocacaoSVM/10)



print("Precisoes")
print("NB", precisaoNB/10)
print("RL", precisaoRL/10)
print("SVM", precisaoSVM/10)

print("F1")
print("NB",f1NB/10)
print("RL",f1RL/10)
print("SVM",f1SVM/10)
