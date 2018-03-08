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


revocacaoSVM = []
precisaoSVM = []
f1SVM = []

revocacaoNB = []
precisaoNB = []
f1NB = []

revocacaoRL = []
precisaoRL = []
f1RL = []

for i in range(0, 10):
    teste = conjunto_atributos[0:200]
    treino = conjunto_atributos[200:]

    LinearSVC_classifier.train(treino)
    y_pred = []
    y_true = []
    for (frase, classe) in teste:
        r = LinearSVC_classifier.classify(frase)
        y_pred.append(r)
        y_true.append(classe)

    revocacaoSVM.append(recall_score(y_true, y_pred, average=None))
    precisaoSVM.append(precision_score(y_true, y_pred, average=None))
    f1SVM.append(f1_score(y_true, y_pred, average=None))

    MNB_classifier.train(treino)

    y_pred = []
    y_true = []
    for (frase, classe) in teste:
        r = MNB_classifier.classify(frase)
        y_pred.append(r)
        y_true.append(classe)

    revocacaoNB.append(recall_score(y_true, y_pred, average=None))
    precisaoNB.append(precision_score(y_true, y_pred, average=None))
    f1NB.append(f1_score(y_true, y_pred, average=None))

    LogisticRegression_classifier.train(treino)
    y_pred = []
    y_true = []
    for (frase, classe) in teste:
        r = LogisticRegression_classifier.classify(frase)
        y_pred.append(r)
        y_true.append(classe)

    revocacaoRL.append(recall_score(y_true, y_pred, average=None))
    precisaoRL.append(precision_score(y_true, y_pred, average=None))
    f1RL.append(f1_score(y_true, y_pred, average=None))


    for rev in teste:
        conjunto_atributos.remove(rev)

    for rev in teste:
        conjunto_atributos.append(rev)


# -----------------------------------------FINAL DA VALIDAÇÃO CRUZADA----------------------------------------------------------------------------------------------------------------




print("F1 NB\n", f1NB)
print("F1 RL\n", f1RL)
print("F1 SVM\n", f1SVM)

print("REVOCACAO NB\n", revocacaoNB)
print("REVOCACAO RL\n", revocacaoRL)
print("REVOCACAO SVM\n", revocacaoSVM)

print("PRECISAO NB\n", precisaoNB)
print("PRECISAO RL\n", precisaoRL)
print("PRECISAO SVM\n", precisaoSVM)


print("----------------MEDIAS  --------------------------------------")

print("F1 NB")
pos = 0
neg = 0

for i in f1NB:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)

print("F1 SVM")
pos = 0
neg = 0

for i in f1SVM:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)

print("F1 RL")
pos = 0
neg = 0

for i in f1RL:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)




print("revocacao NB")
pos = 0
neg = 0

for i in revocacaoNB:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)

print("revocacao SVM")
pos = 0
neg = 0

for i in revocacaoSVM:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)

print("revocacao RL")
pos = 0
neg = 0

for i in revocacaoRL:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)







print("precisao NB")
pos = 0
neg = 0

for i in precisaoNB:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)

print("precisao SVM")
pos = 0
neg = 0

for i in precisaoSVM:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)

print("precisao RL")
pos = 0
neg = 0

for i in precisaoRL:
    neg += float(i[0])
    pos += float(i[1])
print("POS = ", pos/10, "NEG=", neg/10)
