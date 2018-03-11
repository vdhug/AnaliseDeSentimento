from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import cross_val_score




revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]

revisoes_negativas = revisoes[0:1000]

revisoes_positivas = revisoes[1000:2000]


X = []
y = []

for i in range(0, 1000):
    X.append(' '.join(revisoes_positivas[i][0]))
    X.append(' '.join(revisoes_negativas[i][0]))

    y.append(1)
    y.append(0)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer(stop_words="english", analyzer="word", binary=False)

LinearSVC_classifier = LinearSVC()
MNB_classifier = MultinomialNB()
LogisticRegression_classifier = LogisticRegression()

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

aux = []

for classe in y:
    aux.append(classe)


for i in range(0, 10):
    vectors = vectorizer.fit_transform(X)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(vectors, y, test_size=0.1, random_state=None, shuffle=False)

    LinearSVC_classifier.fit(X_train, y_train)
    y_pred = LinearSVC_classifier.predict(X_test)

    revocacaoSVM.append(recall_score(y_test, y_pred, average=None))
    precisaoSVM.append(precision_score(y_test, y_pred, average=None))
    f1SVM.append(f1_score(y_test, y_pred, average=None))

    MNB_classifier.fit(X_train, y_train)

    y_pred = MNB_classifier.predict(X_test)

    revocacaoNB.append(recall_score(y_test, y_pred, average=None))
    precisaoNB.append(precision_score(y_test, y_pred, average=None))
    f1NB.append(f1_score(y_test, y_pred, average=None))

    LogisticRegression_classifier.fit(X_train, y_train)
    y_pred = LogisticRegression_classifier.predict(X_test)

    revocacaoRL.append(recall_score(y_test, y_pred, average=None))
    precisaoRL.append(precision_score(y_test, y_pred, average=None))
    f1RL.append(f1_score(y_test, y_pred, average=None))

    for i in range(0, 200):
        obj = X[0]
        X.remove(obj)
        X.append(obj)

        obj = aux[0]
        aux.remove(obj)
        aux.append(obj)

    y = []
    for classe in aux:
        y.append(classe)


# -----------------------------------------FINAL DA VALIDAÇÃO CRUZADA----------------------------------------------------------------------------------------------------------------




print("F1 NB\n", f1NB)
print("F1 SVM\n", f1SVM)
print("F1 RL\n", f1RL)

print("REVOCACAO NB\n", revocacaoNB)
print("REVOCACAO SVM\n", revocacaoSVM)
print("REVOCACAO RL\n", revocacaoRL)

print("PRECISAO NB\n", precisaoNB)
print("PRECISAO SVM\n", precisaoSVM)
print("PRECISAO RL\n", precisaoRL)


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