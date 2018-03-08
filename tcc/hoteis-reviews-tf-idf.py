from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import cross_val_score
import json
from nltk.tokenize import sent_tokenize, word_tokenize


data1 = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\260444.json'))
data2 = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\774804.json'))
data3 = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\2515499.json'))

reviews_pos = []
reviews_neg = []

for review in data1["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        objeto = word_tokenize(str(review["Content"])), 'neg'
        reviews_neg.append(objeto)

    if float((review["Ratings"]["Overall"])) >=4:
        objeto = word_tokenize(str(review["Content"])), 'pos'
        reviews_pos.append(objeto)

for review in data2["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        objeto = word_tokenize(str(review["Content"])), 'neg'
        reviews_neg.append(objeto)
    if float((review["Ratings"]["Overall"])) >=4:
        objeto = word_tokenize(str(review["Content"])), 'pos'
        reviews_pos.append(objeto)

for review in data3["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        objeto = word_tokenize(str(review["Content"])), 'neg'
        reviews_neg.append(objeto)
    if float((review["Ratings"]["Overall"])) >=4:
        objeto = word_tokenize(str(review["Content"])), 'pos'
        reviews_pos.append(objeto)


#8079
#1059
revisoes_negativas = reviews_neg[0:1000]

revisoes_positivas = reviews_pos[0:1000]


X = []
y = []

for i in range(0, 1000):
    X.append(' '.join(revisoes_positivas[i][0]))
    X.append(' '.join(revisoes_negativas[i][0]))

    y.append(1)
    y.append(0)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,1))


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
        obj = X[i]
        X.remove(obj)
        X.append(obj)

        aux = y[i]
        y.remove(aux)
        y.append(aux)





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
