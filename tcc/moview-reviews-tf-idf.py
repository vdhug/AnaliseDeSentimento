from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.metrics import ConfusionMatrix




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

# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectors = vectorizer.fit_transform(X)

LinearSVC_classifier = LinearSVC()
MNB_classifier = MultinomialNB()
LogisticRegression_classifier = LogisticRegression()

for i in range(0, 10):
    X_test = vectors[:200]
    X_train = vectors[200:]

    y_test = y[:200]
    y_train = y[200:]

    # --------------------------------------------------ALGORITMO MNB INÍCIO-----------------------------------------------------------------------------------------------
    MNB_classifier.fit(X_train, y_train)
    esperado = []
    previsto = []

    for j in range(0, 200):
        resultado = MNB_classifier.predict(X_test[j])
        previsto.append(resultado)
        esperado.append(y_test[j])

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO MULTINOMINAL ALGORITHM-----------------------------")
    print(matrizDeConfusao)

    # -------------------------------ALGORITMO MULTINOMINAL ALGORITHM FINAL------------------------------------------------------------------------------------------------------------------------


    # --------------------------------------------------ALGORITMO REGRESSÃO LOGISTICA INÍCIO-----------------------------------------------------------------------------------------------
    LogisticRegression_classifier.fit(X_train, y_train)

    esperado = []
    previsto = []

    for j in range (0, 200):
        resultado = LogisticRegression_classifier.predict(X_test[j])
        previsto.append(resultado)
        esperado.append(y_test[j])

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO REGRESSÃO LOGISTICA-----------------------------")
    print(matrizDeConfusao)
    # -------------------------------ALGORITMO REGRESSÃO LOGISTICA FINAL------------------------------------------------------------------------------------------------------------------------


    # --------------------------------------------------ALGORITMO LINEAR SVC INÍCIO-----------------------------------------------------------------------------------------------
    LinearSVC_classifier.fit(X_train, y_train)
    esperado = []
    previsto = []

    for j in range(0, 200):
        resultado = LinearSVC_classifier.predict(X_test[j])
        previsto.append(resultado)
        esperado.append(y_test[j])

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO LINEAR SVC-----------------------------")
    print(matrizDeConfusao)
    # -------------------------------ALGORITMO LINEAR SVC FINAL---------------------------------------------------------------------------------------------------------------------------
    for rev in X_test:
        vectors.remove(rev)

    for rev in X_test:
        vectors.append(rev)

    for classe in y_test:
        y.remove(classe)

    for classe in y_test:
        y.append(classe)
