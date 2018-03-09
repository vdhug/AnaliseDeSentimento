from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



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

X_train, X_test, y_train, y_test = train_test_split(vectors, y, test_size=0.33, random_state=42)


# initialise the SVM classifier
classifier = LinearSVC()

classifier.fit(X_train, y_train)

predicoes = classifier.predict(X_test)

print(list(predicoes[0:10]))
print(y_test[0:10])



print(accuracy_score(y_test, predicoes))






































