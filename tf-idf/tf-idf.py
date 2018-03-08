
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 1))

docA = "The movie is really really good"
docB = "Horrible movie"
docC = "Waste of time"


X = []
X.append(docA)
X.append(docB)
X.append(docC)


vectors = vectorizer.fit_transform(X)


terms = vectorizer.get_feature_names()
print(terms)
print(vectors)