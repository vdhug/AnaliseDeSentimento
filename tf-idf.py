from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords


docA = "O gato dorme na cama"
docB = "O cachorro dorme na varanda"

corpus = []
corpus.append(docA)
corpus.append(docB)


stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,1))
X = vectorizer.fit_transform(corpus)
print(X[0])
X.shape
lsa = TruncatedSVD(n_components=2, n_iter=2)
lsa.fit(X)
terms = vectorizer.get_feature_names()

print(terms[0])
print(X[1])
print(terms[1])

