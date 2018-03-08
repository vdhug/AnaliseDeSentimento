from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords


docA = "The The cat"
docB = "The The dog"


corpus = []
corpus.append(docA)
corpus.append(docB)


stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X)
X.shape


lsa = TruncatedSVD(n_components=2, n_iter=2)
lsa.fit(X)
terms = vectorizer.get_feature_names()


