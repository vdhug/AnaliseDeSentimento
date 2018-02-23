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

# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectors = vectorizer.fit_transform(X)

LinearSVC_classifier = LinearSVC()
MNB_classifier = MultinomialNB()
LogisticRegression_classifier = LogisticRegression()

vectors = vectorizer.fit_transform(X)

#métricas disponíveis ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
# 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
# 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error',
# 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score',
# 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
# 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
scores = cross_val_score(LinearSVC_classifier, vectors, y, cv=10, scoring="accuracy")

print(scores)


