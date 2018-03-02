from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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

# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectors = vectorizer.fit_transform(X)

LinearSVC_classifier = LinearSVC()
MNB_classifier = MultinomialNB()
LogisticRegression_classifier = LogisticRegression()

vectors = vectorizer.fit_transform(X)

#f1, precision, recall

#métricas disponíveis ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
# 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
# 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error',
# 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score',
# 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
# 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
metrics = ["f1", "precision", "recall"]


algoritmos = [LinearSVC_classifier, MNB_classifier, LogisticRegression_classifier]
nomes = ["SVM", "NB", "Regressao Logistica"]

for i in range(0,3):
    print("------------------------------------------", nomes[i], "----------------------------------------------")
    for metric in metrics:
        scores = cross_val_score(algoritmos[i], vectors, y, cv=10, scoring=metric)
        print(metric)
        media = 0
        for num in scores:
            media += float(num)

        print("Média =", media/10)

    print("------------------------------------------")

