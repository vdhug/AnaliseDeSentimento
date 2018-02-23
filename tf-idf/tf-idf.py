from nltk.corpus import movie_reviews
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer



revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]

revisoes_negativas = revisoes[0:1000]

revisoes_positivas = revisoes[1000:2000]


X = []
y = []

for i in range(0, 1000):
    X.append(str(revisoes_positivas[i][0]))
    X.append(str(revisoes_negativas[i][0]))

    y.append(1)
    y.append(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)

cv1 = TfidfVectorizer(min_df=1,stop_words='english')

x_traincv = cv1.fit_transform(X_train)

a = x_traincv.toarray()


x_testcv=cv1.transform(X_test)

mnb = MultinomialNB()

mnb.fit(x_traincv,y_train)

predictions=mnb.predict(x_testcv)

a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1


print(count)
print(len(predictions))

print(count/len(predictions)*100)