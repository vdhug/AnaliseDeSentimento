from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

textos = ["The movie is really really good", "Horrible movie", "Waste of time"]


vectorizer = CountVectorizer(stop_words="english", analyzer="word", binary=False)

vector = vectorizer.fit_transform(textos)
print(vector)
tfidf = TfidfTransformer()
tdm_tfidf = tfidf.fit_transform(vector)

print(tdm_tfidf)
print(vectorizer.get_feature_names())

