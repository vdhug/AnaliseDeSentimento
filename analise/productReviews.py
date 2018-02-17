import nltk
from nltk.corpus import product_reviews_1

nokia_reviews = product_reviews_1.reviews('Nokia_6610.txt')

print(len(nokia_reviews))

print(nokia_reviews[0])