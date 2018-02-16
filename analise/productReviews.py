import nltk
from nltk.corpus import product_reviews_1


camera_reviews = product_reviews_1.reviews('Canon_G3.txt')

print(len(camera_reviews))

print(camera_reviews[1])