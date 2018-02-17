from nltk.corpus import PlaintextCorpusReader
import json
from nltk.tokenize import sent_tokenize, word_tokenize

data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TVs\B00B93KG1A.json'))

print(str(data["Reviews"][0]["Content"]))
review = str(data["Reviews"][0]["Content"])

print(word_tokenize(review))

categoria = 'neg'
if float(data["Reviews"][0]["Overall"]) >= 3.0:
    categoria = 'pos'

print(categoria)

print(len(data["Reviews"]))
