from nltk.corpus import PlaintextCorpusReader
import json
from nltk.tokenize import sent_tokenize, word_tokenize

data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\91703.json'))

#C:\Users\Vitor\Documents\TCC\Base de dados\TVs\B00B93KG1A.json 0 negativas 2623 positivas
#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\260444.json')) 469 negativas Positivas = 2052

#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\774804.json')) Negativas = 118 Positivas = 2097

#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\2515499.json')) Negativas = 472 Positivas = 3930

#data = json.load(open(r'C:\Users\Vitor\Documents\TCC\Base de dados\TripAdvisorJson\json\91703.json'))Negativas = 461 Positivas = 3802
neg = 0
pos = 0
for review in data["Reviews"]:

    if float((review["Ratings"]["Overall"])) <=2:
        neg +=1
    if float((review["Ratings"]["Overall"])) >=4:
        pos +=1

print(data["Reviews"][0])
print(data["Reviews"][0]["Ratings"])
print(float(data["Reviews"][0]["Ratings"]["Overall"]))
print("Negativas =", neg)
print("Positivas =", pos)


