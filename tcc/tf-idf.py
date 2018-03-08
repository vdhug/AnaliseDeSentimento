docA = "The movie is really really good"
docB = "Horrible movie"
docC = "Waste of time"

bowA = docA.split(" ")
bowB = docB.split(" ")
bowC = docC.split(" ")

wordSet= set(bowA).union(set(bowB).union(set(bowC)))

print(len(wordSet))
wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)
wordDictC = dict.fromkeys(wordSet, 0)

for word in bowA:
    wordDictA[word]+=1

for word in bowB:
    wordDictB[word]+=1

for word in bowC:
    wordDictC[word]+=1


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict

tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)
tfBowC = computeTF(wordDictC, bowC)

print(tfBowA)

import math
def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    # counts the number of documents that contain a word w
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    #divide N by denominator above, take the log of that
    for word, val in idfDict.items():
        div = N / val
        idfDict[word] = math.log(div)

    return idfDict


idfs = computeIDF([wordDictA, wordDictB, wordDictC])

print(idfs)

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf




tfidfBowA =  computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)
tfidfBowC = computeTFIDF(tfBowC, idfs)


#Lastly I'll stick those into a matrix.
import pandas as pd
print(pd.DataFrame([tfidfBowA, tfidfBowB, tfidfBowC]))