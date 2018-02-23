from nltk.corpus import movie_reviews
import nltk
revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]



todas_palavras  = []

"Começando a construir o classificador Naive Bayes, pegando todas as palavras e montando o algoritmo "
for p in movie_reviews.words():
    todas_palavras.append(p.lower())

"Convertendo a lista de palavras para uma distribuição de frequencia do nltk"

todas_palavras = nltk.FreqDist(todas_palavras)

#Armazenando as palavras que mais aparecem  a partir da posicao 3000 em diante
palavra_atributo = list(todas_palavras.keys())[:3000]


revisoes_negativas = revisoes[0:1000]
print(revisoes_negativas[0][0])
print(len(revisoes_negativas[0][0]))
revisoes_positivas = revisoes[1000:2000]

for i in range(0, 1000):
    rev = []

    for palavra in revisoes_negativas[i][0]:
        if palavra in palavra_atributo:
            rev.append(palavra)
    revisoes_negativas[i][0] = tuple(rev)

    rev = []

    for palavra in revisoes_positivas[i][0]:
        if palavra in palavra_atributo:
            rev.append(palavra)
    revisoes_positivas[i][0] = tuple(rev)

print(revisoes_negativas[0][0])
print(len(revisoes_negativas[0][0]))


X = []
y = []
for i in range(0, 1000):
    X.append(str(revisoes_positivas[i][0]))
    X.append(str(revisoes_negativas[i][0]))

    y.append(revisoes_positivas[i][1])
    y.append(revisoes_negativas[i][1])

