import nltk
from nltk.corpus import movie_reviews
from nltk.metrics import ConfusionMatrix
import pickle



revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]

revisoes_negativas = revisoes[0:1000]

revisoes_positivas = revisoes[1000:2000]

baseBalanceada = []
i = 0

while i < 1000:
    baseBalanceada.append(revisoes_positivas[i])
    baseBalanceada.append(revisoes_negativas[i])
    i= i+1


todas_palavras  = []

"Começando a construir o classificador Naive Bayes, pegando todas as palavras e montando o algoritmo "
for p in movie_reviews.words():
    todas_palavras.append(p.lower())

"Convertendo a lista de palavras para uma distribuição de frequencia do nltk"

todas_palavras = nltk.FreqDist(todas_palavras)

#Armazenando as palavras que mais aparecem  a partir da posicao 3000 em diante
palavra_atributo = list(todas_palavras.keys())[:3000]

#funcao que verifica se as palavras contidas em uma revisão estão entre as mais comuns, mapeia "Eu gostei muito do filme" -> 'Eu', false
#'gostei', true; 'muito', false.... E assim por diante
def encontra_atributos (revisao):
    #In python, set() is an unordered collection with no duplicate elements
    palavras = set(revisao)
    atributo = {}
    for p in palavra_atributo:
        atributo[p] = (p in palavras)
    return atributo



conjunto_atributos = [(encontra_atributos(rev), categoria) for (rev, categoria) in baseBalanceada]



todas = []
j = 0
while j < 10:
    print("Iteração =", j);
    teste = conjunto_atributos[0:200]
    treino = conjunto_atributos[200:]


    classificador = nltk.NaiveBayesClassifier.train(treino)

    print("Acurácia do Naive :", (nltk.classify.accuracy(classificador, teste)) * 100, "%")


    esperado = []
    previsto = []

    for (frase, classe) in conjunto_atributos:
        resultado = classificador.classify(frase)
        previsto.append(resultado)
        esperado.append(classe)

    matrizDeConfusao = ConfusionMatrix(esperado, previsto)
    print("-------------------MATRIZ DE CONFUSAO-----------------------------")
    print(matrizDeConfusao)


    for rev in teste:
        conjunto_atributos.remove(rev)

    for rev in teste:
        conjunto_atributos.append(rev)
    j = j+1;
