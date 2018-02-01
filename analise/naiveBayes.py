import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import pickle




"Lendo todas as coisas em apenas uma linha, confuso (?) melhor o outro jeito"


revisoes = [(list (movie_reviews.words(fileid)), categoria)
            for categoria in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categoria)]

#print("Inicio do meu teste\n")

revisoes_negativas = revisoes[0:1000]
print(revisoes_negativas[0][1])

revisoes_positivas = revisoes[1000:2000]
print(revisoes_positivas[0][1])
print("INICIO")
i = 0
print(i)
revisoes_aux = revisoes_positivas[0:10]
print("REVISOES EM AUX")
for rev in revisoes_aux:
    print(rev[1])
todas = []

while i <10:
    todas.append(revisoes_aux[i])
    i+=1


print("REVISOES EM TODAS")

for rev in todas:
    print(rev[1])
"""
while i<10:
    for rev in revisoes_aux:
          todas.append(rev)

    teste = todas[i:i+2]
    print("Iteracao = ", i)
    for rev in teste:
        todas.remove(rev)

    treino = todas;

    for r in teste:
        print(r)

    print(len(teste))
    print(len(treino))
    i = i + 2

    todas = []
    print(len(todas))


for rev in revisoes_aux:
    print(rev)

print(len(revisoes_aux))
print("FIM")

"""
"""
revisao_aux1 = revisoes[0:10]

revisao_geral = []
i = 0
while i <1:
    aux = i;
    while aux < 3:
        aux_revisao = revisao_aux1[aux:(aux%10)]
        for aux_revisao_aux in aux_revisao:
            revisao_geral.append(aux_revisao_aux);
        aux = aux +1
    i = i+1

print(len(revisao_aux1))
print(len(revisao_geral))
print(revisao_aux1[0])
print(revisao_geral[0])

print("Fim do meu teste")
"""
"""
revisoes = []

for categoria in movie_reviews.categories():
    for fileid in movie_reviews.fileids(categoria):
        revisoes.append(list(movie_reviews.words(fileid), categoria))
"""

#print(len(revisoes), "\n")

random.shuffle(revisoes)

#print(revisoes[1999][1])


todas_palavras  = []

"Começando a construir o classificador Naive Bayes, pegando todas as palavras e montando o algoritmo "
for p in movie_reviews.words():
    todas_palavras.append(p.lower())

"Convertendo a lista de palavras para uma distribuição de frequencia do nltk"

todas_palavras = nltk.FreqDist(todas_palavras)

#print(todas_palavras.most_common(10))

"Bom pra ver como a remoção das stop words são necessárias, elas são a maioria nas palavras"

"Quantidade de palavras, acentuaçoes, e etc diferentes existentes na base de dados"
#print(len(todas_palavras))

#print(todas_palavras["stupid"])

#Armazenando as palavras que mais aparecem  a partir da posicao 3000 em diante
palavra_atributo = list(todas_palavras.keys())[:3000]
palavra_sem_stop = []
stop_words = set(stopwords.words("english"))
for p in palavra_atributo:
    if p not in stop_words:
        palavra_sem_stop.append(p)

#print("Total de palavras removendo as stop words", len(palavra_sem_stop), "\n")


#print("\n", len(todas_palavras))
#print("\n", len(palavra_atributo))
#print("\n", len(todas_palavras)-len(palavra_atributo))

#print("\n", revisoes[1][0])
#print("\n", set(revisoes[1][0]))
#funcao que verifica se as palavras contidas em uma revisão estão entre as mais comuns, mapeia "Eu gostei muito do filme" -> 'Eu', false
#'gostei', true; 'muito', false.... E assim por diante
def encontra_atributos (revisao):
    #In python, set() is an unordered collection with no duplicate elements
    palavras = set(revisao)
    atributo = {}
    for p in palavra_atributo:
        atributo[p] = (p in palavras)
    return atributo


#print("\n", (encontra_atributos(revisoes[1][0])))

conjunto_atributos = [(encontra_atributos(rev), categoria) for (rev, categoria) in revisoes]

#print(conjunto_atributos)
#print("TAMANHO DO CONJUNTO DE ATRIBUTOS")
#print(len(conjunto_atributos))


treino = conjunto_atributos[:1900]
teste = conjunto_atributos[1900:]

classificador = nltk.NaiveBayesClassifier.train(treino)



#Explicacao básica para nao se perder
#Naive bayes funciona de forma que palavras que aparecem mais vezes em revisoe que sao positivas, logo arma aquela tabela louca
# e implica que consegue definir a classe.
print("Acurácia do Naive :", (nltk.classify.accuracy(classificador, teste))*100, "%")
classificador.show_most_informative_features(10)


classificador_salvo = open("classificador_naive", "rb")
classificador = pickle.load(classificador_salvo)
classificador_salvo.close()

print("Acurácia do Naive com classificador salvo = ", (nltk.classify.accuracy(classificador, teste))*100, "%")
classificador.show_most_informative_features(10)


# processo para salvar o nosso classificador

#salvar_classificador = open("classificador_naive", "wb")
#pickle.dump(classificador, salvar_classificador)
#salvar_classificador.close()

