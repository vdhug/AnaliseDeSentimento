from collections import Counter

palavra_atributo = ["Ola", "Mundo", "Cruel"]

def term_frequency(revisao):
    counts = Counter(revisao)
    atributo = {}
    for p in palavra_atributo:
        if p in revisao:
            atributo[p] = counts[p]
        else:
            atributo[p] = 0
    return atributo


