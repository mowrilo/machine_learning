######################################################
#   TP1 - Processamento de Linguagem Natural         #
#   Aluno: Murilo Vale Ferreira Menezes - 2013030996 #
#   Prof. Adriano Veloso                             #
######################################################

import pandas as pd
import numpy as np
import codecs
import os
import gensim
from gensim.models import Word2Vec

def convertascii(text): #Função para codificar a informação dos livros como ASCII
    text = text.encode("utf-8")
    text = text.decode("utf-8")
    textASCII = text.encode("ascii","ignore")
    return textASCII

titles = [] #Informação da coleção
authors = []
numbers = []
filen = []

nModel = 0
for file in os.listdir("."):
    f = codecs.open(file, "r", encoding="utf-8")
    raw = f.read()
    ls=gensim.models.word2vec.LineSentence(file)
    a = Word2Vec(ls, size=200,window=8,workers=4,sg=1) #Treina e salva um modelo para cada livro
    a.save("../models/" + str(nModel))
    
    posTitle = raw.find("Title: ") #Obtém informações como autor, título e o nome do arquivo de cada livro
    tit = raw[(posTitle+7):(raw.find("\n",posTitle)-1)]
    titles.append(convertascii(tit))
    posAut = raw.find("Author: ")
    if (posAut != -1):
        aut = raw[(posAut+8):(raw.find("\n",posAut)-1)]
        authors.append(convertascii(aut))
    else:
        authors.append('NOAUTHOR')
    numbers.append(nModel)
    filen.append(file)
    nModel += 1

bookmap = pd.DataFrame(data={'number':numbers,'title':titles,'author':authors,'filename':filen}) #Informações gerais para análise posterior
bookmap.to_csv('../bookmap.csv')

finalmat = np.zeros([70,70])

for numberTerms in [250,500,750]: #Quantidade de amostragens diferentess
    print("Running on " + str(numberTerms))
    for model1 in xrange(0,70,1):
        print("\tRunning from model " + str(model1))
        m1 = Word2Vec.load("../models/"+str(model1)) #Carrega o primeiro modelo
        voc1 = m1.wv.vocab 
        for model2 in xrange(model1+1,70,1):
            print("\t\tTo model " + str(model2))
            m2 = Word2Vec.load("../models/"+str(model2)) #Carrega o segundo modelo
            voc2 = m2.wv.vocab
            inter = [w for w in voc1 if W in voc2] #Interseção entre os vocabulários
            counts = []
            for i in xrange(0,len(inter),1): #Obtém a frequência geral de cada termo
                val = voc1.get(inter[i]).count + voc2.get(inter[i]).count
                counts.append(val)
            words=[x for _,x in sorted(zip(counts,inter))]
            if (len(words) > numberTerms):
                words = words[int(len(words)/2 - (numberTerms/2)):int(len(words)/2 + (numberTerms/2))] #Retira termos muito e pouco frequentes
                
            score = 0
            for i in xrange(0,len(words),1): #Calcula a distância entre os livros de acordo com a similaridade entre os mesmos pares de termos
                for j in xrange(i+1,len(words),1):
                    simd1 = m1.wv.similarity(words[i],words[j])
                    simd2 = m2.wv.similarity(words[i],words[j])
                    score += (simd1 - simd2)**2
            score = np.sqrt(score)
            finalmat[model1][model2] = score
            
    np.save(("../distances/" + str(numberTerms)),finalmat) #Salva a matriz final


