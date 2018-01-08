######################################################
#   TP2 - Processamento de Linguagem Natural         #
#   Aluno: Murilo Vale Ferreira Menezes - 2013030996 #
#   Prof. Adriano Veloso                             #
######################################################

import os,sys
import nltk
import gensim
import numpy as np
import pandas as pd
from sklearn import svm,naive_bayes,preprocessing
import re
import unidecode

path = "./data" #	Diretório dos dados da coleção
os.chdir(path)
files = os.listdir(".")

#classes = ['sent_init','ART','ADJ','N','NPROP','NUM','PROADJ','PROSUB','PROPESS','PRO-KS','PRO-KS-REL','ADV-KS','ADV-KS-REL','KC','KS','PREP','IN','V','VAUX','PCP','PDEN','CUR']
#complementares = ['EST','AP','DAD','TEL','DAT','HOR']

# 	Mudar os arquivos tirando parts of speech
# 	para treinar o modelo Word2Vec

for file in files:
	f = open(file,'r')
	raw = f.read()
	f.close()
	raw = raw + ' '
	raw = re.sub(r'_.*? ',' ',raw)
	raw = re.sub(r'_.*?\n','\n',raw)
	raw = raw.decode('utf-8')
	raw = raw.lower()
	raw = unidecode.unidecode(raw)
	output_file_name = "NO_POS_" + file
	fw = open(output_file_name,'w')
	fw.write(raw)
	fw.close()

#	Treinamento do Word2Vec

print "Training Word2Vec...\n"
w2v_dims = 200

files = os.listdir('.')
files_nopos = [w for w in files if 'NO_POS' in w and 'train' in w]
print files_nopos[0]
for file in files_nopos:
	print "Training Word2Vec in " + file
	sentences = gensim.models.word2vec.LineSentence(file)
	model = gensim.models.Word2Vec(sentences,size=w2v_dims,window=8,workers=4,sg=1,min_count=0)
	model.save('./W2VModel')

print "Done!\n"



#	Estruturando os dados

possible_tags = []
file_train = [w for w in files if 'NO_POS' not in w and 'train' in w][0]
ft = open(file_train,'r')
line = ft.readline()
dataTrain = []
labelsTrain = []
model = gensim.models.Word2Vec.load('./W2VModel')
print "Structuring dataset...\n"

#	Treino
while line:
	line = line.lower()
	linesplit = line.split(' ')
	n_word = 0
	last_tag = 'sent_init'
	for w in linesplit:
		last_tag = last_tag.replace("\n","")
		if (last_tag not in possible_tags):
			possible_tags.append(last_tag)
		num_lasttag = possible_tags.index(last_tag)
		charac = w.split('_')
		word = charac[0]
		tag = charac[1]
		tag = tag.replace("\n","")
		if (word == ','):
			word = 'virgula'
		if word in model.wv.vocab:
			sample = model.wv[word].tolist()
		else:
			sample = [0] * w2v_dims
		lasttg = [0]*27
		lasttg[num_lasttag] = 1
		sample += lasttg
		if (tag not in possible_tags):
			possible_tags.append(tag)
		
		labelsTrain.append(possible_tags.index(tag))
		n_word += 1
		last_tag = tag
		dataTrain.append(sample)
	line = ft.readline()

#	Teste

file_test = [w for w in files if 'NO_POS' not in w and 'test' in w][0]
ft = open(file_test,'r')
line = ft.readline()
dataTest = []
labelsTest = []
model = gensim.models.Word2Vec.load('./W2VModel')
while line:
	line = line.lower()
	linesplit = line.split(' ')
	n_word = 0
	last_tag = 'sent_init'
	for w in linesplit:
		last_tag = last_tag.replace("\n","")
		if (last_tag not in possible_tags):
			possible_tags.append(last_tag)
		num_lasttag = possible_tags.index(last_tag)
		charac = w.split('_')
		word = charac[0]
		tag = charac[1]
		tag = tag.replace("\n","")
		if (word == ','):
			word = 'virgula'
		if word in model.wv.vocab:
			sample = model.wv[word].tolist()
		else:
			sample = [0] * w2v_dims
		lasttg = [0]*27
		lasttg[num_lasttag] = 1
		sample += lasttg
		if (tag not in possible_tags):
			possible_tags.append(tag)

		labelsTest.append(possible_tags.index(tag))
		n_word += 1
		last_tag = tag
		dataTest.append(sample)
	line = ft.readline()


#	Validação

file_dev = [w for w in files if 'NO_POS' not in w and 'dev' in w][0]
ft = open(file_dev,'r')
line = ft.readline()
dataDev = []
labelsDev = []
model = gensim.models.Word2Vec.load('./W2VModel')
while line:
	line = line.lower()
	linesplit = line.split(' ')
	n_word = 0
	last_tag = 'sent_init'
	for w in linesplit:
		last_tag = last_tag.replace("\n","")
		if (last_tag not in possible_tags):
			possible_tags.append(last_tag)
		num_lasttag = possible_tags.index(last_tag)
		charac = w.split('_')
		word = charac[0]
		tag = charac[1]
		tag = tag.replace("\n","")
		if (word == ','):
			word = 'virgula'
		if word in model.wv.vocab:
			sample = model.wv[word].tolist()
		else:
			sample = [0] * w2v_dims
		lasttg = [0]*27
		lasttg[num_lasttag] = 1
		sample += lasttg
		if (tag not in possible_tags):
			possible_tags.append(tag)

		labelsDev.append(possible_tags.index(tag))
		dataDev.append(sample)
		n_word += 1
		last_tag = tag
	line = ft.readline()
print "Done!\n"

# Subamostragem aleatória no conjunto de treinamento

nTrain = 200000

ici = np.random.choice(range(0,len(labelsTrain),1),size=nTrain)
dataTrain = [dataTrain[i] for i in ici]
labelsTrain = [labelsTrain[i] for i in ici]

#	SVM com Kernel Linear

Clist = range(-5,5,1)
Clist = [2**x for x in Clist]

print "Validating to choose C parameter...\n"

AccValues = []
for c in Clist:
	print "C = " + str(c)
	model = svm.LinearSVC(C=c)
	model.fit(dataTrain,labelsTrain)
	pred = model.predict(dataDev).tolist()
	nacc = 0
	for i in xrange(0,len(pred),1):
		if (pred[i] == labelsDev[i]):
			nacc += 1
	acc = float(nacc)/float(len(pred))
	print "Accuracy: " + str(acc)
	AccValues.append(acc)

index = AccValues.index(max(AccValues))
c = Clist[index]

print "Optimal C chosen: " + str(c) + " With validation accuracy " + str(max(AccValues)) + "\n"

model = svm.LinearSVC(C=c)
model.fit(dataTrain,labelsTrain)
pred_svm = model.predict(dataTest).tolist()
nacc = 0
for i in xrange(0,len(pred),1):
	if (pred[i] == labelsTest[i]):
		nacc += 1
acc = float(nacc)/float(len(pred))

print "Accuracy of SVM: " + str(acc) + "\n"

#	Naive Bayes Gaussiano

nb = naive_bayes.GaussianNB()
model = nb.fit(dataTrain,labelsTrain)
pred_nb = model.predict(dataTest).tolist()
nacc = 0
for i in xrange(0,len(pred),1):
	if (pred[i] == labelsTest[i]):
		nacc += 1
acc = float(nacc)/float(len(pred))

print "Accuracy of Naive Bayes: " + str(acc) + "\n"

#	Escreve os resultados em um arquivo .csv

f = open("./predictions.csv",w)
f.write("real_value,pred_svm,pred_nb\n")
for i in xrange(0,len(labelsTest),1):
	f.write(str(labelsTest[i]) + "," + str(pred_svm[i]) + "," + str(pred_nb[i]) + "\n")


