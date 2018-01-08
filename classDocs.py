# A simple Twitter bot that filters real time tweets with
# a given hashtag, classifies it using TF-IDF weights
# and retweets with the predicted tag.

import os
import csv
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#import gensim
#from gensim.models.doc2vec import LabeledSentence
import re
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import unidecode,collections
#import DocIterator as DocIt
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = 'access_token'
ACCESS_SECRET = 'access_secret'
CONSUMER_KEY = 'consumer_key'
CONSUMER_SECRET = 'consumer_secret'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

twit = Twitter(auth=oauth)

def select_word(index, word, labels):
	if word not in index.keys():
		return False
	counts = [0]*len(labels)
	files_word = index[word]
	if len(files_word) < 5:
		return False
	for f in files_word:
		a = f.split('/')[0]
		counts[labels.index(a)] += 1
	if (float(max(counts))/float(sum(counts))) > .5:
		return True
	return False


index = {}

labels = [f for f in os.listdir('.') if not (f.find('.py') >= 0) and not (f.find('.txt') >= 0) and not (f.find('saved') >= 0)]
datalist = []
labelslist = []
filenames = []
for label in labels:
	files = os.listdir('./' + label)
	for file in files:
		datalist.append(open('./' + label + '/' + file, 'r'))
		labelslist.append(label)# + "/" + file)
		filenames.append(label+"/"+file)

#print labelslist

stop_words = open('stopwords.txt','r').read()
stop_words = unidecode.unidecode(stop_words.decode('utf-8','ignore'))
stop_words = stop_words.split('\n')
#print stop_words
#vectorizer = TfidfVectorizer(stop_words=stop_words)


#Xtest =
#build inverted index
print "Building inverted index..."
rawlist = []
for k in xrange(len(datalist)):
	file = datalist[k]
	filename = filenames[k]
	raw = file.read().lower()
	raw = re.sub("[\\.?!;,]","",raw)
	raw = raw.replace("\n","")
	raw = unidecode.unidecode(raw.decode('utf-8','ignore'))
	raw = raw.lower()
	words = raw.split()
	counts = collections.Counter(words)
	for i in xrange(len(counts.keys())):
		key = counts.keys()[i]
		if key.endswith('s'):
				key = key[:len(key)-1]

		if key not in stop_words:
			value = counts.values()[i]
			if key not in index:
				index[key] = {}
			index[key][filename] = value
	rawlist.append(raw)
print "Done!\n"

vocab = index.keys()

X = []
Y = []
print "Building training set..."
for i in xrange(len(filenames)):
	raw = rawlist[i]
	filename = filenames[i]
	label = labelslist[i]
	labelIndex = labels.index(label)
	sample = []
	words = raw.split()
	for w in xrange(len(words)):
		if words[w].endswith('s'):
			words[w] = words[w][:len(words[w])-1]
	counts = collections.Counter(words)
	for word in vocab:
		if select_word(index,word,labels):
			tf = 0
			if word in counts.keys():
				tf = index[word][filename]
			simple_count = tf
			tf = np.log(float(tf+1))
			idf = np.log(float(len(filenames))/float(len(index[word])+1))
			tf_idf = tf*idf
			#print "TF: " + str(tf) + " IDF: " + str(idf)
			sample.append(tf_idf)
	X.append(sample)
	Y.append(labelIndex)
print "Done!\n"
#
#print "Selecting variables with the chi squared measure..."
#
#print "Done!\n"
#
print "Fitting model..."
model = LinearSVC()#MultinomialNB(
model.fit(X,Y)
print "Done!\n"
with open('index.txt', 'w') as file:
     file.write(json.dumps(index))

#def reading(self):
#	self.whip = eval(open('index.txt','r').read())

print "System running..."

index = json.load(open("index.txt"))
model = joblib.load("model_saved")

twitter_stream = TwitterStream(auth=oauth)

# Get a sample of the public data following through Twitter
iterator = twitter_stream.statuses.filter(track='#Hashtag')
tweet_count = 0
#f = open("data_users.csv","w")
#f.write("id_solicitante,nome_solicitante,email_solicitante,categoria,upload_solicitante,adicional,endereco\n")
#with open("data_users.csv","wb") as csvfile:
#writer = csv.writer(csvfile)
for tweet in iterator:
	tweet_count += 1
	test = tweet["text"]#raw_input("Input the sentence to classify: ")#"tem um buraco enorme na frente da minha casa"
	test = re.sub("[\\.?!;,]","",test)
	test = test.replace("\n","")
	#test = unidecode.unidecode(test.decode('utf-8','ignore'))
	print json.dumps(tweet)
	test = test.replace("#Hashtag","")
	test = test.lower()
	words = test.split()
	for w in xrange(len(words)):
		if words[w].endswith('s'):
			words[w] = words[w][:len(words[w])-1]
	counts = collections.Counter(words)
	#Xtest = vectorizer.fit_transform(test)
	Xtest = []
	sample = []
	for i in xrange(len(vocab)):
	#	if select_word:#selected[i]:
		word = vocab[i]
		if select_word(index,word,labels):
			tf = 0
			if word in counts.keys():
	       			tf = counts[word]
	#		if tf > 0:
	#			print "tf: " + str(tf)
			tf = np.log(float(tf+1))
			idf = np.log(float(len(filenames))/float(len(index[word])+1))
			tf_idf = tf*idf
			sample.append(tf_idf)
	Xtest.append(sample)
	#coord = tweet["geo"]["coordinates"]
	#best_clf = grid_search_tune.best_estimator_
	#predictions = best_clf.predict(Xtest)
	#
	pred = model.predict(Xtest)
	#print "Class predicted: " + labels[pred[0]]
	newText = tweet["text"]
	newText = newText.replace("#Hashtag","")
#	f = open("data_users.csv","a")
#	f.write(str(tweet_count)+","+tweet["user"]["name"]+",@" + tweet["user"]["screen_name"]+","+ labels[pred[0]]+","+newText+",,\n")
#	f.close()
	results = twit.statuses.update(status = "Usuario " + tweet["user"]["name"] + " (@" + tweet["user"]["screen_name"] + ")" + " Tweetou:\n\t" + newText + " #" + labels[pred[0]])

#f.close()

#test = raw_input("Input the sentence to classify: ")

#it = DocIterator(rawlist,labelslist)
#
#model = gensim.models.Doc2Vec(size=20, window=10, min_count=0, workers=4,alpha=0.025, min_alpha=0.025)
#model.build_vocab(it)
#for epoch in xrange(100):
#model.train(it,total_examples=len(labelslist),epochs=100)
#model.alpha -= .002
#model.min_alpha -= .002
#model.train(it)
