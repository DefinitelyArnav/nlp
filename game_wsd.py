import nltk
from nltk.corpus import wordnet
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import inaugural
from nltk.corpus import gutenberg
from nltk.corpus import names
from nltk.util import ngrams
from itertools import product
import numpy as np


# def get_z():

# On the train, he listened to a music track
word_list = ["train", "listen", "music", "track"]

# I need money to pay doctor's bills
# word_list = ["need","money","pay","doctor","bill"]

# There is a financial institution near the river bank
# word_list = ["financial", "institution", "river", "bank"]

n = 15

# Creating w matrix
w = [[0.0 for x in range(len(word_list))] for y in range(len(word_list))]
for word1, word2 in product(word_list, word_list):
	count = 0
	
	n_grams = ngrams(brown.words(), n)
	
	for grams in n_grams:
		if word1 in grams and word2 in grams:
			count += 1

	n_grams = ngrams(treebank.words(), n)
	
	for grams in n_grams:
		if word1 in grams and word2 in grams:
			count += 1
	
	n_grams = ngrams(inaugural.words(), n)
	
	for grams in n_grams:
		if word1 in grams and word2 in grams:
			count += 1
	
	n_grams = ngrams(names.words(), n)
	
	for grams in n_grams:
		if word1 in grams and word2 in grams:
			count += 1

	n_grams = ngrams(gutenberg.words(), n)
	
	for grams in n_grams:
		if word1 in grams and word2 in grams:
			count += 1

	w[word_list.index(word1)][word_list.index(word2)] = count   

for i in range(len(word_list)):
	w[i][i] = 1


print(w)
# Calculating mean of w
mean = 0.0
for i in range(len(word_list)):
	for j in range(len(word_list)):
		mean += w[i][j]
mean = mean / ( len(word_list)*len(word_list) )

# Adding co-occurence bias
for i in range(len(word_list)-1):
	w[i][i+1] += mean
	w[i+1][i] += mean
		

print(w)

# Creating list of senses of each word in word_list
senses = [0 for i in range(len(word_list))]
for i in range(len(word_list)):
	senses[i] = wordnet.synsets(word_list[i])

# Creating probability list of each word
x = []
for word in word_list:
	x.append(np.array( [ 1.0/len(wordnet.synsets(word)) for i in range(len(wordnet.synsets(word))) ] ))

y = x

for _ in range(20):
	for word1 in word_list:
		m = len(wordnet.synsets(word1))
		u1 = np.array([0.0 for i in range(m)]) # u(eh,x)
		u2 = 0.0 # u(x,x)
		i = word_list.index(word1)

		for word2 in word_list:
			# Creating the Z matrix
			n = len(wordnet.synsets(word2))
			Z = np.array( [[0.0 for i in range(n)] for j in range(m)] )

			j = word_list.index(word2)

			for si in senses[i]:
				for sj in senses[j]:
					sim = wordnet.wup_similarity(si,sj)
					if sim is None:
						sim = 0.0
					Z[senses[i].index(si)][senses[j].index(sj)] = sim

			u1 += w[i][j] * Z.dot(x[j])
			u2 += w[i][j] * Z.dot(x[j]).dot(x[i])

		y[i] = np.multiply(x[i], u1/u2)
	x = y

	if _ == 0:
		for word in word_list:
			i = word_list.index(word)
			print(x[i], word_list[i])	


for word in word_list:
	i = word_list.index(word)
	print(x[i], word_list[i])	




# for word1, word2 in product(word_list, word_list):
# 	syns1 = wordnet.synsets(word1)
# 	syns2 = wordnet.synsets(word2)
# 	d = 0
# 	for sense1, sense2 in product(syns1,syns2):
# 		# x = wordnet.wup_similarity(sense1,sense2)
# 		x = sense1.jcn_similarity(sense2)
# 		if x is None:
# 			x = 0
# 		d += x
#
# 	w[word_list.index(word1)][word_list.index(word2)] = d
#
# for i in w:
# 	print(i)