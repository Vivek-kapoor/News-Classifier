import nltk
import sklearn
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB,MultinomialNB  
from sklearn import preprocessing
from collections import Counter
import csv
import numpy as np
import math
import re
import codecs
import matplotlib.pyplot as plt
filename = "final_training.csv"
filename1 = "final_testing.csv"
fp = codecs.open(filename , "r", encoding ="utf-8",errors = "ignore")
fp1 = codecs.open(filename1 , "r", encoding = "utf-8" , errors = "ignore")

reader1 = csv.reader(fp , delimiter = ',', quoting = csv.QUOTE_NONE)
reader2 = csv.reader(fp1 , delimiter= ',', quoting = csv.QUOTE_NONE)
next(reader1)
next(reader2)
row1,row2 = zip(*reader1)
test_row1,test_row2 = zip(*reader2)
print(row1[1])
docs_b=[]
docs_t=[]
docs_e=[]
docs_m=[]
for i in range(1,60000):
	if(row2[i]=='b'):
		docs_b.append(row1[i])
	elif(row2[i]=='t'):
		docs_t.append(row1[i])
	elif(row2[i]=='e'):
		docs_e.append(row1[i])
	else:
		docs_m.append(row1[i])
bigrams_b=[]
bigrams_t=[]
bigrams_e=[]
bigrams_m=[]
unigrams_b=[]
unigrams_t=[]
unigrams_e=[]
unigrams_m=[]

for i in range(1,len(docs_b)):
	tokens= nltk.word_tokenize(docs_b[i])
	bgs= list(nltk.bigrams(tokens))
	for j in bgs:
		bigrams_b.append(j)
	for k in tokens:
		unigrams_b.append(k)
for i in range(1,len(docs_t)):
	tokens= nltk.word_tokenize(docs_t[i])
	bgs= list(nltk.bigrams(tokens))
	for j in bgs:
		bigrams_t.append(j)
	for k in tokens:
		unigrams_t.append(k)
for i in range(1,len(docs_e)):
	tokens= nltk.word_tokenize(docs_e[i])
	bgs= list(nltk.bigrams(tokens))
	for j in bgs:
		bigrams_e.append(j)
	for k in tokens:
		unigrams_e.append(k)
for i in range(1,len(docs_m)):
	tokens= nltk.word_tokenize(docs_m[i])
	bgs= list(nltk.bigrams(tokens))
	for j in bgs:
		bigrams_m.append(j)
	for k in tokens:
		unigrams_m.append(k)

bigrams_count_b = []
unigrams_count_b = []
for c in set(bigrams_b):
    count = bigrams_b.count(c)
    bigrams_count_b.append((c,count))

for c in set(unigrams_b):
    count = unigrams_b.count(c)
    unigrams_count_b.append((c,count))

bigrams_count_t = []
unigrams_count_t = []
for c in set(bigrams_t):
    count = bigrams_t.count(c)
    bigrams_count_t.append((c,count))

for c in set(unigrams_t):
    count = unigrams_t.count(c)
    unigrams_count_t.append((c,count))

bigrams_count_e = []
unigrams_count_e = []
for c in set(bigrams_e):
    count = bigrams_e.count(c)
    bigrams_count_e.append((c,count))

for c in set(unigrams_e):
    count = unigrams_e.count(c)
    unigrams_count_e.append((c,count))

bigrams_count_m = []
unigrams_count_m = []
for c in set(bigrams_m):
    count = bigrams_m.count(c)
    bigrams_count_m.append((c,count))

for c in set(unigrams_m):
    count = unigrams_m.count(c)
    unigrams_count_m.append((c,count))


doc= test_row1[5]
print(doc)
doc_tokens = nltk.word_tokenize(doc)
doc_bgs = list(nltk.bigrams(doc_tokens)) 
print(doc_bgs)
p_doc_bgs=0
for i in doc_bgs:
	for j in bigrams_count_b:
		if i==j[0]:
			count=j[1]+1
			break
		else:
			count=1
	print(count)
	for j in unigrams_count_b:
		if i[0]==j[0]:
			count_deno=j[1]+ len(unigrams_b)
			break

		else:
			count_deno=len(unigrams_b)
	print(count_deno)
	print(count/count_deno)
	print('----------------')
	p_doc_bgs+=math.log(count/count_deno)
print(p_doc_bgs)
print('----------')
p_doc_bgs=0
for i in doc_bgs:
	for j in bigrams_count_e:
		if i==j[0]:
			count=j[1]+1
			break
		else:
			count=1
	print(count)
	for j in unigrams_count_e:
		if i[0]==j[0]:
			count_deno=j[1]+ len(unigrams_e)
			break

		else:
			count_deno=len(unigrams_e)
	print(count_deno)
	print(count/count_deno)
	print('----------------')
	p_doc_bgs+=math.log(count/count_deno)
print(p_doc_bgs)
print('----------')

p_doc_bgs=0
for i in doc_bgs:
	for j in bigrams_count_t:
		if i==j[0]:
			count=j[1]+1
			break
		else:
			count=1
	print(count)
	for j in unigrams_count_t:
		if i[0]==j[0]:
			count_deno=j[1]+ len(unigrams_t)
			break

		else:
			count_deno=len(unigrams_t)
	print(count_deno)
	print(count/count_deno)
	print('----------------')
	p_doc_bgs+=math.log(count/count_deno)
print(p_doc_bgs)
print('----------')
p_doc_bgs=0
for i in doc_bgs:
	for j in bigrams_count_m:
		if i==j[0]:
			count=j[1]+1
			break
		else:
			count=1
	print(count)
	for j in unigrams_count_m:
		if i[0]==j[0]:
			count_deno=j[1]+ len(unigrams_m)
			break

		else:
			count_deno=len(unigrams_m)
	print(count_deno)
	print(count/count_deno)
	print('----------------')
	p_doc_bgs+=math.log(count/count_deno)
print(p_doc_bgs)
print('----------')