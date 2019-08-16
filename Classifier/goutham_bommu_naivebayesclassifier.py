# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:37:59 2019

@author: Goutham
"""


import pickle
import cell_review_sentiment

f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

#pass in an unseen document and see how it performs 
print("Output: \n0: Negative comment \n1: Positive comment ")

while True:
	inputComment = input("Type your comment on your latest bought cell phone: ")
	print("Sentiment Result: " + classifier.classify(cell_review_sentiment.document_features({'doc':inputComment})))