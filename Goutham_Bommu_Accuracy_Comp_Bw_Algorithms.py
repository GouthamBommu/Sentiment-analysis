# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:07:56 2019

@author: Goutham
"""

import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


data = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t', quoting=3)
nltk.download('stopwords')

cleanedData = []
for i in range(0, len(data.index)):
    sentences = re.sub('[^a-zA-Z]', ' ', data.get_values()[i][0])
    sentences = sentences.lower()
    sentences = sentences.split()
    portStemmer = PorterStemmer()
    review = [portStemmer.stem(word) for word in sentences if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    cleanedData.append(review)

sparseMatrix = CountVectorizer(max_features=1500)
X = sparseMatrix.fit_transform(cleanedData).toarray()
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

try:
    f = open('classifier_nb.pickle', 'rb')
    naive_bayes_classifier = pickle.load(f)
    f.close()
    f = open('classifier_rf.pickle', 'rb')
    random_forest_classifier = pickle.load(f)
    f.close()

    print("Naive Bayes classifier loaded")
    print("Random Forest classifier loaded")

except IOError:
    print("Saving the classifier")
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)

    f = open('classifier_nb.pickle', 'wb')
    pickle.dump(naive_bayes_classifier, f)
    f.close()

    random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    random_forest_classifier.fit(X_train, y_train)

    f = open('classifier_rf.pickle', 'wb')
    pickle.dump(random_forest_classifier, f)
    f.close()

pred_nb = naive_bayes_classifier.predict(X_test)
pred_rf = random_forest_classifier.predict(X_test)

cm_nb = confusion_matrix(y_test, pred_nb)
cm_rf = confusion_matrix(y_test, pred_rf)

print("\nAccuracy percent for Naive Bayes classifier : ", accuracy_score(y_test,pred_nb)*100,"%")
print("\nAccuracy percent for Random Forest classifier : ", accuracy_score(y_test,pred_rf)*100,"%")