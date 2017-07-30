from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import sklearn.linear_model as lm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#Imports pickle to unload the model
import pickle
#Imports sys and sklearn
import sys
import sklearn
#Imports argparse
import argparse
# Loads pandas
import pandas as pd
# Loads numpy
import numpy as np
dataframe = pandas.read_csv("EURUSDmulticlasscsv.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
#X = dataset[:,0:4050].astype(float)
X = dataset[:,0:34]
y = dataset[:,34]
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
#lr = LogisticRegression()
#sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
#                          meta_classifier=lr)
#print('3-fold cross validation:\n')

#for clf, label in zip([clf1, clf2, clf3, sclf], 
 #                     ['KNN', 
  #                     'Random Forest', 
   #                    'Naive Bayes',
    #                   'StackingClassifier']):

    #scores = model_selection.cross_val_score(clf, X, y, 
     #                                         cv=3, scoring='accuracy')
    #print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
     #     % (scores.mean(), scores.std(), label))
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, eclf], labels):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

#save model to disk
filename = 'model.sav'
pickle.dump(eclf, open(filename, 'wb'))
print "all done Teerth"
