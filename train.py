import sklearn.linear_model as lm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

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

dataframe = pandas.read_csv("L-all-blip-else-trainingset.csv", header=None)

dataset = dataframe.values

# split into input (X) and output (Y) variables

#X = dataset[:,0:4050].astype(float)

X = dataset[:,0:34]

y = dataset[:,34]

model_ridge = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=9081)
model_randomforest = RandomForestClassifier(n_estimators = 200)
model_lasso = lm.LogisticRegression(penalty = "l1", C = 9081)
model_gbt = GradientBoostingClassifier(n_estimators = 200)

pred_ridge = []
pred_randomforest = []
pred_lasso = []
pred_gbt = []
new_Y = []
for i in range(10):
    indxs = np.arange(i, X.shape[0], 10)
    indxs_to_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], 10)))
    pred_ridge = pred_ridge + list(model_ridge.fit(X[indxs_to_fit[:]], y[indxs_to_fit[:]]).predict_proba(X[indxs,:])[:,1])
    pred_randomforest = pred_randomforest + list(model_randomforest.fit(X[indxs_to_fit[:]], y[indxs_to_fit[:]]).predict_proba(X[indxs,:])[:,1])
    pred_lasso = pred_lasso + list(model_lasso.fit(X[indxs_to_fit[:]], y[indxs_to_fit[:]]).predict_proba(X[indxs,:])[:,1])
    pred_gbt = pred_gbt + list(model_gbt.fit(X[indxs_to_fit[:]], y[indxs_to_fit[:]]).predict_proba(X[indxs,:])[:,1])
    new_Y = new_Y + list(y[indxs[:]])
	
                                                                   
new_X = np.hstack((np.array(pred_ridge).reshape(len(pred_ridge), 1), np.array(pred_randomforest).reshape(len(pred_randomforest), 1), np.array(pred_lasso).reshape(len(pred_lasso), 1), np.array(pred_gbt).reshape(len(pred_gbt), 1)))
new_Y = np.array(new_Y).reshape(len(new_Y), 1)

model_stacker = lm.LogisticRegression()
print np.mean(cross_val_score(model_stacker, new_X, new_Y.reshape(new_Y.shape[0]), cv=5))

model_stacker.fit(new_X, new_Y.reshape(new_Y.shape[0]))
#save model to disk
filename = 'model.sav'
pickle.dump(model_stacker, open(filename, 'wb'))
print "all done Teerth"
