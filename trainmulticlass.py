from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
# Loads pandas
import pandas
# Loads numpy
import numpy as np
from xgboost import XGBClassifier
dataframe = pandas.read_csv("USDJPY,5multiclass.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
#X = dataset[:,0:4050].astype(float)
X = dataset[:,0:59]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y = dataset[:,59]
clf0 = sklearn.ensemble.GradientBoostingClassifier(n_estimators=200)
clf1 = lm.LogisticRegression(penalty = "l1", C = 9081)
clf2 = RandomForestClassifier(random_state=1, n_estimators=200)
clf3 = lm.LogisticRegression(penalty = "l2", C = 5000)
clf4 = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
eclf = EnsembleVoteClassifier(clfs=[clf0, clf1, clf2, clf3, clf4], weights=[1,1,1,3,1])

labels = ['GBC','Lasso', 'Random Forest', 'Ridge', 'MLP','Ensemble']
for clf, label in zip([clf0, clf1, clf2, clf3, clf4, eclf], labels):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

eclf.fit(X,y)
#sclf.fit(X,y)
#save model to disk
filename = 'modelmulti.sav'
pickle.dump(eclf, open(filename, 'wb'))
print "all done Teerth"
