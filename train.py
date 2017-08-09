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
dataframe = pandas.read_csv("USDJPY,5.csv", header=None)
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
extra = sklearn.linear_model.SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5000, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)
clf5 = sklearn.ensemble.AdaBoostClassifier(base_estimator=clf0, n_estimators=500, learning_rate=0.0001, algorithm='SAMME.R', random_state=None)
clf6 = sklearn.ensemble.AdaBoostClassifier(base_estimator=clf1, n_estimators=500, learning_rate=0.0001, algorithm='SAMME.R', random_state=None)
clf7 = sklearn.ensemble.AdaBoostClassifier(base_estimator=clf2, n_estimators=500, learning_rate=0.0001, algorithm='SAMME.R', random_state=None)
clf8 = sklearn.ensemble.AdaBoostClassifier(base_estimator=clf3, n_estimators=500, learning_rate=0.0001, algorithm='SAMME.R', random_state=None)
clf9 = sklearn.ensemble.AdaBoostClassifier(base_estimator=extra, n_estimators=500, learning_rate=0.0001, algorithm='SAMME.R', random_state=None)
clf10 = sklearn.ensemble.BaggingClassifier(base_estimator=clf0, n_estimators=500, max_samples=10000, max_features=59, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=0)
clf11 = sklearn.ensemble.BaggingClassifier(base_estimator=clf1, n_estimators=500, max_samples=10000, max_features=59, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=0)
clf12 = sklearn.ensemble.BaggingClassifier(base_estimator=clf2, n_estimators=500, max_samples=10000, max_features=59, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=0)
clf13 = sklearn.ensemble.BaggingClassifier(base_estimator=clf3, n_estimators=500, max_samples=10000, max_features=59, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=0)
clf14 = sklearn.ensemble.BaggingClassifier(base_estimator=clf4, n_estimators=500, max_samples=10000, max_features=59, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1, random_state=None, verbose=0)
#lr = LogisticRegression()
#sclf = StackingClassifier(classifiers=[clf0, clf1, clf2, clf3, clf4], 
#                          meta_classifier=lr)
#print('3-fold cross validation:\n')

#for clf, label in zip([clf0, clf1, clf2, clf3, clf4, sclf], 
#                      ['XGB',
#                       'KNN', 
#                       'Random Forest', 
#                       'Naive Bayes',
#		       'MLP',
#                       'StackingClassifier']):

#    scores = model_selection.cross_val_score(clf, X, y, 
#                                              cv=3, scoring='accuracy')
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
#          % (scores.mean(), scores.std(), label))
eclf = EnsembleVoteClassifier(clfs=[clf0, clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13, clf14], weights=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

labels = ['GBC','Lasso', 'Random Forest', 'Ridge', 'MLP', 'AdaboostGBC', 'AdaboostLasso', 'AdaboostRandom_Forest', 'AdaboostRidge', 'AdaboostMLP', 'BaggedGBC', 'BaggedLasso', 'BaggedRandom_Forest', 'BaggedRidge', 'BaggedMLP','Ensemble']
for clf, label in zip([clf0, clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13, clf14, eclf], labels):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

eclf.fit(X,y)
#sclf.fit(X,y)
#save model to disk
filename = 'model.sav'
pickle.dump(eclf, open(filename, 'wb'))
print "all done Teerth"
