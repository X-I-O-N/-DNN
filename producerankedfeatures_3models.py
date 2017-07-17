#Trimmed to only produce ranked features using 3 models only
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import pickle
import sklearn
from sklearn import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"  
import numpy
import numpy as np
import pandas
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from gplearn.genetic import SymbolicRegressor
from mlxtend.classifier import StackingClassifier
from mlxtend.regressor import StackingRegressor
import sklearn.linear_model as lm
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
#from xgboost import plot_importance
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("L-all-blip-else-trainingset.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
#X = dataset[:,0:4050].astype(float)
X = dataset[:,0:1288]
y = dataset[:,1288]

# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(Y)
#y = encoder.transform(Y)

#forest = ExtraTreesClassifier(n_estimators=250,
                             # random_state=0)
#xgb = XGBClassifier()
#xgb.fit(X, y)
#plot_importance(xgb)
#DTC = sklearn.tree.DecisionTreeClassifier()
#DTC.fit(X, y)



#forest.fit(X, y)
#print cross_val_score(forest, X, y, cv=5)

#importances1 = forest.feature_importances_
#importances2= xgb.feature_importances_
#importances3 = DTC.feature_importances_

#importances = ((importances1 + importances2 + importances3)/3)
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],
           #  axis=0)
#indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")

#for f in range(X.shape[1]):
    #print("%d. %d %f" % (f + 1, (indices[f])+1, importances[indices[f]]))

classifier = XGBClassifier(
    objective='binary:logistic', 
    n_estimators=200, 
    learning_rate=0.08, 
    max_depth=5, 
    nthread=4,
    subsample=0.9,
#    colsample_bytree=0.8,
    reg_lambda=6,
    reg_alpha=5,
    seed=1301,
    silent=True
)
selector = RFECV(estimator=classifier, step=3, 
   cv=5)
selector.fit(X, y)

print('The optimal number of features is {}'.format(selector.n_features_))
features = [f for f,s in zip(X_train.columns, selector.support_) if s]
print('The selected features are:')
print ('{}'.format(features))

#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (roc auc)")
#plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
#plt.savefig('feature_auc_nselected.png', bbox_inches='tight', pad_inches=1)

# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
 #      color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), indices)
#plt.xlim([-1, X.shape[1]])
#plt.show()
