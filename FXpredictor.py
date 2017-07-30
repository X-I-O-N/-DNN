#Imports pickle to unload the model
import pickle

#Imports sys and sklearn
import sys
import sklearn

# Loads pandas
import pandas

# Loads numpy
import numpy as np


dataframe = pandas.read_csv("EURUSDmulticlasscsv.csv", header=None)

dataset = dataframe.values

#Reads the training data
x = dataset[:,0:34]

#loads the model and prints its accuracy
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
accuracy = loaded_model.score
print(accuracy)
model = loaded_model

#runs the prediction and prints out to terminal
pred = model.predict(x)
print (pred)
