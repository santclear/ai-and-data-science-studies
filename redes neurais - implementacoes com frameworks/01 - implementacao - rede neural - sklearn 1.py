# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:18:52 2021

@author: santc
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(
    verbose = True, 
    max_iter = 100000, 
    tol = 0.00001, 
    activation = 'logistic',
    learning_rate_init = 0.001)

redeNeural.fit(entradas, saidas)
redeNeural.predict([[5, 7.2, 5.1, 2.2]])