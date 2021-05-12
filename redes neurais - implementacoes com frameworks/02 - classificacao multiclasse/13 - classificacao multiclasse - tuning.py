# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:28:44 2021

@author: santc
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV

# http://archive.ics.uci.edu/ml/index.php
base = pd.read_csv('iris.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:, 4]
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criarRede(dropout, optimizer, kernel_initializer, activation, neurons):
    classificador = Sequential()

    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 4))
    classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = 3, activation = 'softmax'))
    

    classificador.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede)

parametros = {'batch_size': [10, 30, 60],
              'epochs': [100, 500, 1000],
              'dropout': [0.2, 0.25, 0.3],
              'optimizer': ['adam', 'adamax', 'adadelta'],
              'kernel_initializer': ['random_normal', 'normal', 'random_uniform', 'uniform'],
              'activation': ['relu', 'softplus'],
              'neurons': [4, 8, 16]}


grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

"""
Melhores parâmetros
{'activation': 'softplus', 'batch_size': 10, 'dropout': 0.2, 'epochs': 500, 'kernel_initializer': 'normal', 'neurons': 16, 'optimizer': 'adam'}
"""