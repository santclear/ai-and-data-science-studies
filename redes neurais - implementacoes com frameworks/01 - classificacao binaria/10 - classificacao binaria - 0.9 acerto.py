# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:51:56 2021

@author: santc
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('../datasets/entradas_breast.csv')
classe = pd.read_csv('../datasets/saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    

    classificador.add(Dense(units = 16, activation = 'softplus', kernel_initializer = 'normal', input_dim = 30))
    classificador.add(Dropout(0.2))
    

    classificador.add(Dense(units = 16, activation = 'softplus', kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    

    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 1.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 100, batch_size = 32)

resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvioPadrao = resultados.std()