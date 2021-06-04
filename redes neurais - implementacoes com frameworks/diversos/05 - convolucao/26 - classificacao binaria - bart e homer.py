# -*- coding: utf-8 -*-
"""
Created on Thu May 20 23:05:58 2021

@author: santc
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import keras

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv('personagens.csv')


previsores = base.iloc[:,0:6].values

classe = base.iloc[:, 6]

labelencoder = LabelEncoder()

classe = labelencoder.fit_transform(classe)

(previsoresTreinamento, 
 previsoresTeste, 
 classeTreinamento, 
 classeTeste) = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

classificador.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 6))
classificador.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'random_uniform'))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])


classificador.fit(previsoresTreinamento, classeTreinamento, batch_size = 10, epochs = 100)

previsoes = classificador.predict(previsoresTeste)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classeTeste, previsoes)

matriz = confusion_matrix(classeTeste, previsoes)

resultado = classificador.evaluate(previsoresTeste, classeTeste)