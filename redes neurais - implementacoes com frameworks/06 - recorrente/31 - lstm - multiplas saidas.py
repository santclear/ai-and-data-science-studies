# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:59:29 2021

@author: santc
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

baseTreinamentoPrecosAbertura = base.iloc[:, 1:2].values
baseTreinamentoPrecosMaxima = base.iloc[:, 2:3].values

normalizador = MinMaxScaler(feature_range=(0,1))
baseTreinamentoPrecosAberturaNormalizada = normalizador.fit_transform(baseTreinamentoPrecosAbertura)
baseTreinamentoPrecosMaximaNormalizada = normalizador.fit_transform(baseTreinamentoPrecosMaxima)

previsores = []
precosReaisAbertura = []
precosReaisMaxima = []

periodoAmostral = 90
for i in range(periodoAmostral, 1242):
    previsores.append(baseTreinamentoPrecosAberturaNormalizada[i-periodoAmostral:i, 0])
    precosReaisAbertura.append(baseTreinamentoPrecosAberturaNormalizada[i, 0])
    precosReaisMaxima.append(baseTreinamentoPrecosMaximaNormalizada[i, 0])
	
previsores, precosReaisAbertura, precosReaisMaxima = np.array(previsores), np.array(precosReaisAbertura), np.array(precosReaisMaxima)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

precosReaisAberturaMaxima = np.column_stack((precosReaisAbertura, precosReaisMaxima))

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 2, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, precosReaisAberturaMaxima, epochs = 100, batch_size = 32)

base_teste = pd.read_csv('petr4_teste.csv')
precosReaisAberturaTeste = base_teste.iloc[:, 1:2].values
precosReaisMaximaTeste = base_teste.iloc[:, 2:3].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - periodoAmostral:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

XTeste = []
for i in range(periodoAmostral, 112):
    XTeste.append(entradas[i-periodoAmostral:i, 0])
XTeste = np.array(XTeste)
XTeste = np.reshape(XTeste, (XTeste.shape[0], XTeste.shape[1], 1))

previsoes = regressor.predict(XTeste)
previsoes = normalizador.inverse_transform(previsoes)

   
plt.plot(precosReaisAberturaTeste, color = 'red', label = 'Preços abertura real')
plt.plot(precosReaisMaximaTeste, color = 'black', label = 'Preços máxima real')

plt.plot(previsoes[:, 0], color = 'blue', label = 'Previsões abertura')
plt.plot(previsoes[:, 1], color = 'orange', label = 'Previsões máxima')

plt.title('Previsão de preços das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()