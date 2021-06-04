# -*- coding: utf-8 -*-
"""
Created on Sun May 23 02:43:39 2021

@author: santc
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento_ex.csv')
base = base.dropna()
baseTreinamento = base.iloc[:, 1:2].values

normalizador = MinMaxScaler(feature_range=(0,1))
baseTreinamentoNormalizada = normalizador.fit_transform(baseTreinamento)

previsores = []
precoReal = []

qtdRegistrosTreinamento = len(baseTreinamento)

for i in range(90, qtdRegistrosTreinamento):
    amostraPeriodos = baseTreinamentoNormalizada[i-90:i,0]
    previsores.append(amostraPeriodos)
    
    precoNormalizado = baseTreinamentoNormalizada[i, 0]
    precoReal.append(precoNormalizado)

previsores, precoReal = np.array(previsores), np.array(precoReal)


qtdRegistros = previsores.shape[0]
intervalosTempo = previsores.shape[1]
qtdAtributosPrevisores = 1
previsores = np.reshape(previsores, (qtdRegistros, intervalosTempo, qtdAtributosPrevisores))

regressor = Sequential()

regressor.add(LSTM(units=100, return_sequences=True, input_shape=(intervalosTempo, 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))


regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])
regressor.fit(previsores, precoReal, epochs=100, batch_size=32)

baseTeste = pd.read_csv('petr4_teste_ex.csv')
precoRealTeste = baseTeste.iloc[:, 1:2].values
baseCompleta = pd.concat((base['Open'], baseTeste['Open']), axis = 0)

entradas = baseCompleta[len(baseCompleta) - len(baseTeste) - intervalosTempo:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

XTeste = []
for i in range(intervalosTempo, len(entradas)):
    XTeste.append(entradas[i-intervalosTempo:i, 0])
	
	
XTeste = np.array(XTeste)
XTeste = np.reshape(XTeste, (XTeste.shape[0], XTeste.shape[1], 1))
previsoes = regressor.predict(XTeste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
precoRealTeste.mean()
    
plt.plot(precoRealTeste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço da ação PETR4')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()