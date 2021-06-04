# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:35:05 2021

@author: santc
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

base = pd.read_csv('poluicao.csv')

base = base.dropna()

base = base.drop('No', axis = 1)
base = base.drop('year', axis = 1)
base = base.drop('month', axis = 1)
base = base.drop('day', axis = 1)
base = base.drop('hour', axis = 1)
base = base.drop('cbwd', axis = 1)

baseTreinamento = base.iloc[:, 1:7].values

poluicao = base.iloc[:, 0].values

normalizador = MinMaxScaler(feature_range = (0, 1))
baseTreinamentoNormalizada = normalizador.fit_transform(baseTreinamento)

poluicao = poluicao.reshape(-1, 1)
poluicaoNormalizado = normalizador.fit_transform(poluicao)

previsores = []
poluicaoReal = []

periodoAmostral = 10
qtdRegistrosPoluicao = len(poluicao)
for i in range(periodoAmostral, qtdRegistrosPoluicao):
    previsores.append(baseTreinamentoNormalizada[i-periodoAmostral:i, 0:6])
    poluicaoReal.append(poluicaoNormalizado[i, 0])
	
previsores, poluicaoReal = np.array(previsores), np.array(poluicaoReal)


regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])

interrompeComMelhoresResultados = EarlyStopping(monitor = 'loss', min_delta=1e-10, patience=10, verbose=1)
reduzTaxaAprendizagemSeParouDeMelhorar = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
salvaMelhoresPesos = ModelCheckpoint(filepath='pesos_poluicao_china.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, poluicaoReal, epochs = 100, batch_size = 64, 
			  callbacks=(
				  interrompeComMelhoresResultados,
				  reduzTaxaAprendizagemSeParouDeMelhorar,
				  salvaMelhoresPesos)
			  )


previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
poluicao.mean()

plt.plot(poluicao, color = 'red', label = 'Poluição na China')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão de poluição na China')
plt.xlabel('Horas')
plt.ylabel('Valor poluição')
plt.legend()
plt.show()