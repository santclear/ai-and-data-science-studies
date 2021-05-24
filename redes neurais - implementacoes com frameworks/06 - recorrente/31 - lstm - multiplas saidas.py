# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:59:29 2021

@author: santc
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')

# Exclui os valores na
base = base.dropna()

# Todas as linhas ":" e a coluna 1 "1:2" (Open - Abertura)
baseTreinamentoAbertura = base.iloc[:, 1:2].values
# Todas as linhas ":" e a coluna 1 "1:2" (Máxima)
baseTreinamentoMaxima = base.iloc[:, 2:3].values

# Converte os preços de abertura para a escala de valores entre 0 e 1, isso
# dimui o custo de processamento
normalizador = MinMaxScaler(feature_range=(0,1))
baseTreinamentoAberturaNormalizada = normalizador.fit_transform(baseTreinamentoAbertura)
baseTreinamentoMaximaNormalizada = normalizador.fit_transform(baseTreinamentoMaxima)

previsores = []

precoRealAbertura = []
precoRealMaxima = []

# 1242 qtdRegistrosTreinamento
qtdRegistrosTreinamentoAbertura = len(baseTreinamentoAbertura)
# Quantidade de períodos que serão utilizados a cada iteração no treinamento e teste
# da rede neural
periodoAmostral = 90

for i in range(periodoAmostral, qtdRegistrosTreinamentoAbertura):
    # A cada iteração separa 90 amostras de preços de abertura
    # Por exemplo, se a série tivesse somente 10 preços de abertura (3,5,1,1,7,8,2,9,3,4) 
    # e a amostra fosse 2 a variável amostraPeiodos ficaria assim:
    # 1º iteração -> amostraPeriodos = [3,5]
    # 2ª iteração -> amostraPeriodos = [5,1]
    # 3ª iteração -> amostraPeriodos = [1,1]
    # 4ª iteração -> amostraPeriodos = [1,7]
    # ...
	# 0 é preço de de abertura (Open)
    amostraPeriodosAbertura = baseTreinamentoAbertura[i-periodoAmostral:i,0]
    previsores.append(amostraPeriodosAbertura)
    
    # Para cada amostra, um preço de abertura para treinamento será usado
    # Tomando o exemplo anterior (3,5,1,1,7,8,2,9,3,4), a cada interação ficará assim:
    # 1º iteração [3,5] -> precoNormalizado = 1
    # 2ª iteração [5,1] -> precoNormalizado = 1
    # 3ª iteração [1,1] -> precoNormalizado = 7
    # 4ª iteração [1,7] -> precoNormalizado = 8
    # ...
	# 0 é preço de de abertura (Open)
    precoAberturaNormalizada = baseTreinamentoAberturaNormalizada[i, 0]
    precoRealAbertura.append(precoAberturaNormalizada)
	
    precoMaximaNormalizada = baseTreinamentoMaximaNormalizada[i, 0]
    precoRealMaxima.append(precoMaximaNormalizada)
	
previsores, precoRealAbertura, precoRealMaxima = (
	np.array(previsores), 
	np.array(precoRealAbertura), 
	np.array(precoRealMaxima))

# batch_size
# 1152 registros
qtdRegistrosAbertura = previsores.shape[0]
# timesteps
# 90 intervalos
periodoAberturaAmostral = previsores.shape[1]
# 1 atributo previsor -> Preço de abertura Open
# input_dim
qtdAtributosPrevisores = 1
previsores = np.reshape(previsores, (qtdRegistrosAbertura, periodoAberturaAmostral, qtdAtributosPrevisores))

precosReais = np.column_stack((precoRealAbertura, precoRealMaxima))


regressor = Sequential()

regressor.add(LSTM(units=100, return_sequences=True, input_shape=(periodoAmostral, qtdAtributosPrevisores)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

# Como os dados foram normalizados para retornar valores entre 0 e 1, a função
# sigmoid também pode ser usada
regressor.add(Dense(units=2, activation='linear'))
# Para esse problema também poderia ser usado o optimizer adam
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])