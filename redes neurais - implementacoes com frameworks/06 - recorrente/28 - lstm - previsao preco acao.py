# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:06:27 2021

@author: santc
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

base = pd.read_csv('petr4_treinamento.csv')

# Exclui os valores na
base = base.dropna()

# Todas as linhas ":" e a coluna 1 "1:2" (Open - Abertura)
baseTreinamento = base.iloc[:, 1:2].values

# Converte os preços de abertura para a escala de valores entre 0 e 1, isso
# dimui o custo de processamento
normalizador = MinMaxScaler(feature_range=(0,1))
baseTreinamentoNormalizada = normalizador.fit_transform(baseTreinamento)

previsores = []
precoReal = []

# 1242 qtdRegistrosTreinamento
qtdRegistrosTreinamento = len(baseTreinamento)
# 90 Períodos
for i in range(90, qtdRegistrosTreinamento):
    # A cada iteração separa 90 amostras de preços de abertura
    # Por exemplo, se a série tivesse somente 10 preços de abertura (3,5,1,1,7,8,2,9,3,4) 
    # e a amostra fosse 2 a variável amostraPeiodos ficaria assim:
    # 1º iteração -> amostraPeriodos = [3,5]
    # 2ª iteração -> amostraPeriodos = [5,1]
    # 3ª iteração -> amostraPeriodos = [1,1]
    # 4ª iteração -> amostraPeriodos = [1,7]
    # ...
    amostraPeriodos = baseTreinamentoNormalizada[i-90:i,0]
    previsores.append(amostraPeriodos)
    
    # Para cada amostra, um preço de abertura para treinamento será usado
    # Tomando o exemplo anterior (3,5,1,1,7,8,2,9,3,4), a cada interação ficará assim:
    # 1º iteração [3,5] -> precoNormalizado = 1
    # 2ª iteração [5,1] -> precoNormalizado = 1
    # 3ª iteração [1,1] -> precoNormalizado = 7
    # 4ª iteração [1,7] -> precoNormalizado = 8
    # ...
    precoNormalizado = baseTreinamentoNormalizada[i, 0]
    precoReal.append(precoNormalizado)

previsores, precoReal = np.array(previsores), np.array(precoReal)

# batch_size
# 1152 registros
qtdRegistros = previsores.shape[0]
# timesteps
# 90 intervalos
intervalosTempo = previsores.shape[1]
# 1 atributo previsor -> Preço de abertura Open
# input_dim
qtdAtributosPrevisores = 1
previsores = np.reshape(previsores, (qtdRegistros, intervalosTempo, qtdAtributosPrevisores))

regressor = Sequential()

# units: 100 células de memória, para dimensionalidade, captura tendência no decorrer
# do tempo, captura a variação temporal
# return_sequences: indica que a informação será encaminhada a diante para a camadas,
# subsequentes, obrigatório quando há mais de 1 camada LSTM
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(intervalosTempo, 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

# Retirado o parâmetro return_sequences, pois não há mais camadas LSTM subsequentes
regressor.add(LSTM(units=50, input_shape=(intervalosTempo, 1)))
regressor.add(Dropout(0.3))

# Como os dados foram normalizados para retornar valores entre 0 e 1, a função
# sigmoid também pode ser usada
regressor.add(Dense(units=1, activation='linear'))

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, precoReal, epochs=100, batch_size=32)