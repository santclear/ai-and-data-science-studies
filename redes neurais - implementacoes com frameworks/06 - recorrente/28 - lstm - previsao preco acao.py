# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:06:27 2021

@author: santc
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
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

# 90: Índice inicial 
for i in range(90, 1242):
    # i-90:i -> Índice 0 até i(90 registros iniciais) da coluna 0
    previsores.append(baseTreinamentoNormalizada[i-90:i,0])
    precoReal.append(baseTreinamentoNormalizada[i, 0])

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