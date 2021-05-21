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

# 1142 qtdRegistrosTreinamento
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