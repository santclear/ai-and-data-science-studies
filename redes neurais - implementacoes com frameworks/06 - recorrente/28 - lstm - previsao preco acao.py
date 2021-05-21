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