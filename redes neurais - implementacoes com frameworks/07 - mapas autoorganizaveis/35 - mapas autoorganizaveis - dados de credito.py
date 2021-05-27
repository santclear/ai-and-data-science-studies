# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:32:32 2021

@author: santc
"""

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna()

# Para os registros age(idade) menores que zero, serão substituídos pelo valor
# retorna pela média de idades 40.927688881035856
base.age.mean()
base.loc[base.age < 0, 'age'] = 40.92

# X: Atributos previsores
# Todas as linhas ":" e colunas de 0 até 3 (Atributos: clientid até loan)
X = base.iloc[:, 0:4].values
# y: Classes
# Todas as linhas e coluna 4 (Atributo: default)
y = base.iloc[:, 4].values

# Normalização dos dados (diminui a escala dos valores) para acelerar o 
# processamento dos valores no treinamento e teste
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)