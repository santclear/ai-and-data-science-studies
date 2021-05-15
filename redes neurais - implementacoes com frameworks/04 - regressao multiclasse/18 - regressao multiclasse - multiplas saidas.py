# -*- coding: utf-8 -*-
"""
Created on Sat May 15 01:22:04 2021

@author: santc
"""

import pandas as pd

import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout, Activation, Input 
from tensorflow.keras.models import Model

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Base obtida do site: kaggle.com
base = pd.read_csv('games.csv')

################################################
### PREPROCESSAMENTO: Exclusão de colunas (axis = 1) que não serão necessárias no treinamento e teste,
### após esse processamento a base terá 13 atributos (colunas)
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

# Excluí todas as linhas (axis = 0) que possui a o valor nan
base = base.dropna(axis = 0)
# Mantém os dados maior que 1 e exclui o restante
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

# Para o que se deseja prever, não é bom para esse modelo uma variabilidade
# alta de dados categóricos (Length: 223) em comparação com o tamanho 
# total da base (Size: 258)
base['Name'].value_counts()
# A coluna Name será excluída, mas antes será salva na variável nomeJogos para
# fins de análise
nomeJogos = base.Name
# Excluí a coluna Name
base = base.drop('Name', axis = 1)