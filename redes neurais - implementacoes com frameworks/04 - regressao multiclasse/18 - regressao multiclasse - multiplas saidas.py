# -*- coding: utf-8 -*-
"""
Created on Sat May 15 01:22:04 2021

@author: santc
"""

import pandas as pd

from tensorflow.keras.layers import Dense, Input 
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
# Pega as colunas desejadas
previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
# Como não é problema de classificação, as variáveis são definidas com o nome de
# valores que se deseja prever através de aproximação numérica.
# Pega as colunas NA Sales, EU Sales e JP Sales  (multi tipos (3) de valores para
# treinamento)
vendaNA = base.iloc[:, 4].values
vendaEU = base.iloc[:, 5].values
vendaJP = base.iloc[:, 6].values

################################################
### PREPROCESSAMENTO: Converte dados numéricos em Onehot Encoder
# Transforma as categorias 0, 2, 3, 8 em algo parecido com:
# 0: 1 0 0 ... 0 
# 2: 0 1 0 ... 0
# 3: 0 0 1 ... 0
# ... ...
# onehot encoder pode ser usado quando não há uma ordem de importância entre 
# os atributos, por exemplo o tipo de câmbio não pode ser afirmado que "manual 
# é maior que automático" e vice-versa
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

camadaEntrada = Input(shape=(61,))
camadaOculta1 = Dense(units = 32, activation = 'sigmoid')(camadaEntrada)
camadaOculta2 = Dense(units = 32, activation = 'sigmoid')(camadaOculta1)
camadaSaida1 = Dense(units = 1, activation = 'linear')(camadaOculta2)
camadaSaida2 = Dense(units = 1, activation = 'linear')(camadaOculta2)
camadaSaida3 = Dense(units = 1, activation = 'linear')(camadaOculta2)

regressor = Model(inputs = camadaEntrada, outputs = [camadaSaida1, camadaSaida2, camadaSaida3])
regressor.compile(optimizer = 'adam', loss = 'mse')
regressor.fit(previsores, [vendaNA, vendaEU, vendaJP], epochs = 5000, batch_size = 100)
previsaoNA, previsaoEU, previsaoJP = regressor.predict(previsores)