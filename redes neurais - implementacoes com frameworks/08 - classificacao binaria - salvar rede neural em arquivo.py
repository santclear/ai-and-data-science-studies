# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:26:17 2021

@author: santc
"""

import pandas as pd

# Sequential, modelo sequencial de ligação entre camadas
from keras.models import Sequential
# Dense, camada densa, cada neurônio será ligado com todos os neurônios da 
# camada subsequente, rede neural fully connected
from keras.layers import Dense, Dropout

previsores = pd.read_csv('../datasets/entradas_breast.csv')
classe = pd.read_csv('../datasets/saidas_breast.csv')

classificador = Sequential()

### CRIAÇÃO DA CAMADA OCULTA E DEFINIÇÃO DA CAMADA DE ENTRADA
# units: quantidade de neurônios da camada oculta. 16 escolhido com base no modelo (30 + 1) / 2
# activation: função de ativação
# kernel_initializer: inicialização dos pesos
# input_dim: quantidade de atributos da camada de entrada. Nesse caso são 30 porque o dataset possuí 30 colunas. (atributos)
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))

# dropout de 20%(0.2) na camada de entrada
# O dropout zera aleatóriamente a entrada de alguns neurônios, com o objetivo 
# de mitigar o overfitting, é recomendável aplicar dropout entre 20% e 30%
classificador.add(Dropout(0.2))

# aprofundamento da camada oculta, 16 neurônios em um segundo nível
# não é necessário declarar input_dim porque não está conectado com a camada de
# entrada, está conectado com a camada oculta de primeiro nível
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))

# dropout de 20%(0.2) na camada oculta
classificador.add(Dropout(0.2))

### CRIAÇÃO DA CAMADA DE SAÍDA
# 1 neurônio na camada de saída, como é um problema de classificação binária, 
# saída retornará 0 ou 1, a função de ativação utilizada é a sigmoid
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

# Salva o modelo da rede neural em arquivo json
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

# Salva todos os pesos da rede neural em arquivo h5
classificador.save_weights('classificador_breast.h5')