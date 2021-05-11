# -*- coding: utf-8 -*-
"""
Created on Tue May 11 01:21:56 2021

@author: santc
"""
# http://archive.ics.uci.edu/ml/index.php

import pandas as pd
# Sequential, modelo sequencial de ligação entre camadas
from keras.models import Sequential
# Dense, camada densa, cada neurônio será ligado com todos os neurônios da 
# camada subsequente, rede neural fully connected
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
# LabelEncoder: classe para converter atributo categórico em numérico
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')

# Todas as linhas (:) e colunas do 0 ao 3 (0:4), 4 é o limite!
previsores = base.iloc[:,0:4].values
# Todas as linhas (:) e coluna 4
classe = base.iloc[:, 4]

labelencoder = LabelEncoder()
# Converte (transforma) atributos categóricos em numéricos
# cada uma das 3 classes serão convertidas para os valores 0, 1, 2
classe = labelencoder.fit_transform(classe)
# Transforma as classes 0, 1 e 2 em:
# 0 (Iris setosa): 1 0 0
# 1 (Iris virginica): 0 1 0
# 2 (Iris versicolor): 0 0 1
classeDummy = np_utils.to_categorical(classe)

# test_size = 0.25 indica que serão utilizados 25% da quantidade total de 
# registros para realizar testes e o restante 75% para treinar
# nesse caso a base tem 150 registros que serão divididos 
# 38 (25%) registros para teste e 112 (75%) registros para treinamento
# classeTeste: saídas esperadas
(previsoresTreinamento, 
 previsoresTeste, 
 classeTreinamento, 
 classeTeste) = train_test_split(previsores, classeDummy, test_size=0.25)

classificador = Sequential()
### CRIAÇÃO DA CAMADA OCULTA E DEFINIÇÃO DA CAMADA DE ENTRADA
# units: quantidade de neurônios da camada oculta. 16 escolhido com base no modelo (4 + 3) / 2
# activation: função de ativação
# input_dim: quantidade de atributos da camada de entrada. Nesse caso são 4 porque o dataset possuí 4 colunas. (atributos)
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))

### CRIAÇÃO DA CAMADA DE SAÍDA
# 3 neurônio na camada de saída, como é um problema de classificação multiclasse, 
# a função de ativação sugerida é a softmax
classificador.add(Dense(units = 3, activation = 'softmax'))

# optimizer: método de cálculo da descida do gradiente, definido como 
# otimizador adam (um tipo de otimizador estocástico - recomendável em muitos casos)
# loss: função de perda, método de tratamento de erro, como é um problema de 
# classificação multiclasse, foi definido como categorical_crossentropy
# metrics: métrica utilizada na avaliação, quantos regs, classificados certos e 
# quantos errados
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

classificador.fit(previsoresTreinamento, classeTreinamento, batch_size = 10, epochs = 1000)