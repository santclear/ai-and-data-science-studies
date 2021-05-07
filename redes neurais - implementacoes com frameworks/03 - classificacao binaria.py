# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:39:15 2021

@author: santc
"""

import pandas
from sklearn.model_selection import train_test_split

# Sequential, modelo sequencial de ligação entre camadas
from keras.models import Sequential
# Dense, camada densa, cada neurônio será ligado com todos os neurônios da 
# camada subsequente, rede neural fully connected
from keras.layers import Dense

previsores = pandas.read_csv('../datasets/entradas_breast.csv')
classe = pandas.read_csv('../datasets/saidas_breast.csv')

# test_size = 0.25 indica que serão utilizados 25% da quantidade total de 
# registros para realizar testes e o restante 75% para treinar
# nesse caso a base tem 569 registros que serão divididos 
# 143 (25%) registros para teste e 426 (75%) registros para treinamento
(previsoresTreinamento, 
 previsoresTeste, 
 classeTreinamento, 
 classeTeste) = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

### CRIAÇÃO DA CAMADA O OCULTA E DEFINIÇÃO DA CAMADA DE ENTRADA
# units: quantidade de neurônios da camada oculta. 16 escolhido com base no modelo (30 + 1) / 2
# activation: função de ativação
# kernel_initializer: inicialização dos pesos
# input_dim: quantidade de atributos da camada de entrada. Nesse caso são 30 porque o dataset possuí 30 colunas. (atributos)
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))

### CRIAÇÃO DA CAMADA DE SAÍDA
# como é um problema de classificação binária, saída retornará 0 ou 1, 
# a função de ativação utilizada é a sigmoid
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# optimizer: método de cálculo da descida do gradiente, definido como 
# otimizador adam (um tipo de otimizador estocástico - recomendável em muitos casos)
# loss: função de perda, método de tratamento de erro, como é um problema de 
# classificação binária, foi definido como binary_crossentropy
# metrics: métrica utilizada na avaliação, quantos regs, classificados certos e 
# quantos errados
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# batch_size: a cada 10 registros calculados efetua o processo de ajuste dos pesos
# quantas vezes serão efetuados os ajustes dos pesos
classificador.fit(previsoresTreinamento, classeTreinamento, batch_size = 10, epochs = 100)