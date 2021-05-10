# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:36:01 2021

@author: santc
"""

import pandas as pd
import keras

# Sequential, modelo sequencial de ligação entre camadas
from keras.models import Sequential
# Dense, camada densa, cada neurônio será ligado com todos os neurônios da 
# camada subsequente, rede neural fully connected
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('../datasets/entradas_breast.csv')
classe = pd.read_csv('../datasets/saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    
    ### CRIAÇÃO DA CAMADA OCULTA E DEFINIÇÃO DA CAMADA DE ENTRADA
    # units: quantidade de neurônios da camada oculta. 16 escolhido com base no modelo (30 + 1) / 2
    # activation: função de ativação
    # kernel_initializer: inicialização dos pesos
    # input_dim: quantidade de atributos da camada de entrada. Nesse caso são 30 porque o dataset possuí 30 colunas. (atributos)
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
    # aprofundamento da camada oculta, 16 neurônios em um segundo nível
    # não é necessário declarar input_dim porque não está conectado com a camada de
    # entrada, está conectado com a camada oculta de primeiro nível
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    
    ### CRIAÇÃO DA CAMADA DE SAÍDA
    # 1 neurônio na camada de saída, como é um problema de classificação binária, 
    # saída retornará 0 ou 1, a função de ativação utilizada é a sigmoid
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    # optimizer: método de cálculo da descida do gradiente, definido como 
    # otimizador adam (um tipo de otimizador estocástico - recomendável em muitos casos)
    # loss: função de perda, método de tratamento de erro, como é um problema de 
    # classificação binária, foi definido como binary_crossentropy
    # metrics: métrica utilizada na avaliação, quantos regs, classificados certos e 
    # quantos errados
    #classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    # lr: taxa de aprendizagem
    # decay: decremento da taxa de aprendizagem a cada iteração, objetivo de suavizar
    # a descida do gradiente
    # clipvalue: prende o valor ao chegar no mínimo global da descida do gradiente
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 100, batch_size = 10)

### A validação cruzada é uma das formas mais utilizadas pelos cientístas e
### eficiente para a avaliação de redes neurais artificias
# validação cruzada entre bases de treinamento e teste (cross validation),
# cv: quantidade de execuções (nesse caso executado 10 vezes as 100 épocas)
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')
media = resultados.mean()
# quantos valores estão variando em relação a média
# quanto maior esse valor, maior é a tendência de overfitting (rede neural muito
# adaptada aos dados de treinamento e teste, algo ruim, pois quando é passada
# uma base nova à rede, ela não terá bons resultados)
desvioPadrao = resultados.std()