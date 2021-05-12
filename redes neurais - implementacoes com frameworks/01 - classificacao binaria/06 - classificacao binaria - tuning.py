# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:57:15 2021

@author: santc
"""

import pandas as pd

# Sequential, modelo sequencial de ligação entre camadas
from keras.models import Sequential
# Dense, camada densa, cada neurônio será ligado com todos os neurônios da 
# camada subsequente, rede neural fully connected
from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    
    ### CRIAÇÃO DA CAMADA OCULTA E DEFINIÇÃO DA CAMADA DE ENTRADA
    # units: quantidade de neurônios da camada oculta. 16 escolhido com base no modelo (30 + 1) / 2
    # activation: função de ativação
    # kernel_initializer: inicialização dos pesos
    # input_dim: quantidade de atributos da camada de entrada. Nesse caso são 30 porque o dataset possuí 30 colunas. (atributos)
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))
    
    # dropout de 20%(0.2) na camada de entrada
    # O dropout zera aleatóriamente a entrada de alguns neurônios, com o objetivo 
    # de mitigar o overfitting, é recomendável aplicar dropout entre 20% e 30%
    classificador.add(Dropout(0.2))
    
    # aprofundamento da camada oculta, 16 neurônios em um segundo nível
    # não é necessário declarar input_dim porque não está conectado com a camada de
    # entrada, está conectado com a camada oculta de primeiro nível
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    
    # dropout de 20%(0.2) na camada oculta
    classificador.add(Dropout(0.2))
    
    ### CRIAÇÃO DA CAMADA DE SAÍDA
    # 1 neurônio na camada de saída, como é um problema de classificação binária, 
    # saída retornará 0 ou 1, a função de ativação utilizada é a sigmoid
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede)

parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}
# GridSearchCV: tuning, busca os melhores parâmetros de configuração de uma rede neural
# param_grid: parâmetros que serão testados na busca das melhores configurações
# cv: quantidade de execuções
grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring = 'accuracy', cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_