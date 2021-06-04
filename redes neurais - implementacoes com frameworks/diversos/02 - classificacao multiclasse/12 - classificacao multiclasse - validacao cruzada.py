# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:11:16 2021

@author: santc
"""

import pandas as pd
# Sequential, modelo sequencial de ligação entre camadas
from keras.models import Sequential
# Dense, camada densa, cada neurônio será ligado com todos os neurônios da 
# camada subsequente, rede neural fully connected
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# LabelEncoder: classe para converter atributo categórico em numérico
from sklearn.preprocessing import LabelEncoder

# http://archive.ics.uci.edu/ml/index.php
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

def criarRede():
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
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 1000, batch_size = 10)

### A validação cruzada é uma das formas mais utilizadas pelos cientístas e
### eficiente para a avaliação de redes neurais artificias
# validação cruzada entre bases de treinamento e teste (cross validation),
# cv: quantidade de execuções (nesse caso executado 10 vezes as 100 épocas),
# o valor 10 é muito utilizado por pesquisadores
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring = 'accuracy')
media = resultados.mean()
# quantos valores estão variando em relação a média
# quanto maior esse valor, maior é a tendência de overfitting (rede neural muito
# adaptada aos dados de treinamento e teste, algo ruim, pois quando é passada
# uma base nova à rede, ela não terá bons resultados)
desvioPadrao = resultados.std()