# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:59:29 2021

@author: santc
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')

# Exclui os valores na
base = base.dropna()

# Todas as linhas ":" e a coluna 1 "1:2" (Open)
baseTreinamentoPrecosAbertura = base.iloc[:, 1:2].values
# Todas as linhas ":" e a coluna 1 "2:3" (High)
baseTreinamentoPrecosMaxima = base.iloc[:, 2:3].values

# Converte os preços de abertura para a escala de valores entre 0 e 1, isso
# dimui o custo de processamento
normalizador = MinMaxScaler(feature_range=(0,1))
baseTreinamentoPrecosAberturaNormalizada = normalizador.fit_transform(baseTreinamentoPrecosAbertura)
baseTreinamentoPrecosMaximaNormalizada = normalizador.fit_transform(baseTreinamentoPrecosMaxima)

previsores = []
precosReaisAbertura = []
precosReaisMaxima = []

# 1242 qtdRegistrosTreinamento
qtdRegistrosTreinamento = len(baseTreinamentoPrecosAbertura)

periodoAmostral = 90
for i in range(periodoAmostral, qtdRegistrosTreinamento):
    # A cada iteração separa 90 amostras de preços de abertura
    # Por exemplo, se a série tivesse somente 10 preços de abertura (3,5,1,1,7,8,2,9,3,4) 
    # e a amostra fosse 2 a variável amostraPeiodos ficaria assim:
    # 1º iteração -> amostraPeriodos = [3,5]
    # 2ª iteração -> amostraPeriodos = [5,1]
    # 3ª iteração -> amostraPeriodos = [1,1]
    # 4ª iteração -> amostraPeriodos = [1,7]
    # ...
	# 0 é preço de de abertura (Open)
    previsores.append(baseTreinamentoPrecosAberturaNormalizada[i-periodoAmostral:i, 0])
    # Para cada amostra, um preço de abertura para treinamento será usado
    # Tomando o exemplo anterior (3,5,1,1,7,8,2,9,3,4), a cada interação ficará assim:
    # 1º iteração [3,5] -> precoNormalizado = 1
    # 2ª iteração [5,1] -> precoNormalizado = 1
    # 3ª iteração [1,1] -> precoNormalizado = 7
    # 4ª iteração [1,7] -> precoNormalizado = 8
    # ...
	# 0 é preço de abertura (Open)
    precosReaisAbertura.append(baseTreinamentoPrecosAberturaNormalizada[i, 0])
    # Para cada amostra, um preço de abertura para treinamento será usado
    # Tomando o exemplo anterior (3,5,1,1,7,8,2,9,3,4), a cada interação ficará assim:
    # 1º iteração [3,5] -> precoNormalizado = 1
    # 2ª iteração [5,1] -> precoNormalizado = 1
    # 3ª iteração [1,1] -> precoNormalizado = 7
    # 4ª iteração [1,7] -> precoNormalizado = 8
    # ...
	# 0 é preço de máxima (High)
    precosReaisMaxima.append(baseTreinamentoPrecosMaximaNormalizada[i, 0])

previsores, precosReaisAbertura, precosReaisMaxima = np.array(previsores), np.array(precosReaisAbertura), np.array(precosReaisMaxima)

# batch_size
# 1152 registros
qtdRegistrosPrevisores = previsores.shape[0]
# 1 atributo previsor -> Preço de abertura Open
# input_dim
qtdAtributosPrevisores = 1

previsores = np.reshape(previsores, (qtdRegistrosPrevisores, periodoAmostral, qtdAtributosPrevisores))

precosReaisAberturaMaxima = np.column_stack((precosReaisAbertura, precosReaisMaxima))

regressor = Sequential()
# units: 100 células de memória, para dimensionalidade, captura tendência no decorrer
# do tempo, captura a variação temporal
# return_sequences: indica que a informação será encaminhada a diante para a camadas,
# subsequentes, obrigatório quando há mais de 1 camada LSTM
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# Como os dados foram normalizados para retornar valores entre 0 e 1, a função
# sigmoid também pode ser usada
regressor.add(Dense(units = 2, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, precosReaisAberturaMaxima, epochs = 100, batch_size = 32)


### TESTE ##############################

baseTeste = pd.read_csv('petr4_teste.csv')
precosReaisAberturaTeste = baseTeste.iloc[:, 1:2].values
precosReaisMaximaTeste = baseTeste.iloc[:, 2:3].values

# A base de dados de teste tem apenas 22 registros, dessa forma uma opção encontrada,
# apenas para o teste, foi concatenar a base de treinamento com a base de teste
# concatena base e baseTeste por coluna (axis=0)
baseCompletaTeste = pd.concat((base['Open'], baseTeste['Open']), axis = 0)
# Seleciona todos preços da base de teste + um pedaço da base de treinamento
# totalizando 112 registros.
# Isso é necessário nesse caso, porque a LSTM espera que cada registro seja
# fatias de períodos subsequentes, como são 22 registros é necessário que o tamanho
# da base seja 112 para o período 90. No laço abaixo os registros dessa base serão
# iterados 22 vezes (porque 112 - 90 = 22), a cada iteração será concatenado 90 preços,
# é como se fosse uma janela de 90 perídos sendo deslocada para obter os preços
entradas = baseCompletaTeste[len(baseCompletaTeste) - len(baseTeste) - periodoAmostral:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

qtdEntradasTeste = len(entradas)

XTeste = []
for i in range(periodoAmostral, qtdEntradasTeste):
    XTeste.append(entradas[i-periodoAmostral:i, 0])
XTeste = np.array(XTeste)

qtdRegistrosTeste = XTeste.shape[0]
XTeste = np.reshape(XTeste, (qtdRegistrosTeste, periodoAmostral, 1))

previsoes = regressor.predict(XTeste)
previsoes = normalizador.inverse_transform(previsoes)

plt.plot(precosReaisAberturaTeste, color = 'blue', label = 'Preços abertura real')
plt.plot(previsoes[:, 0], color = 'green', label = 'Previsões abertura')

plt.plot(precosReaisMaximaTeste, color = 'red', label = 'Preços máxima real')
plt.plot(previsoes[:, 1], color = 'orange', label = 'Previsões máxima')

plt.title('Previsão de preços das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()