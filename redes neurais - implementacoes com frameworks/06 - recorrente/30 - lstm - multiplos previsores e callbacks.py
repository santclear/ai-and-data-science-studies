# -*- coding: utf-8 -*-
"""
Created on Mon May 24 01:19:41 2021

@author: santc
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')

# Exclui os valores na
base = base.dropna()

# Todas as linhas ":" e colunas de 1 até 6 "1:7" (Open, High, Low, Close e Adj Close)
baseTreinamento = base.iloc[:, 1:7].values

# Converte os preços de abertura para a escala de valores entre 0 e 1, isso
# dimui o custo de processamento
normalizador = MinMaxScaler(feature_range=(0,1))
baseTreinamentoNormalizada = normalizador.fit_transform(baseTreinamento)

previsores = []
precoReal = []

# 1242 qtdRegistrosTreinamento
qtdRegistrosTreinamento = len(baseTreinamento)
# Quantidade de períodos que serão utilizados a cada iteração no treinamento e teste
# da rede neural
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
	# 0:6 -> colunas de 0 até 6 (Open, High, Low, Close e Adj Close)
    amostraPeriodos = baseTreinamentoNormalizada[i-periodoAmostral:i,0:6]
    previsores.append(amostraPeriodos)
    
    # Para cada amostra, um preço de abertura para treinamento será usado
    # Tomando o exemplo anterior (3,5,1,1,7,8,2,9,3,4), a cada interação ficará assim:
    # 1º iteração [3,5] -> precoNormalizado = 1
    # 2ª iteração [5,1] -> precoNormalizado = 1
    # 3ª iteração [1,1] -> precoNormalizado = 7
    # 4ª iteração [1,7] -> precoNormalizado = 8
    # ...
	# 0 -> coluna 0 (Open)
    precoNormalizado = baseTreinamentoNormalizada[i, 0]
    precoReal.append(precoNormalizado)

previsores, precoReal = np.array(previsores), np.array(precoReal)
# batch_size
# 1152 registros
qtdRegistros = previsores.shape[0]
# timesteps
# 90 intervalos
periodoAmostral = previsores.shape[1]
# 1 atributo previsor -> Preço de abertura Open
# input_dim
qtdAtributosPrevisores = 6

regressor = Sequential()

regressor.add(LSTM(units=100, return_sequences=True, input_shape=(periodoAmostral, qtdAtributosPrevisores)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

# Como os dados foram normalizados para retornar valores entre 0 e 1, a função
# linear também pode ser usada
regressor.add(Dense(units=1, activation='sigmoid'))
# Para esse problema também poderia ser usado o optimizer rmsprop
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

### Callbacks: coleta informações de estados ou dados estatísticos do modelo 
# durante treinamento
#
# Interrompe o treinamento quando uma função monitorada parou de
# melhorar
# monitor: tipo de função que será monitorada, geralmente a loss function
# min_delta: mundança mínima que será considerada como melhoria. Por exemplo, se
# o valor definido é 0.01, se na próxima iteração a rede não conseguir melhorar
# esse valor então o treinamento é interrompido
# patience: número de épocas que seguirão sem ter melhorias no resultado, após
# alcançar esse número então o treinamento será interrompido
interrompeComMelhoresResultados = EarlyStopping(monitor = 'loss', min_delta=1e-10, patience=10, verbose=1)
# Reduz a taxa de aprendizagem quando uma métrica parou de melhorar
# factor: valor que a taxa de aprendizagem será reduzida
reduzTaxaAprendizagemSeParouDeMelhorar = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)

# Salva os pesos a cada época
# save_best_only: salva somente os pesos com melhores resultados
salvaMelhoresPesos = ModelCheckpoint(filepath='pesos.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, precoReal, epochs=100, batch_size=32, 
			  callbacks=(
				  interrompeComMelhoresResultados,
				  reduzTaxaAprendizagemSeParouDeMelhorar,
				  salvaMelhoresPesos)
			  )

### TESTE ##############################

baseTeste = pd.read_csv('petr4_teste.csv')

# Todas as linhas ":" e a coluna 1 "1:2" (Open - Abertura)
precoRealTeste = baseTeste.iloc[:, 1:2].values

frames = [base, baseTeste]
# A base de dados de teste tem apenas 22 registros, dessa forma uma opção encontrada,
# apenas para o teste, foi concatenar a base de treinamento com a base de teste
# concatena base e baseTeste por coluna (axis=0)
baseCompleta = pd.concat(frames, axis = 0)
baseCompleta = baseCompleta.drop('Date', axis=1)

# Seleciona todos preços da base de teste + um pedaço da base de treinamento
# totalizando 112 registros.
# Isso é necessário nesse caso, porque a LSTM espera que cada registro seja
# fatias de períodos subsequentes, como são 22 registros é necessário que o tamanho
# da base seja 112 para o período 90. No laço abaixo os registros dessa base serão
# iterados 22 vezes (porque 112 - 90 = 22), a cada iteração será concatenado 90 preços,
# é como se fosse uma janela de 90 perídos sendo deslocada para obter os preços
entradas = baseCompleta[len(baseCompleta) - len(baseTeste) - periodoAmostral:].values
entradas = normalizador.transform(entradas)

qtdRegistros = len(entradas)
XTeste = []
for i in range(periodoAmostral, qtdRegistros):
    XTeste.append(entradas[i-periodoAmostral:i, 0:6])

XTeste = np.array(XTeste)
previsoes = regressor.predict(XTeste)

normalizadorPrevisao = MinMaxScaler(feature_range=(0,1))
normalizadorPrevisao.fit_transform(baseTreinamento[:,0:1])
previsoes = normalizadorPrevisao.inverse_transform(previsoes)

previsoes.mean()
precoRealTeste.mean()
    
plt.plot(precoRealTeste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço da ação PETR4')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()