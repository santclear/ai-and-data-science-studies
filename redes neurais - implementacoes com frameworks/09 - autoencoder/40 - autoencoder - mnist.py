# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:08:33 2021

@author: santc
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from keras.utils import np_utils

(previsoresTreinamento, classeTreinamento), (previsoresTeste, classe_teste) = mnist.load_data()

# divisão por 255 para normalização dos dados (a escala ficará entre 0 e 1)
# para esse caso específico é possível normalizar através do divisor 255 porque
# os valores da base chegam no máximo até 255 (Poderia ser o usado o MinMaxScaler)
previsoresTreinamento = previsoresTreinamento.astype('float32') / 255
previsoresTeste = previsoresTeste.astype('float32') / 255


classeDummyTreinamento = np_utils.to_categorical(classeTreinamento)
classeDummyTeste = np_utils.to_categorical(classe_teste)


previsoresTreinamento = previsoresTreinamento.reshape((len(previsoresTreinamento), np.prod(previsoresTreinamento.shape[1:])))
previsoresTeste = previsoresTeste.reshape((len(previsoresTeste), np.prod(previsoresTeste.shape[1:])))

# 784 - 32 - 784
autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsoresTreinamento, previsoresTreinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsoresTeste, previsoresTeste))

dimensaoOriginal = Input(shape=(784,))
camadaEncoder = autoencoder.layers[0]
encoder = Model(dimensaoOriginal, camadaEncoder(dimensaoOriginal))

previsoresTreinamentoCodificados = encoder.predict(previsoresTreinamento)
previsoresTesteCodificados = encoder.predict(previsoresTeste)

# sem redução de dimensionalidade
c1 = Sequential()
c1.add(Dense(units = 397, activation = 'relu', input_dim = 784))
c1.add(Dense(units = 397, activation = 'relu'))
c1.add(Dense(units = 10, activation = 'softmax'))
c1.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
           metrics = ['accuracy'])
c1.fit(previsoresTreinamento, classeDummyTreinamento, batch_size = 256,
       epochs = 100, validation_data=(previsoresTeste, classeDummyTeste))

# com redução de dimensionalidade
c2 = Sequential()
c2.add(Dense(units = 21, activation = 'relu', input_dim = 32))
c2.add(Dense(units = 21, activation = 'relu'))
c2.add(Dense(units = 10, activation = 'softmax'))
c2.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
           metrics = ['accuracy'])
c2.fit(previsoresTreinamentoCodificados, classeDummyTreinamento, batch_size = 256,
       epochs = 100, validation_data=(previsoresTesteCodificados, classeDummyTeste))