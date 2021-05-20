# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:06:15 2021

@author: santc
"""

import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


(XTreinamento, yTreinamento), (XTeste, yTeste) = cifar10.load_data()
plt.imshow(XTreinamento[0])
plt.title('classe '+ str(yTreinamento[0]))


previsoresTreinamento = XTreinamento.reshape(XTreinamento.shape[0], 32, 32, 3)
previsoresTeste = XTeste.reshape(XTeste.shape[0], 32, 32, 3)

previsoresTreinamento = previsoresTreinamento.astype('float32')
previsoresTeste = previsoresTeste.astype('float32')

previsoresTreinamento /= 255
previsoresTeste /= 255

classeTreinamento = np_utils.to_categorical(yTreinamento, 10)
classeTeste = np_utils.to_categorical(yTeste, 10)

classificador = Sequential()

classificador.add(Conv2D(64,(3,3),input_shape=(32,32,3), activation='softplus'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(64, (3,3), activation='softplus'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

classificador.add(Dense(units=256, activation='softplus'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=256, activation='softplus'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classificador.fit(previsoresTreinamento, classeTreinamento, batch_size = 256, epochs = 5, validation_data=(previsoresTeste, classeTeste))

resultado = classificador.evaluate(previsoresTeste, classeTeste)

# [1.1708025932312012, 0.6434000134468079]