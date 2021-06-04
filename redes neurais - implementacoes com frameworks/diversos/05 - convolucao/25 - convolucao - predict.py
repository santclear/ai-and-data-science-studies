# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:28:09 2021

@author: santc
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(XTreinamento, yTreinamento), (XTeste, yTeste) = mnist.load_data()
plt.imshow(XTreinamento[0], cmap = 'gray')
plt.title('classe '+ str(yTreinamento[0]))

previsoresTreinamento = XTreinamento.reshape(XTreinamento.shape[0], 28, 28, 1)
previsoresTeste = XTeste.reshape(XTeste.shape[0], 28, 28, 1)

previsoresTreinamento = previsoresTreinamento.astype('float32')
previsoresTeste = previsoresTeste.astype('float32')

previsoresTreinamento /= 255
previsoresTeste /= 255

classeTreinamento = np_utils.to_categorical(yTreinamento, 10)
classeTeste = np_utils.to_categorical(yTeste, 10)

classificador = Sequential()

classificador.add(Conv2D(32,(3,3),input_shape=(28,28,1), activation='relu'))

classificador.add(BatchNormalization())

classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classificador.fit(previsoresTreinamento, classeTreinamento, batch_size = 128, epochs = 5, validation_data=(previsoresTeste, classeTeste))

resultado = classificador.evaluate(previsoresTeste, classeTeste)

import numpy as np
imagemTeste = np.expand_dims(previsoresTeste[1], axis=0)
previsao = classificador.predict(imagemTeste)
previsao = (previsao > 0.5)

i = 0
for numero in previsao[0]:
    if(numero):
        print('O número é: ', i)
    i += 1
    