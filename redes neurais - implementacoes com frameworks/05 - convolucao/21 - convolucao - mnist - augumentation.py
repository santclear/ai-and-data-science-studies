# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:44:56 2021

@author: santc
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(XTreinamento, yTreinamento), (XTeste, yTeste) = mnist.load_data()
previsoresTreinamento = XTreinamento.reshape(XTreinamento.shape[0], 28, 28, 1)
previsoresTeste = XTeste.reshape(XTeste.shape[0], 28, 28, 1)
previsoresTreinamento = previsoresTreinamento.astype('float32')
previsoresTeste = previsoresTeste.astype('float32')
previsoresTreinamento /= 255
previsoresTeste /= 255
classeTreinamento = np_utils.to_categorical(yTreinamento, 10)
classeTeste = np_utils.to_categorical(yTeste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation = 'relu'))
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# augumentation: técnica usada quando há poucas imagens para o treinamento,
# auto gera novas imagens com base nas imagens existentes, criando novas rotacionadas
# aumentadas, reduzidas, etc. Quando há poucas imagens é recomendável esse técnica
# para diminuir o overfitting
geradorTreinamento = ImageDataGenerator(rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
geradorTeste = ImageDataGenerator()

baseTreinamento = geradorTreinamento.flow(previsoresTreinamento, classeTreinamento, batch_size = 128)
baseTeste = geradorTeste.flow(previsoresTeste, classeTeste, batch_size = 128)

classificador.fit_generator(baseTreinamento, steps_per_epoch = 600000 / 128, 
                            epochs = 5, validation_data = baseTeste, validation_steps = 10000 / 128)