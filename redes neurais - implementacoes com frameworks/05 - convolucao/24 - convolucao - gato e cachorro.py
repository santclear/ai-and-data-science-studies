# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:36:08 2021

@author: santc
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classificador = Sequential()

classificador.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation='softplus'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation='softplus'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

classificador.add(Dense(units=128, activation=('softplus')))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation=('softplus')))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation=('sigmoid')))

classificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

geradorTreinamento = ImageDataGenerator(rescale=1./255, rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

geradorTeste = ImageDataGenerator(rescale=1./255)

baseTreinamento = geradorTreinamento.flow_from_directory('gatosecachorros/training_set',
                                                         target_size=(64,64), 
                                                         batch_size = 32,
                                                         class_mode='binary')

baseTeste = geradorTeste.flow_from_directory('gatosecachorros/test_set',
                                                         target_size=(64,64), 
                                                         batch_size = 32,
                                                         class_mode='binary')

classificador.fit_generator(baseTreinamento, steps_per_epoch = 4000 / 32, 
                            epochs = 5, validation_data = baseTeste, validation_steps = 1000 / 32)

imagemTeste = image.load_img('crystal.jpg', target_size=(64,64))

imagemTeste = image.img_to_array(imagemTeste)
imagemTeste /= 255
imagemTeste = np.expand_dims(imagemTeste, axis=0)
previsao = classificador.predict(imagemTeste)

baseTreinamento.class_indices

if(previsao > 0.5): print('Gato')
else: print('Cachorro')