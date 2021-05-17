# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:02:40 2021

@author: santc
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

(XTreinamento, yTreinamento), (XTeste, yTeste) = mnist.load_data()
plt.imshow(XTreinamento[0], cmap = 'gray')
plt.title('classe '+ str(yTreinamento[0]))

# XTreinamento.shape[0]: quantidade de imagens (registros), 
# 28: altura
# 28: largura
# 1: 1 Canal(Cinza)
previsoresTreinamento = XTreinamento.reshape(XTreinamento.shape[0], 28, 28, 1)

previsoresTeste = XTeste.reshape(XTeste.shape[0], 28, 28, 1)
previsoresTreinamento = previsoresTreinamento.astype('float32')
previsoresTeste = previsoresTeste.astype('float32')

previsoresTreinamento /= 255
previsoresTeste /= 255

classeTreinamento = np_utils.to_categorical(yTreinamento, 10)
classeTeste = np_utils.to_categorical(yTeste, 10)