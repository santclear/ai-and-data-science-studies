# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:29:47 2021

@author: santc
"""

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

(previsoresTreinamento, _), (previsoresTeste, _) = mnist.load_data()
previsoresTreinamento = previsoresTreinamento.astype('float32') / 255
previsoresTeste = previsoresTeste.astype('float32') / 255

previsoresTreinamento = previsoresTreinamento.reshape((len(previsoresTreinamento), np.prod(previsoresTreinamento.shape[1:])))
previsoresTeste = previsoresTeste.reshape((len(previsoresTeste), np.prod(previsoresTeste.shape[1:])))

# 784 -> 128 -> 64 -> |32| -> 64 -> 128 -> 784
autoencoder = Sequential()

# Encode
autoencoder.add(Dense(units = 128, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 64, activation = 'relu'))
autoencoder.add(Dense(units = 32, activation = 'relu'))

# Decode
autoencoder.add(Dense(units = 64, activation = 'relu'))
autoencoder.add(Dense(units = 128, activation = 'relu'))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))

autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'
])
autoencoder.fit(previsoresTreinamento, previsoresTreinamento,
                epochs = 300, batch_size = 256, 
                validation_data = (previsoresTeste, previsoresTeste))

dimensaoOriginal = Input(shape=(784,))
camadaEncoder1 = autoencoder.layers[0]
camadaEncoder2 = autoencoder.layers[1]
camadaEncoder3 = autoencoder.layers[2]
encoder = Model(dimensaoOriginal,
                camadaEncoder3(camadaEncoder2(camadaEncoder1(dimensaoOriginal))))
encoder.summary()

imagensCodificadas = encoder.predict(previsoresTeste)
imagensDecodificadas = autoencoder.predict(previsoresTeste)

numeroImagens = 10
imagensTeste = np.random.randint(previsoresTeste.shape[0], size = numeroImagens)
plt.figure(figsize=(18,18))
for i, indiceImagem in enumerate(imagensTeste):   
    # imagem original
    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(previsoresTeste[indiceImagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10,10,i + 1 + numeroImagens)
    plt.imshow(imagensCodificadas[indiceImagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numeroImagens * 2)
    plt.imshow(imagensDecodificadas[indiceImagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())