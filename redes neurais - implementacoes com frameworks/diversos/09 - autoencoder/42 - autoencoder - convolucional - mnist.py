# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:32:41 2021

@author: santc
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

# A rede irá codificar e decodificar, devido a isso não serão criadas as classes
# de treinamento e teste, então "_" para definir que não será criada as variáveis
# de classe
(previsoresTreinamento, _), (previsoresTeste, _) = mnist.load_data()
previsoresTreinamento = previsoresTreinamento.reshape((len(previsoresTreinamento), 28, 28, 1))
previsoresTeste = previsoresTeste.reshape((len(previsoresTeste), 28, 28, 1))

# divisão por 255 para normalização dos dados (a escala ficará entre 0 e 1)
# para esse caso específico é possível normalizar através do divisor 255 porque
# os valores da base chegam no máximo até 255 (Poderia ser o usado o MinMaxScaler)
previsoresTreinamento = previsoresTreinamento.astype('float32') / 255
previsoresTeste = previsoresTeste.astype('float32') / 255

autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape=(28,28,1)))
autoencoder.add(MaxPooling2D(pool_size = (2,2)))

autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size = (2,2), padding='same'))

# 4, 4, 8
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same', strides = (2,2)))

autoencoder.add(Flatten())

autoencoder.add(Reshape((4,4,8)))

# Decoder
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding='same'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu'))
autoencoder.add(UpSampling2D(size = (2,2)))
autoencoder.add(Conv2D(filters = 1, kernel_size = (3,3), activation = 'sigmoid', padding='same'))
autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsoresTreinamento, previsoresTreinamento,
                epochs = 100, batch_size = 256, 
                validation_data = (previsoresTeste, previsoresTeste))

# Flatten 128
encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('flatten_6').output)
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
    plt.imshow(imagensCodificadas[indiceImagem].reshape(16,8))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numeroImagens * 2)
    plt.imshow(imagensDecodificadas[indiceImagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
