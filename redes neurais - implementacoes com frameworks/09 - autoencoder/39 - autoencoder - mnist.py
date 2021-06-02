# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:18:19 2021

@author: santc
"""

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

# A rede irá codificar e decodificar, devido a isso não serão criadas as classes
# de treinamento e teste, então "_" para definir que não será criada as variáveis
# de classe
(previsoresTreinamento, _), (previsoresTeste, _) = mnist.load_data()

# divisão por 255 para normalização dos dados (a escala ficará entre 0 e 1)
# para esse caso específico é possível normalizar através do divisor 255 porque
# os valores da base chegam no máximo até 255 (Poderia ser o usado o MinMaxScaler)
previsoresTreinamento = previsoresTreinamento.astype('float32') / 255
previsoresTeste = previsoresTeste.astype('float32') / 255

# 60000
qtdRegistrosPrevisoresTreinamanento = len(previsoresTreinamento)
# 28x28
dimensoesPrevisoresTreinamento = previsoresTreinamento.shape[1:]
# 784
dimensoesPrevisoresMutiplicadaTreinamento = np.prod(dimensoesPrevisoresTreinamento)
previsoresTreinamento = previsoresTreinamento.reshape(
	(qtdRegistrosPrevisoresTreinamanento, dimensoesPrevisoresMutiplicadaTreinamento))

# 10000
qtdRegistrosPrevisoresTeste = len(previsoresTeste)
# 28x28
dimensoesPrevisoresTeste = previsoresTeste.shape[1:]
# 784
dimensoesPrevisoresMutiplicadaTeste = np.prod(dimensoesPrevisoresTeste)
previsoresTeste = previsoresTeste.reshape((qtdRegistrosPrevisoresTeste, dimensoesPrevisoresMutiplicadaTeste))

# Camada entrada: 784 neurônios (valor dos pixels)
# Camada oculta (para reduz de dimensinalidade (encode)): 32 neurônios (valor dos pixels)
# Camada de saída (decode): 784 neurônios (valor dos pixels)
fatorCompactacao = 784 / 32

autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
autoencoder.summary()
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

# 1º parâmetro X: previsores
# 2º parâmetro y: classes (nesses caso, como a rede terá apenas o objetivo de
# treinar a codificação e decodificação ao invés de classificação e como 
# não é permitido passar nulo, será passado o mesmo parâmetro declarado 
# anteriormente, previsores). Previsão supervisionado por si própria, característica
# principal de uma rede com finalidade de autoencoder
autoencoder.fit(previsoresTreinamento, previsoresTreinamento, epochs = 50, 
				batch_size = 256, validation_data = (previsoresTeste, previsoresTeste))

dimensaoOriginal = Input(shape=(784,))
camadaEncoder = autoencoder.layers[0]
encoder = Model(dimensaoOriginal, camadaEncoder(dimensaoOriginal))
encoder.summary()

imagensCodificadas = encoder.predict(previsoresTeste)
imagensDecodificadas = autoencoder.predict(previsoresTeste)

numeroImagens = 10
imagensTeste = np.random.randint(previsoresTeste.shape[0], size = numeroImagens)
plt.figure(figsize=(18,18))
for i, indiceImagem in enumerate(imagensTeste):
    #print(i)
    #print(indiceImagem)
    
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