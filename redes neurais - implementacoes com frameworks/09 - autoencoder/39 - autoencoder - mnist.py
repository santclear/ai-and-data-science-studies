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
