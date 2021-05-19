# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:08:52 2021

@author: santc
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X, y), (XTeste, yTeste) = mnist.load_data()
previsores = X.reshape(X.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')
previsores /= 255
# Converte cada uma das 10 classes para onehot encoding (dummy)
classe = np_utils.to_categorical(y, 10)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
resultados = []

a = np.zeros(5)
b = np.zeros(shape = (classe.shape[0], 1))

for indiceTreinamento, indiceTeste in kfold.split(previsores, 
                                                    np.zeros(shape = (classe.shape[0], 1))):
    #print('Índices treinamento: ', indice_treinamento, 'Índice teste', indice_teste)
    classificador = Sequential()
    
    classificador.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation = 'relu'))
    classificador.add(MaxPooling2D(pool_size = (2,2)))
    classificador.add(Flatten())
    
    classificador.add(Dense(units = 128, activation = 'relu'))
    
    classificador.add(Dense(units = 10, activation = 'softmax'))
    classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    classificador.fit(previsores[indiceTreinamento], classe[indiceTreinamento], batch_size = 128, epochs = 5)
    
    precisao = classificador.evaluate(previsores[indiceTeste], classe[indiceTeste])
    
    resultados.append(precisao[1])

#media = resultados.mean()
media = sum(resultados) / len(resultados)
