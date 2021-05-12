# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:02:05 2021

@author: santc
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np

# http://archive.ics.uci.edu/ml/index.php
base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:, 4]

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

classificador = Sequential()

classificador.add(Dense(units = 16, activation = 'softplus', input_dim = 4))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 16, activation = 'softplus'))
classificador.add(Dropout(0.2))


classificador.add(Dense(units = 3, activation = 'softmax'))

classificador.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 500)


classificador_json = classificador.to_json()
with open('iris.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('iris.h5')


arquivo = open('iris.json','r')
estruturaRede = arquivo.read()
arquivo.close()

classificador = model_from_json(estruturaRede)
classificador.load_weights('iris.h5')


novo = np.array([[7.1, 3.2, 4.7, 1.4]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)