# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:50:05 2021

@author: santc
"""

import pandas as pd

from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Input, Activation 
from tensorflow.keras.models import Model

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Base obtida do site: kaggle.com
base = pd.read_csv('games.csv')


base = base.drop('Other_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('NA_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.dropna(axis = 0)

base = base.loc[base['Global_Sales'] > 1]

nomeJogos = base.Name
base = base.drop('Name', axis = 1)
previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values

vendaGlobal = base.iloc[:, 4].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# (99 + 1) / 2 = 50
camadaEntrada = Input(shape=(99,))
camadaOculta1 = Dense(units = 50, activation = Activation(activations.sigmoid))(camadaEntrada)
camadaOculta2 = Dense(units = 50, activation = Activation(activations.sigmoid))(camadaOculta1)
camadaSaida1 = Dense(units = 1, activation = Activation(activations.linear))(camadaOculta2)

regressor = Model(inputs = camadaEntrada, outputs = [camadaSaida1])

regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')
regressor.fit(previsores, [vendaGlobal], epochs = 1000, batch_size = 100)
vendaGlobal2 = regressor.predict(previsores)