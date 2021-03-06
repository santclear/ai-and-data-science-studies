# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:11:31 2021

@author: santc
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Base obtida do site https://www.kaggle.com/
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')


base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)
i1 = base.loc[base.price <= 10]
base.price.mean()
base = base [base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts()
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts()
base.loc[pd.isnull(base['model'])]
base['model'].value_counts()
base.loc[pd.isnull(base['fuelType'])] 
base['fuelType'].value_counts()
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()

valores = {'vehicleType':'limousine',
           'gearbox':'manuell',
           'model':'golf',
           'fuelType':'benzin',
           'notRepairedDamage':'nein'}

base = base.fillna(value = valores)
previsores = base.iloc[:, 1:13].values
precoReal = base.iloc[:,0].values
labelEncoderPrevisores = LabelEncoder()
previsores[:, 0] = labelEncoderPrevisores.fit_transform(previsores[:,0])
previsores[:, 1] = labelEncoderPrevisores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoderPrevisores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoderPrevisores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelEncoderPrevisores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoderPrevisores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelEncoderPrevisores.fit_transform(previsores[:, 10])
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

def criarRede(loss):
    regressor = Sequential()
    regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
    regressor.add(Dense(units = 158, activation = 'relu'))

    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss = loss)
    
    return regressor

regressor = KerasRegressor(build_fn = criarRede, epochs = 100, batch_size = 300)

parametros = {'loss': ['mean_squared_error', 
                       'mean_absolute_error', 
                       'mean_absolute_percentage_error', 
                       'mean_squared_logarithmic_error', 
                       'squared_hinge']}

grid_search = GridSearchCV(estimator = regressor, param_grid = parametros, scoring = 'accuracy', cv = 5)
grid_search = grid_search.fit(previsores, precoReal)
melhores_parametros = grid_search.best_params_ # {'loss': 'mean_squared_error'}