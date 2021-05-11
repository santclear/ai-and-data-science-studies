# -*- coding: utf-8 -*-
"""
Created on Tue May 11 01:21:56 2021

@author: santc
"""
# http://archive.ics.uci.edu/ml/index.php

import pandas as pd

base = pd.read_csv('iris.csv')

# Todas as linhas (:) e colunas do 0 ao 3 (0:4), 4 é o limite!
previsores = base.iloc[:,0:4].values
# Todas as linhas (:) e coluna 4
classe = base.iloc[:, 4]

from sklearn.model_selection import train_test_split

# test_size = 0.25 indica que serão utilizados 25% da quantidade total de 
# registros para realizar testes e o restante 75% para treinar
# nesse caso a base tem 150 registros que serão divididos 
# 38 (25%) registros para teste e 112 (75%) registros para treinamento
# classeTeste: saídas esperadas
(previsoresTreinamento, 
 previsoresTeste, 
 classeTreinamento, 
 classeTeste) = train_test_split(previsores, classe, test_size=0.25)