# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:34:39 2021

@author: santc
"""

import numpy as np
import pandas as pd
from keras.models import model_from_json

### Carregamento de rede neural
arquivo = open('classificador_breast.json','r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')
### ###

novo = np.array([[15.80, 8.34, 119, 890, 0.11, 0.21, 0.07, 0.124, 0.190,
                  0.21, 0.1, 1100, 0.86, 4700, 145.3, 0.009, 0.01, 0.06, 0.018,
                  0.05, 0.010, 24.15, 17.64, 179.5, 2020, 0.12, 0.215,
                  0.85, 159, 0.361]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

previsores = pd.read_csv('../datasets/entradas_breast.csv')
classe = pd.read_csv('../datasets/saidas_breast.csv')

classificador.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])
# 1ª linha: valor da função de erro
# 2ª linha: precisão
resultado = classificador.evaluate(previsores, classe)