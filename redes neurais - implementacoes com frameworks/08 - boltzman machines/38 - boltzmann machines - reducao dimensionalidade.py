# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:45:16 2021

@author: santc
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits() # Base MNIST reduzida (dígitos escritos manualmente)
previsores = np.asarray(base.data, 'float32')
# 8px x 8px = 64px
classe = base.target

# Converte os preços de abertura para a escala de valores entre 0 e 1, isso
# dimui o custo de processamento
# Intervalo entre 0 e 1 (feature_range=(0,1))
normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

# test_size = 0.20 indica que serão utilizados 20% da quantidade total de 
# registros para realizar testes e o restante 80% para treinar
# nesse caso a base tem 1797 registros que serão divididos 
# 360 (20%) registros para teste e 1437 (80%) registros para treinamento
# classeTeste: saídas esperadas
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, test_size = 0.2, random_state=0)

rbm = BernoulliRBM(random_state = 0)
# Quantidade de épocas
rbm.n_iter = 25
# Quantidade de neurônios na camada oculta
# Recomenda-se a técnica de tuning para encontrar o valor mais adequado
rbm.n_components = 50
naiveRbm = GaussianNB()
# Executa mais de 1 processo de uma vez, nesse caso primeiro executa o rbm
# para fazer a redução de dimensionalidade e em seguida, pega os resultados gerados,
# envia para o naive
classificadorRbm = Pipeline(steps = [('rbm', rbm), ('naive', naiveRbm)])
classificadorRbm.fit(previsoresTreinamento, classeTreinamento)

# 20x20 é tamanho em pixels das imagens que serão exibidas
plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
	# 10x10 é o tamanho das imagens no console
    plt.subplot(10, 10, i + 1)
	# 8x8 é tamanho original, cmap em escala de cinza
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

previsoresRbm = classificadorRbm.predict(previsoresTeste)
precisaoRbm = metrics.accuracy_score(previsoresRbm, classeTeste)

naiveSimples = GaussianNB()
naiveSimples.fit(previsoresTreinamento, classeTreinamento)
previsoesNaive = naiveSimples.predict(previsoresTeste)
precisaoNaive = metrics.accuracy_score(previsoesNaive, classeTeste)