# -*- coding: utf-8 -*-
"""
Created on Tue May 25 18:43:25 2021

@author: santc
"""
import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import pcolor, colorbar, plot

base = pd.read_csv('wines.csv')
# X: Atributos previsores
# Todas as linhas ":" e colunas de 1 até 13 (Atributos: Alcohol até Proline)
X = base.iloc[:,1:14].values
# y: Classes
# Todas as linhas e coluna 0 (Atributo: Class)
y = base.iloc[:,0].values
# Normalização dos dados (diminui a escala dos valores) para acelerar o 
# processamento dos valores no treinamento e teste
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# x: quantidade de linhas do mapa
# y: quantidade de colunas do mapa
# Total de 64 neurônios, 8 * 8 = 64
# O modelo usado para definir os valores de x e y é: 
# 1. 5√N onde N é igual ao número de registros que nesse caso é 178, pois len(base) = 178
# 2. 5√178 = 65,65 células então 8 * 8 = 64 (65,65 foi arredondado para 64)
# input_len: quantidade de entradas (para as entradas são usado os 13 atributos 
# previsores definidos na variável X)
# sigma: raio dos neurônios centróides (Best Match Units, B.M.U.), apartir desse raio
# que são calculados os neurônios em volta que fazem parte de cada grupo
# learning_rate: taxa de aprendizagem
# random_seed: mantém o mesmo resultado a cada execução
som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2)
# Inicialização dos pesos
som.random_weights_init(X)
# Treinamento
som.train_random(data = X, num_iteration = 100)

# Mostra no console os pesos
som._weights
# Mostra no console os valores do mapa auto organizável
som._activation_map
# q: variável para observação no "Explorador de variáveis" das quantidades de vezes
# que cada neurônio de saída foi ativado (selecionado)
q = som.activation_response(X)

# MID - mean inter neuron distance (média euclidiana entre os neurônios vizinhos)
# Matriz transposta (T) com os valores de distância entre os neurônios
pcolor(som.distance_map().T)
# Gera um gráfico em forma de mapa com uma escala de cores, quanto mais
# próximo do amarelo (1) mais diferente o neurônio é de seus vizinhos e sendo assim
# a distância entre ele e seus vizinhos próximos é grande tornando-o pouco confiável
colorbar()
