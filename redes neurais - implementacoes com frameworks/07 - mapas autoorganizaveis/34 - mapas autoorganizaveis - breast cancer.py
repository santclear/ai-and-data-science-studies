# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:30:19 2021

@author: santc
"""

import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import pcolor, colorbar, plot

XBase = pd.read_csv('entradas_breast.csv')
yBase = pd.read_csv('saidas_breast.csv')
# X: Atributos previsores
X = XBase.iloc[:,0:30].values
# y: Classes
y = yBase.iloc[:,0].values
# Normalização dos dados (diminui a escala dos valores) para acelerar o 
# processamento dos valores no treinamento e teste
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# x: quantidade de linhas do mapa
# y: quantidade de colunas do mapa
# Total de 64 neurônios, 5 * 5 = 25
# O modelo usado para definir os valores de x e y é: 
# 1. 5√N onde N é igual ao número de registros que nesse caso é 178, pois len(base) = 178
# 2. 5√569 = 23,85 células então 5 * 5 = 25 (23,85 foi arredondado para 25)
# input_len: quantidade de entradas (para as entradas são usado os 13 atributos 
# previsores definidos na variável X)
# sigma: raio dos neurônios centróides (Best Match Units, B.M.U.), apartir desse raio
# que são calculados os neurônios em volta que fazem parte de cada grupo
# learning_rate: taxa de aprendizagem
# random_seed: mantém o mesmo resultado a cada execução
som = MiniSom(x = 11, y = 11, input_len = 30, sigma = 3.0, learning_rate = 0.5, random_seed = 0)
# Inicialização dos pesos
som.random_weights_init(X)
# Treinamento
som.train_random(data = X, num_iteration = 1000)

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

# w: Coordenadas(horizontal,vertical) do neurônio centróide (BMU) do 3º registro
w = som.winner(X[2])
# o: circulo -> r: vermelho
# s: quadrado -> g: verde
# D: quadrado rotacionado 90º -> b: azul
markers = ['o', 's']
color = ['r', 'g']
# O vetor color começa em 0, é necessário essa tranformação para associar as cores
# r, g e b aos markers
y[y == 1] = 0
y[y == 2] = 1

for i, x in enumerate(X):
	# w: Coordenadas(horizontal,vertical) do neurônio centróide (BMU) do 3º registro
    w = som.winner(x)
	
	# 0.5 posiciona cada figura no meio de cada box do mapa (coordenadas 
	#(horizontal, vertical) => (w[0], w[1]) )
	# markers[y[i]]: marcadores
	# markerfacecolor: Cor da fonte
	# markersize: tamanho do marcador
	# markeredgecolor: cor dos símbolos
	# markeredgewidth: borda
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)