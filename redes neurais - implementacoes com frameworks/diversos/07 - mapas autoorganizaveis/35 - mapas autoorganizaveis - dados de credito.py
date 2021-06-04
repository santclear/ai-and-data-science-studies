# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:32:32 2021

@author: santc
"""

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna()

# Para os registros age(idade) menores que zero, serão substituídos pelo valor
# retorna pela média de idades 40.927688881035856
base.age.mean()
base.loc[base.age < 0, 'age'] = 40.92

# X: Atributos previsores
# Todas as linhas ":" e colunas de 0 até 3 (Atributos: clientid até loan)
X = base.iloc[:, 0:4].values
# y: Classes
# Todas as linhas e coluna 4 (Atributo: default)
y = base.iloc[:, 4].values

# Normalização dos dados (diminui a escala dos valores) para acelerar o 
# processamento dos valores no treinamento e teste
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# x: quantidade de linhas do mapa
# y: quantidade de colunas do mapa
# Total de 225 neurônios, 15 * 15 = 225
# O modelo usado para definir os valores de x e y é: 
# 1. 5*√N onde N é igual ao número de registros que nesse caso é 1997, pois len(base) = 1997
# 2. 5*√1997 = 5*(44,69) = 223,44 ===> √223,44 = 14,95 (aproximadamente 15)
# input_len: quantidade de entradas (para as entradas são usado os 13 atributos 
# previsores definidos na variável X)
# sigma: raio dos neurônios centróides (Best Match Units, B.M.U.), apartir desse raio
# que são calculados os neurônios em volta que fazem parte de cada grupo
# learning_rate: taxa de aprendizagem
# random_seed: mantém o mesmo resultado a cada execução
som = MiniSom(x = 15, y = 15, input_len = 4, random_seed = 0)
# Inicialização dos pesos
som.random_weights_init(X)
# Treinamento
som.train_random(data = X, num_iteration = 100)
# MID - mean inter neuron distance (média euclidiana entre os neurônios vizinhos)
# Matriz transposta (T) com os valores de distância entre os neurônios
pcolor(som.distance_map().T)
# Gera um gráfico em forma de mapa com uma escala de cores, quanto mais
# próximo do amarelo (1) mais diferente o neurônio é de seus vizinhos e sendo assim
# a distância entre ele e seus vizinhos próximos é grande tornando-o pouco confiável
colorbar()
# o: circulo -> w: branco (Crédito aprovado)
# s: quadrado -> r: verde (Crédito não aprovado)
markers = ['o', 's']
colors = ['w', 'r']
# O vetor color começa em 0, é necessário essa tranformação para associar as cores
# w e r aos markers

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)
    
mapeamento = som.win_map(X)
suspeitos = np.concatenate((mapeamento[(13,9)], mapeamento[(1,10)]), axis = 0)
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []
for i in range(len(base)):
    for j in range(len(suspeitos)):
       if base.iloc[i, 0] == int(round(suspeitos[j,0])):
           classe.append(base.iloc[i,4])
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]