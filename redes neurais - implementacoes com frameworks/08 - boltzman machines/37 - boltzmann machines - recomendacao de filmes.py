# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:36:15 2021

@author: santc
"""

from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 3)


baseTreinamento = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])

filmes = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek", 
          "Exterminador do Futuro", "Norbit", "Star Wars"]

rbm.train(baseTreinamento, max_epochs = 5000) 

leonardo = np.array([[0,1,0,1,0,0]]) 

camadaOculta = rbm.run_visible(leonardo)

recomendacao = rbm.run_hidden(camadaOculta)
for i in range(len(leonardo[0])):
    if leonardo[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])