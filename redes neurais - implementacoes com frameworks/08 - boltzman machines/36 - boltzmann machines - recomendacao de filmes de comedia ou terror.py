# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:41:43 2021

@author: santc
"""

from rbm import RBM
import numpy as np

# num_visible: Nós visíveis (equivalente a entradas)
# num_hidden: Quantidade neurônios da camada oculta
rbm = RBM(num_visible = 6, num_hidden = 2)

# Cada registro da matriz indica se o usuário assistiu ou não um determinado filme
base = np.array([[1,1,1,0,0,0],# Esse usuário assistiu todos os filmes de terror e nenhum de comédia
                 [1,0,1,0,0,0],# Esse usuário assistiu 2 filmes de terror e nenhum de comédia
                 [1,1,1,0,0,0],# Esse usuário assistiu todos os filmes de terror e nenhum de comédia
                 [0,0,1,1,1,1],# Esse usuário assistiu 3 filmes de comédia e 1 de terror
                 [0,0,1,1,0,1],# Esse usuário assistiu 2 filmes de comédia e 1 de terror
                 [0,0,1,1,0,1]])# Esse usuário assistiu 2 filmes de comédia e 1 de terror

# Treinamento
rbm.train(base, max_epochs=5000)
# Primeira coluna e linha são os valores da unidade de bias, os demais valores 
# das demais colunas e linhas representam 1 determinado filme
# 2ª coluna = terror
# 3ª coluna = comédia
# Da 2ª linha até a final representam os filmes 
# "A bruxa","Invocação do mal",... e "American pie"
# Neurônios com maior valor, significa que está ativado
rbm.weights