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

# Base de treinamento
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

# Dados para teste de previsão
# Usuário assistiu 2 filmes de terror e 1 de comédia
usuario1 = np.array([[1,1,0,1,0,0]])
# Usuário assistiu 2 filmes de comédia e nenhum de terror
usuario2 = np.array([[0,0,0,1,1,0]])

# Retorna qual dos neurônimos foram ativados
# Índice 0 comédia e índice 1 terror
rbm.run_visible(usuario1)
rbm.run_visible(usuario2)

# Retorna os neurônios que estão ativados na camada oculta para o usuário 2
camada_escondida = np.array([[1,0]])
# Com base na ativação dos neurônios da camada oculta, retorna uma recomendação
recomendacao = rbm.run_hidden(camada_escondida)

filmes = ["A bruxa", "Invocação do mal", "O chamado", "Se beber não case", "Gente grande", "American pie"]
for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
	# Se o usuario2 não assistiu (0) e recomendacao for 1 então recomenda
    if usuario2[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
    