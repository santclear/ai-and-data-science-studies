# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:31:13 2021

@author: santc
"""

import numpy as numpy

entradas = numpy.array([[0,0], [0,1], [1,0], [1,1]])
saidas = numpy.array([0, 0, 0, 1])

pesos = numpy.array([0.0, 0.0])
taxaAprendizagem = 0.1

# Função de ativação degrau
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    # dot: Multiplica elementos de mesmo indice entre 2 arrays e 
    # soma cada um com as multiplicações subsequentes.
    soma = registro.dot(pesos)
    return stepFunction(soma)

def treinar():
    erroTotal = 1
    while(erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(numpy.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('Peso atualizado: '+ str(pesos[j]))
        print('Total de erros: '+ str(erroTotal))
        
treinar()

print("Teste de previsão x1 = 0 e x2 = 0 :: ", calculaSaida(entradas[0]))
print("Teste de previsão x1 = 0 e x2 = 1 :: ", calculaSaida(entradas[1]))
print("Teste de previsão x1 = 1 e x2 = 0 :: ", calculaSaida(entradas[2]))
print("Teste de previsão x1 = 1 e x2 = 1 :: ", calculaSaida(entradas[3]))