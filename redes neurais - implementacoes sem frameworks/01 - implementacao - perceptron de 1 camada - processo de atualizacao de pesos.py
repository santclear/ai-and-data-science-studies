# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:52:10 2021

@author: santc
"""

entradasArray = [-1, 7, 5]
pesosArray = [0.8, 0.1, 0]

def soma(entradasArray, pesosArray):
    soma = 0
    # Quantidade de entrada deve ser IGUAL quantidade de pesos
    if(len(entradasArray) == len(pesosArray)):
        # Soma todas as entradas com seus respectivos pesos
        for i in range(len(entradasArray)):
            soma += entradasArray[i] * pesosArray[i]
        return soma
        
soma = soma(entradasArray, pesosArray)

# Função de ativação degrau
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

previsao = stepFunction(soma)