# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:52:10 2021

@author: santc
"""

import numpy as numpy

entradasNumpy = numpy.array([1, 7, 5])
pesosNumpy = numpy.array([0.8, 0.1, 0])

# dot: Multiplica elementos de mesmo indice entre 2 arrays e 
# soma cada um com as multiplicações subsequentes.
soma = entradasNumpy.dot(pesosNumpy)

# Função de ativação degrau
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

previsao = stepFunction(soma)