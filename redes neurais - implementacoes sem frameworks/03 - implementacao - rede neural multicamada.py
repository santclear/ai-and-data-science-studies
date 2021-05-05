# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:07:32 2021

@author: santc
"""

import numpy as numpy

def sigmoid(soma):
    return 1 / (1 + numpy.exp(-soma))

# Parte do cálculo da descida do gradiente
def sigmoidDerivada(sigmoid):
    return sigmoid * (1 - sigmoid)

entradas = numpy.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

saidasEsperadas = numpy.array([
    [0],
    [1],
    [1],
    [0]])

pesosEntradaCamadaOculta = numpy.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469]])
    
pesosSaidaCamadaOculta = numpy.array([
    [-0.017],
    [-0.893],
    [0.148]])

epocas = 100

for j in range(epocas):
    camadaEntrada = entradas
    
    """ Como está sendo calculado:
    0 --> (0 * -424) + (1 * 0,358) = 0,358 ... (0 * -0,740) + (1 * -0,577) = -0,577 ...
    1 _/
    ...
    1 --> ...
    0 _/
    ...
    """
    somasSinapse0 = numpy.dot(camadaEntrada, pesosEntradaCamadaOculta)
    camadaOculta = sigmoid(somasSinapse0)
    
    somaSinapse1 = numpy.dot(camadaOculta, pesosSaidaCamadaOculta)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidasEsperadas - camadaSaida
    mediaAbsolutaErroCamadaSaida = numpy.mean(numpy.abs(erroCamadaSaida))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    # Necessário fazer a transposta porque os 2 arrays não tem o mesmo tamanho
    pesosSaidaCamadaOcultaTransposta = pesosSaidaCamadaOculta.T
    deltaSaidaXPesoSaidaCamadaOculta = deltaSaida.dot(pesosSaidaCamadaOcultaTransposta)
    deltaCamadaOculta = deltaSaidaXPesoSaidaCamadaOculta * sigmoidDerivada(camadaOculta)