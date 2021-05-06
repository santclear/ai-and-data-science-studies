# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:51:07 2021

@author: santc
"""

import numpy as numpy
from sklearn import datasets

def sigmoid(soma):
    return 1 / (1 + numpy.exp(-soma))

# Parte do cálculo da descida do gradiente
def sigmoidDerivada(sigmoid):
    return sigmoid * (1 - sigmoid)

base = datasets.load_breast_cancer()
entradasBase = base.data
saidasBase = base.target

tamanhoSaidasBase = len(saidasBase) # 569
saidasEsperadas = numpy.empty([tamanhoSaidasBase, 1], dtype = int)
for i in range(tamanhoSaidasBase): # 569
    saidasEsperadas[i] = saidasBase[i]

quantidadeAtributosEntradaBase = len(entradasBase[0]) # 30
# 3 quantidade de neurônios na camada oculta
pesosEntradaCamadaOculta = 2 * numpy.random.random((quantidadeAtributosEntradaBase,3)) - 1
# 3 idem anterios, deve ser igual
pesosSaidaCamadaOculta = 2 * numpy.random.random((3,1)) - 1

epocas = 1000000

# velocidade de deslocamento da descida do gradiente
taxaAprendizagem = 0.3

# objetivo de evitar mínimos locais na descida do gradiente 
momento = 1

for j in range(epocas):
    camadaEntrada = entradasBase
    
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
    print("Média absoluta do erro: "+ str(mediaAbsolutaErroCamadaSaida * 100))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    # Necessário fazer a transposta porque os 2 arrays não tem o mesmo tamanho
    pesosSaidaCamadaOcultaTransposta = pesosSaidaCamadaOculta.T
    deltaSaidaXPesoSaidaCamadaOculta = deltaSaida.dot(pesosSaidaCamadaOcultaTransposta)
    deltaCamadaOculta = deltaSaidaXPesoSaidaCamadaOculta * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    camadaOcultaXDeltaSaida = camadaOcultaTransposta.dot(deltaSaida)
    
    # Atualização de pesos
    pesosSaidaCamadaOculta = (pesosSaidaCamadaOculta * momento) + (camadaOcultaXDeltaSaida * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    camadaEntradaXDeltaCamadaOculta = camadaEntradaTransposta.dot(deltaCamadaOculta)
    
    pesosEntradaCamadaOculta = (pesosEntradaCamadaOculta * momento) + (camadaEntradaXDeltaCamadaOculta * taxaAprendizagem)
    
print("Resultado da previsão: ")
for k in range(len(saidasEsperadas)):
    camadaSaidaArredondada = round(camadaSaida[k][0])
    if(saidasEsperadas[k][0] == camadaSaidaArredondada):
        print("Acertou! Esperava: "+ str(saidasEsperadas[k][0]) +" | Encontrou: "+ str(camadaSaidaArredondada))
    else:
        print("Errou :-( Esperava: "+ str(saidasEsperadas[k][0]) +" | Encontrou: "+ str(camadaSaidaArredondada))