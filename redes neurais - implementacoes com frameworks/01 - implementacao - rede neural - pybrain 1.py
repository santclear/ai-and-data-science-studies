# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:41:10 2021

@author: santc
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()
# LinearLayer não submete a fuções de ativação
camadaEntrada = LinearLayer(2)
# SigmoidLayer submete a função de ativação sigmoid
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print("-- Parâmetros --")
print("Rede: "+ str(rede))
print("Entrada camada oculta: "+ str(entradaOculta.params))
print("Saída camada oculta: "+ str(ocultaSaida.params))
print("Bias entrada camada oculta: "+ str(biasOculta.params))
print("Bias saída camada oculta: "+ str(biasSaida.params))