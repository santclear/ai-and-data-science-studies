# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:41:10 2021

@author: santc
"""

import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

# 2 neurônios na camada de entrada
# 3 neurônios na camada oculta
# 1 neurônio na camada de saída
# outclass camada de saída com função de ativação Softmax
# hiddenclass camada oculta com função de ativação Sigmoid
# bias por default a rede é criada com unidades de bias, configurado como false para demonstrar que é possível retirar
#rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer, hiddenclass = SigmoidLayer, bias = False)

rede = buildNetwork(2, 3, 1)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])

# Criação da base de dados de treinamento, para esse exemplo, operador XOR
base = SupervisedDataSet(2, 1)
# Primeiros parâmetros são as entradas e o segundo a saída esperada
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

print(base['input'])
print(base['target'])

treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01, momentum = 0.06)

for i in range(1, 30000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro: %s" % erro)
        
print(np.round(rede.activate([0, 0])))
print(np.round(rede.activate([1, 0])))
print(np.round(rede.activate([0, 1])))
print(np.round(rede.activate([1, 1])))

"""from pybrain.structure import FeedForwardNetwork
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
print("Bias saída camada oculta: "+ str(biasSaida.params))"""