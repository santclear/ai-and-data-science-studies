# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:07:32 2021

@author: santc
"""

import numpy as numpy

def sigmoid(soma):
    return 1 / (1 + numpy.exp(-soma))

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