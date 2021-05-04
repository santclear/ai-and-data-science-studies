# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:07:32 2021

@author: santc
"""

import numpy as numpy

def sigmoid(soma):
    return 1 / (1 + numpy.exp(-soma))