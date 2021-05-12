# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:49:32 2021

@author: santc
"""

import pandas as pd

# Base obtida do site https://www.kaggle.com/
# 371528 registros (linhas - veículos cadastrados)
# 20 atributos (colunas)
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

################################################
### Exclusão de colunas que não serão necessárias no treinamento e teste,
### após esse processamento a base terá 12 atributos (colunas)

base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)


# Dados desbalanceados 
# Existem muitos carros com apenas 1 ocorrência, isso ocorreu porque os dados
# foram coletados de um crawler, onde usuários cadastraram sem padrão de nome,
# essa falta de padrão poderá gerar problema no treinamento e teste,
# esse atributo deve ser excluído
base['name'].value_counts() # para analisar desbalanceamento
base = base.drop('name', axis = 1)

# Dados desbalanceados 
# A maioria dos veículos vendidos é privado (pravat) e comercial (gewerblich) somente 3, 
# como a variabilidade entre privado e comercial é baixa, esse atributo deve 
# ser excluído porque a rede dificilmente classificará corretamente venda de 
# carros comerciais
base['seller'].value_counts() # para analisar desbalanceamento
base = base.drop('seller', axis = 1)

# Dados desbalanceados 
# Excluído por apresentar o mesmo problema anterior
base['offerType'].value_counts() # para analisar desbalanceamento
base = base.drop('offerType', axis = 1)
################################################