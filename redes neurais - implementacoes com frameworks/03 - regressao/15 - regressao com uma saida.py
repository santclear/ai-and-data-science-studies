# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:49:32 2021

@author: santc
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

# Base obtida do site https://www.kaggle.com/
# 371528 registros (linhas - veículos cadastrados)
# 20 atributos (colunas)
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

################################################
### PREPROCESSAMENTO: Exclusão de colunas que não serão necessárias no treinamento e teste,
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

################################################
### PREPROCESSAMENTO: Exclusão de valores inconsistentes de alguns registros
### Obs.: Isso deve ser feito com cautela, pois a base deve ter um número
### relativamente aceitável para um bom treinamento e teste
# loc: localiza registros com base em parâmetros, nesse caso registros em que
# o preço na coluna price seja menor ou igual a 10
# nessa etapa é possível verificar que tem carros com valores baixos que não
# fazem sentido, essa busca é apenas para constatar isso
i1 = base.loc[base.price <= 10]
# Em algumas situações é possível obter a média e aplicar nos registros inconsistentes
# nesse caso, não será feito isso, a função deixo aqui apenas para ilustrar
# a possibilidade para alguns casos.
base.price.mean()
# Para esse caso será separado os registros com preços superiores a 10
base = base [base.price > 10]
# Aqui também é possível constatar que tem mais valores inconsistentes, esses
# valores estão acima de 350000, provavelmente foi algum erro no cadastro ou
# na coleta do crawler
i2 = base.loc[base.price > 350000]
# Desse modo será separado os registros com preços inferiores a 350000
base = base.loc[base.price < 350000]
################################################

################################################
### PREPROCESSAMENTO: Substituição de valores nulos por valores que tem maior
### número de ocorrências
### Obs.: Uma estratégia de tratamento de dados que pode ser usada para evitar
### a estratégia anterior de exclusão de registros, desse modo a quantidade de 
### registros da base se mantém o mesmo
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # maior: limousine 93627
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # maior: manuell 266612
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # maior: golf 28998
base.loc[pd.isnull(base['fuelType'])] 
base['fuelType'].value_counts() # maior: benzin 217648
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # maior: nein 259366

valores = {'vehicleType':'limousine',
           'gearbox':'manuell',
           'model':'golf',
           'fuelType':'benzin',
           'notRepairedDamage':'nein'}

base = base.fillna(value = valores)
################################################

# Todas a linhas(:) e colunas de 1 até 12 (1:13), 13 é o limite!
previsores = base.iloc[:, 1:13].values
# Como não é problema de classificação, a variável é definida com o nome de um
# valor que se deseja prever através de aproximação numérica.
# Nesse caso é desejado prever o preço real dado um conjunto de parâmetros
precoReal = base.iloc[:,0].values

################################################
### PREPROCESSAMENTO: Converte dados categóricos em numéricos
labelEncoderPrevisores = LabelEncoder()
previsores[:, 0] = labelEncoderPrevisores.fit_transform(previsores[:,0])
previsores[:, 1] = labelEncoderPrevisores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoderPrevisores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoderPrevisores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelEncoderPrevisores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoderPrevisores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelEncoderPrevisores.fit_transform(previsores[:, 10])

################################################
### PREPROCESSAMENTO: Converte dados numéricos em Onehot Encoder
# Transforma as categorias 0, 1, 3, 5, 8, 9, 10 em algo parecido com:
# 0: 1 0 0 ... 0 
# 1: 0 1 0 ... 0
# 3: 0 0 1 ... 0
# ... ...
# 10: ...
# onehot ecoder pode ser usado quando não há uma ordem de importância entre 
# os atributos, por exemplo o tipo de câmbio não pode ser afirmado que "manual 
# é maior que automático" e vice-versa
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# Como não é problema de classificação, a variável é definida com o nome de um
# substantivo que reflete a estratégia utilizada pela rede neural para realizar 
# a previsão.
# Nesse caso estratégia é de regressão
regressor = Sequential()
### CRIAÇÃO DA CAMADA OCULTA E DEFINIÇÃO DA CAMADA DE ENTRADA
# units: quantidade de neurônios da camada oculta. 158 escolhido com base no
# modelo (316(entradas) + 1(saída)) / 2 = 158
# activation: função de ativação
# input_dim: quantidade de atributos da camada de entrada. Nesse caso são 4 porque o dataset possuí 4 colunas. (atributos)
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
### CRIAÇÃO DA CAMADA DE SAÍDA
# 1 neurônio na camada de saída, como é um problema de regressão, não é necessário 
# o uso de função de ativação, pois o objetivo é prever um número e não uma 
# probabilidade
regressor.add(Dense(units = 1, activation = 'linear'))
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

# Treinamento
regressor.fit(previsores, precoReal, batch_size = 300, epochs = 100)
# Previsão
previsoes = regressor.predict(previsores)

precoReal.mean()
previsoes.mean()