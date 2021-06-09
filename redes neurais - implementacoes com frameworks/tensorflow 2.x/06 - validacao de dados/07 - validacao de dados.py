# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:49:56 2021

@author: santc
"""

import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv

print("Tensorflow",tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Etapa 3: Análise simples da base de dados
dataset = pd.read_csv("pollution-small.csv")
dataset.head()
dataset.shape
training_data = dataset[:1600]
training_data.describe()
test_set = dataset[1600:]
test_set.describe()

## Etapa 4: Análise de dados e validação com TFDV
### Geração de estatísticas dos dados de treinamento
train_stats = tfdv.generate_statistics_from_dataframe(dataframe = training_data)
train_stats

### Inferindo o esquema
schema = tfdv.infer_schema(statistics = train_stats)
tfdv.display_schema(schema)

### Cálculo das estatísticas da base de teste
test_stats = tfdv.generate_statistics_from_dataframe(dataframe = test_set)

## Etapa 5: Comparação das estatísticas de teste com o esquema
### Checagem de anomalias nos novos dados
anomalies = tfdv.validate_statistics(statistics = test_stats, schema = schema)

### Mostrando as anomalias detectadas
# - Inteiros maiores do que 10
# - Esperava o tipo STRING mas a coluna estava com o tipo INT
# - Esperava o tipo FLOAT mas a coluna estava com o tipo INT
# - Inteiros menores do que 0
tfdv.display_anomalies(anomalies)

### Novos dados COM anomalias
test_set_copy = test_set.copy()
test_set_copy.drop("soot", axis = 1, inplace = True)
test_set_copy.describe()

### Estatísticas baseadas nos dados com anomalias

test_set_copy_stats = tfdv.generate_statistics_from_dataframe(dataframe = test_set_copy)

anomalies_new = tfdv.validate_statistics(statistics = test_set_copy_stats, schema = schema)

tfdv.display_anomalies(anomalies_new)

## Etapa 6: Preparação do esquema para produção (Serving)

schema.default_environment.append("TRAINING")
schema.default_environment.append("SERVING")

### Removendo a coluna alvo do esquema para produção

tfdv.get_feature(schema, "soot").not_in_environment.append("SERVING")

### Checando anomalias entre o ambiente em produção (Serving) e a nova base de teste

serving_env_anomalies = tfdv.validate_statistics(test_set_copy_stats, schema, environment = "SERVING")

tfdv.display_anomalies(serving_env_anomalies)

## Etapa 7: Salvando o esquema

tfdv.write_schema_text(schema = schema, output_path = "pollution_schema.pbtxt")
