# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 02:15:22 2021

@author: santc
"""

import tensorflow as tf
import numpy as np

print('Tensorflow', tf.__version__)

### Constantes
# Definindo uma constante no TensorFlow 2.0
tensor_20 = tf.constant([[23, 4], [32, 51]])

tensor_20

# Acessando as dimensões de um tensor
tensor_20.shape

# Acessando os valores de um tensor com o numpy e sem precisar de uma sessão
tensor_20.numpy()

# Podemos converter um numpy array para um tensor do TensorFlow
numpy_tensor = np.array([[23,  4], [32, 51]])
tensor_from_numpy = tf.constant(numpy_tensor)
tensor_from_numpy



### Variáveis
#### Definindo uma variável
tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
tf2_variable

#### Acessando os valores de uma variável
tf2_variable.numpy()

#### Alterando um valor específico de uma variável
tf2_variable[0, 2].assign(100)
tf2_variable



### Operações com tensores
tensor = tf.constant([[1, 2], [3, 4]])
tensor
#### Adição entre um escalar e um tensor
tensor + 2
#### Multiplicação entre um escalar e um tensor
tensor * 5
#### Usando funções do numpy nos tensores do TensorFlow
# Obtendo o quadrado de todos os membros de um tensor
np.square(tensor)
# Obtendo a raiz quadrada de todos os membros de um tensor
np.sqrt(tensor)

#### Dot product (produto escalar) entre dois tensores
tensor
tensor_20
np.dot(tensor, tensor_20)



### Strings no TensorFlow 2.0
tf_string = tf.constant("TensorFlow")
tf_string

#### Operações simples com strings
tf.strings.length(tf_string)
tf.strings.unicode_decode(tf_string, "UTF8")

#### Armazenando arrays (vetores) de strings
tf_string_array = tf.constant(["TensorFlow", "Deep Learning", "AI"])

# Iterating through the TF string array
for string in tf_string_array:
  print(string)