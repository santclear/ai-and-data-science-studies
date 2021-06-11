# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:00:29 2021

@author: santc
"""

# Etapa 1: Importação das bibliotecas
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, jsonify

print(tf.__version__)

# Etapa 2: Carregamento do modelo pré-treinado
with open("fashion_model_flask.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("fashion_model_flask.h5")
model.summary()

# Etapa 3: Criação da API em Flask
app = Flask(__name__)

# Função para classificação de imagens
@app.route("/<string:img_name>", methods = ["POST"])
def classify_image(img_name):
	#img_name = '0.png'
	upload_dir = "uploads/"
	image = load_img(upload_dir + img_name)
    
	classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # [1, 28, 28] -> [1, 784]
	image = img_to_array(image)
	image_reshape = image.reshape(3, 28 * 28)
	prediction = model.predict([image_reshape])
    
	return jsonify({"object_identified": classes[np.argmax(prediction[0])]})

# Iniciando a aplicação Flask
app.run(port = 8081, debug = False)   
