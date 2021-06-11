# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 21:31:06 2021

@author: santc
"""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
	img_array = img_to_array(X_test[i])
	save_img(path = "uploads/{}.png".format(i), x = img_array)
	