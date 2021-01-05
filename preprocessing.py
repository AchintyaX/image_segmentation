import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 

from tensorflow_examples.models.pix2pix import pix2pix 

import tensorflow_datasets as tfds

from IPython.display import clear_output 


class pre_preocessing:
	def __init__(self, datapoint):
		self.datapoint = datapoint

	def normalize(self, input_image, input_mask):
		input_image = tf.cast(input_imagex, tf.float32) /255.0
		input_mask -= 1

		return input_image, input_mask

	@tf.function 
	def load_image_train(self):
		input_image = tf.image.resize(self.datapoint['image'], (128,128))
		input_mask = tf.image.resize((self.datapoint['segmentation_mask']), (128,128))

		if tf.random.uniform(()) > 0.5 :
			input_image = tf.image.flip_left_right((input_image))
			input_mask = tf.image.flip_left_right(input_mask)

		input_image, input_mask = self.normalize(input_image, input_mask)

		return input_image, input_mask

	def load_image_test(self):
		input_image = tf.image.resize(self.datapoint['image'] , (128,128))
		input_mask = tf.image.resize(self.datapoint['segmentation_mask'] , (128, 128))

		input_image, input_mask = self.normalize(input_image, input_mask)

		return input_image, input_mask

	def display(self, display_list):
		plt.figure(figsize=(15,15))

		title = ['Input Image', 'True Mask', 'Predicted Mask']

		for i in range(len(display_list)):
			plt.subplot(1,len(display_list), i+1)
			plt.title(title[i])
			plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
			plt.axis('off')
		plt.show()
