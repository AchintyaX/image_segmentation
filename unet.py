import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 

from tensorflow_examples.models.pix2pix import pix2pix 

import tensorflow_datasets as tfds

from IPython.display import clear_output 


class unet_model :
	def __init__(self, OUTPUT_CHANNELS = 3):
		self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
		self.base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

		layer_names = [
			'block_1_expand_relu',   # 64x64
			'block_3_expand_relu',   # 32x32
			'block_6_expand_relu',   # 16x16
			'block_13_expand_relu',  # 8x8
			'block_16_project',      # 4x4
		]

		self.layers = [base_model.get_layer(name).output for name in layer_names]

		self.down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=self.layers)
		self.down_stack.trainable = False

		self.up_stack = [
			pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    		pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    		pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    		pix2pix.upsample(64, 3),   # 32x32 -> 64x64
		]

	def unet_model(self):
		inputs = tf.keras.layers.Input(shape=[128, 128, 3])
		x = inputs

		# Dowwsampling through the model 
		skips = self.down_stack(x)
		x = skips[-1]
		skips = reversed(skips[:-1])

		# upsampling and establishing the skip connections 
		for up, skip in zip(self.up_stack, skips):
			x = up(x)
			concat = tf.keras.layers.Concatenate()
			x = concat([x, skip])

		# This is the last layer of the moodel 
		last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 3, strides=2,
			padding='same') # 64X64 -> 128X128

		x = last(x)

		return tf.keras.Model(inputs=inputs, outputs=x)
		
