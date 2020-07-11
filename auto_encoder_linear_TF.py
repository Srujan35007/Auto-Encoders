import tensorflow as tf 
from tensorflow.keras import models, datasets, layers
import numpy as np 
print('Imports complete')

bottle_neck_dims = 10
class Encoder_Decoder(layers.Layer):
    def __init__(self):
        super(Encoder_Decoder, self).__init__()
        self.flatten = layers.Flatten(input_shape = (28,28))
        self.enc_1 = layers.Dense(500, activation='relu')
        self.enc_2 = layers.Dense(200, activation='relu')
        self.enc_out = layers.Dense(bottle_neck_dims, activation='elu')
        self.dec_1 = layers.Dense(200, activation='relu')
        self.dec_2 = layers.Dense(500, activation='relu')
        self.dec_out = layers.Dense(28*28, activation='sigmoid')
        print('Encoder Decoder setup created')
    def call(self, x):
        x = self.enc_pass(x)
        x = self.dec_pass(x)
        return x
    def enc_pass(self, x):
        x = self.flatten(x)
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_out(x)
        return x
    def dec_pass(self, x):
        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_out(x)
        return x

net = Encoder_Decoder()
