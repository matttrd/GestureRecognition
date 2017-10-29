# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:07:23 2017

@author: Matteo
"""

from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import non_neg
class NormalizationLayer(Layer):
    '''
    Implementation of narmalization layer of
    Deep Convolutional Neural Networks On Multichannel Time Series
        For Human Activity Recognition
    '''
    def __init__(self, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.k = self.add_weight(name='k',
                                      shape=(),
                                      initializer='ones',
                                      trainable=True,
                                      constraint=non_neg())
        self.alpha = self.add_weight(name='alpha', 
                                      shape=(),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=non_neg())
        self.beta = self.add_weight(name='beta', 
                                      shape=(),
                                      initializer='uniform',
                                      trainable=True)
        super(NormalizationLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        rescaling = (self.k + self.alpha * K.tf.norm(x))**self.beta
        return x / rescaling

    def compute_output_shape(self, input_shape):
        return input_shape