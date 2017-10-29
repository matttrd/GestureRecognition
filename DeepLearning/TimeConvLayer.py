# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 12:35:44 2017

@author: Matteo
Implentation of tim
"""


from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
#from keras import backend as K
#from keras.engine.topology import Layer
from keras.engine import InputSpec

class TimeConvLayer(TimeDistributed):
    """Multi-channel 1D convolution: each channel is treated separately 
            but with parameter sharing through channels
            
        # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilate rate.
        
        # Returns
        A tensor, result of Multi-channel 1D convolution.
    """
    def __init__(self, 
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TimeConvLayer, self).__init__(Conv1D(filters = filters, 
                 kernel_size = kernel_size,
                 strides=strides,
                 padding=padding,
                 dilation_rate=dilation_rate,
                 activation=activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint,
             
             ),**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(TimeConvLayer, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

    def build(self, input_shape):
        super(TimeConvLayer, self).build(input_shape)

    def call(self, x):
        return super(TimeConvLayer, self).call(x)

    def compute_output_shape(self, input_shape):
        return  super(TimeConvLayer, self).compute_output_shape(input_shape)
    
    

class TimeMaxPooLayer(TimeDistributed):
    """Multi-channel 1D pooling along temporal axis: 
        # Returns
        A tensor, result of Multi-channel 1D pooling.
    """
    def __init__(self, 
                 pool_size=2, 
                 strides=None, 
                 padding='valid',
                 **kwargs):
        super(TimeConvLayer, self).__init__(MaxPooling1D(pool_size=2, 
                 strides=None, 
                 padding='valid'
             ),**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(TimeConvLayer, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

    def build(self, input_shape):
        super(TimeConvLayer, self).build(input_shape)

    def call(self, x):
        return super(TimeConvLayer, self).call(x)

    def compute_output_shape(self, input_shape):
        return  super(TimeConvLayer, self).compute_output_shape(input_shape)