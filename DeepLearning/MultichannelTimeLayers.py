"""
Implementation of Conv Layer with independent time-channels 
"""


from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
import numpy as np

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
        
        #Input shape: (N_channels, steps, 1)
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

#    def get_config(self):
#        config = super(TimeConvLayer, self).get_config()
#        config.pop('rank')
#        config.pop('data_format')
#        return config

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
        super(TimeMaxPooLayer, self).__init__(MaxPooling1D(pool_size=pool_size, 
                 strides=strides, 
                 padding=padding
             ),**kwargs)
        self.input_spec = InputSpec(ndim=4)

#    def get_config(self):
#        config = super(TimeMaxPooLayer, self).get_config()
#        config.pop('rank')
#        config.pop('data_format')
#        return config

    def build(self, input_shape):
        super(TimeMaxPooLayer, self).build(input_shape)

    def call(self, x):
        return super(TimeMaxPooLayer, self).call(x)

    def compute_output_shape(self, input_shape):
        return  super(TimeMaxPooLayer, self).compute_output_shape(input_shape)
    
    
class Rotout(Layer):
   '''Apply a random rotation to the input with input shape (None, 3, steps, 1)
   
   # Returns
        A tensor, result of rotout
    '''
   def __init__(self, low = -np.pi,high = np.pi, **kwargs):
        super(Rotout, self).__init__(**kwargs)
        self._low = low
        self._high = high
        self.input_spec = InputSpec(ndim = 4)
        
   def build(self, input_shape):
        super(Rotout, self).build(input_shape)
         
    
   def call(self, x):
       '''
       x must be a (None, ch, steps, 1) tensor
       
       '''
       def get_rot_matrix(angle_x,angle_y,angle_z):
           Rx = np.array([[1, 0, 0],[0,np.cos(angle_x), -np.sin(angle_x)],[0, np.sin(angle_x), np.cos(angle_x)]])
           Ry = np.array([[np.cos(angle_y), 0, -np.sin(angle_y)],[0, 1, 0],[np.sin(angle_y), 0, np.cos(angle_y)]])
           Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],[np.sin(angle_z), np.cos(angle_z),0],[0, 0, 1]])
           R = K.tf.convert_to_tensor(np.dot(np.dot(Rx,Ry),Rz), dtype=K.tf.float32)
           return R
       
       def rotout(low,high, inp):
           #rotout, inp must be a vector with shape (steps,3)
           random_angle_x = np.random.uniform(low = low, high = high)
           random_angle_y = np.random.uniform(low = low, high = high)
           random_angle_z = np.random.uniform(low = low, high = high)
           R = get_rot_matrix(random_angle_x,random_angle_y,random_angle_z)
           return K.dot(R,inp)
       low = self._low
       high = self._high
       red_tensor = K.permute_dimensions(rotout(low, high, inp = x[:,:,:,0]), (1,0,2))
       return K.expand_dims(red_tensor, axis=-1)
     
    
   def compute_output_shape(self, input_shape):
       return input_shape
    
    
class TimeUpSamplingLayer(TimeDistributed):
    """Multi-channel 1D up-sampling along temporal axis: 
        # Returns
        A tensor, (batch, channels, upsampled_steps, features)
    """
    def __init__(self, 
                 size=2, 
                 **kwargs):
        super(TimeUpSamplingLayer, self).__init__(UpSampling1D(size = size)
             ,**kwargs)
        self.input_spec = InputSpec(ndim = 4)

#    def get_config(self):
#        config = super(TimeMaxPooLayer, self).get_config()
#        config.pop('rank')
#        config.pop('data_format')
#        return config

    def build(self, input_shape):
        super(TimeUpSamplingLayer, self).build(input_shape)

    def call(self, x):
        return super(TimeUpSamplingLayer, self).call(x)

    def compute_output_shape(self, input_shape):
        return  super(TimeUpSamplingLayer, self).compute_output_shape(input_shape) 
