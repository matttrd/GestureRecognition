# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:57:15 2017

@author: Matteo
"""

import pandas as pd
import numpy as np
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Lambda
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
#from keras.callbacks import TensorBoard
import keras.backend as K
#import tensorflow as tf
from Norm_layer import NormalizationLayer
from MultichannelTimeLayers import TimeConvLayer, TimeMaxPooLayer, Rotout
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
import h5py
#from import_Opp_Dataset import df_train,df_test




#%%

# remove quaternions
quat = 'Quat'
cols = [col for col in df_train.columns if not quat in col]
task_c_col = 'ML_Both_Arms'


X_train = df_train[predictors_col].as_matrix()
X_test = df_test[predictdhtnors_col].as_matrix()


#%%
n_ch = X_train.shape[2]
steps =  X_train.shape[1]
n_classes = len(np.unique(y_train))
kernel_size1  = 20
strides1 = 2
kernel_size2 = 10
strides2 = 1
kernel_size3 = 5
strides3 = 1
pool_size1 = 2
pool_size2 = 2
pool_size3 = 4
nb_filter1 = 20
nb_filter2 = 10
nb_filter3 = 10


# first layer
mod
el_cnn = Sequential()
model_cnn.add(TimeConvLayer(filters = nb_filter1,strides= strides1,
                            kernel_size = kernel_size1, padding = 'causal', activation='relu', input_shape = (n_ch,steps,1), name = 'layer_1_conv'))

model_cnn.add(TimeMaxPooLayer(pool_size = pool_size1,name = 'layer_1_pool'))

model_cnn.add(GaussianNoise(0.01))
model_cnn.add(BatchNormalization())
#model_cnn.add(NormalizationLayer(name = 'layer_1_norm'))

# second layer
model_cnn.add(TimeConvLayer(filters = nb_filter2,strides= strides2,
                            kernel_size = kernel_size2, activation='relu', name = 'layer_2_conv'))
model_cnn.add(TimeMaxPooLayer(pool_size = pool_size2,name = 'layer_2_pool'))
#model_cnn.add(NormalizationLayer(name = 'layer_2_norm'))
model_cnn.add(BatchNormalization())

#third layer
model_cnn.add(TimeConvLayer(filters = nb_filter3,strides= strides3,
                            kernel_size = kernel_size3, activation='relu', name = 'layer_3_conv'))
model_cnn.add(TimeMaxPooLayer(pool_size = pool_size3, name = 'layer_3_pool'))
#model_cnn.add(NormalizationLayer(name = 'layer_3_norm'))

model_cnn.add(BatchNormalization())

model_cnn.add(Flatten())
model_cnn.add(Dense(units=100, activation='tanh'))
model_cnn.add(Activation('relu'))
#model_cnn.add(NormalizationLayer())
model_cnn.add(BatchNormalization())

model_cnn.add(Dense(units = n_classes, activation='sigmoid'))
model_cnn.summary()


#%% training
Y_train = to_categorical(y_train-1, num_classes=6)
X_train_reshaped = X_train[:,:,:,np.newaxis]
X_train_reshaped = np.swapaxes(X_train_reshaped, 1,2)
epochs = 50
with h5py.File('model_checkpoint.hdf5','w') as f:
    model_checkpoint = ModelCheckpoint(f.filename, period = 1)
csv_logger = CSVLogger('training.log')

model_cnn.compile(loss='categorical_crossentropy', 
              optimizer = 'Adam',
              metrics=['accuracy'])

model_cnn.fit(X_train_reshaped, Y_train, batch_size = 32, epochs = epochs, callbacks = [csv_logger,model_checkpoint])

#%% test
Y_test = to_categorical(y_test-1, num_classes=6)
X_test_reshaped = X_test[:,:,:,np.newaxis]
X_test_reshaped = np.swapaxes(X_test_reshaped, 1,2)
score = model_cnn.evaluate(X_test_reshaped, Y_test)