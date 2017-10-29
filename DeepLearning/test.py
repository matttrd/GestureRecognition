# -*- coding: utf-8 -*-
"""
Activity Recognition on UCI HAR Dataset
Multichannel Conv Net: each channel (axis) is convolved independently and finally flattened in a encoding layers which is followed by a Dense net
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
#%% import dataset real dataset
source = 'D:\\DEEP_TIME_SERIES\\data\\UCI HAR Dataset\\'
def get_acc(bodyOrtotal, trainOrtest):
    assert trainOrtest in {'test','train'}
    assert bodyOrtotal in {'body','total'}
    
    acc_x = pd.read_csv(source + trainOrtest + '\\Inertial Signals\\' + bodyOrtotal + '_acc_x_' + trainOrtest + '.txt', \
                    delimiter = '\t', header = -1)[0].apply(lambda x: np.fromstring(x, sep = ' ')).to_frame("acc_x")
    acc_y = pd.read_csv(source + trainOrtest + '\\Inertial Signals\\' + bodyOrtotal + '_acc_y_' + trainOrtest + '.txt', \
                    delimiter = '\t', header = -1)[0].apply(lambda x: np.fromstring(x, sep = ' ')).to_frame("acc_y")
    acc_z = pd.read_csv(source + trainOrtest + '\\Inertial Signals\\' + bodyOrtotal + '_acc_z_' + trainOrtest + '.txt', \
                    delimiter = '\t', header = -1)[0].apply(lambda x: np.fromstring(x, sep = ' ')).to_frame("acc_z")
    return acc_x, acc_y, acc_z

#acc_x, acc_y, acc_z = get_acc('body','train')
acc_x, acc_y, acc_z = get_acc('total','train')

train_df = pd.concat([acc_x,acc_y,acc_z],axis=1).iloc[1:]

#acc_x, acc_y, acc_z = get_acc('body','test')
acc_x, acc_y, acc_z = get_acc('total','test')
test_df = pd.concat([acc_x,acc_y,acc_z],axis=1).iloc[1:]


def recur_append(x):
    res = np.array([])
    for i in range(len(x)):
        res = np.append(res, x[i])
    res = res.reshape((int(len(res)/3),3))
    #res = res.reshape((3,int(len(res)/3)))
    return res

X_train = train_df.as_matrix()
X_train = np.array(list(map(lambda x: recur_append(x), X_train)))
X_test = test_df.as_matrix()
X_test = np.array(list(map(lambda x: recur_append(x), X_test)))

y_train = pd.read_csv(source + 'train\\y_train.txt').as_matrix()
y_test = pd.read_csv(source + 'test\\y_test.txt').as_matrix()


#%% 

#time-series length
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
model_cnn = Sequential()
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
epochs = 100
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

#%%
from scipy.ndimage import rotate
angle_x = 20*np.pi/180.0
angle_y = 20*np.pi/180.0
angle_z = 20*np.pi/180.0

def get_rot_matrix(angle_x,angle_y,angle_z):
    Rx = np.array([[1, 0, 0],[0,np.cos(angle_x), -np.sin(angle_x)],[0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, -np.sin(angle_y)],[0, 1, 0],[np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],[np.sin(angle_z), np.cos(angle_z),0],[0, 0, 1]])
    R = np.dot(np.dot(Rx,Ry),Rz)
    return R

#X_train_rot = np.array([np.dot(R,np.transpose(X_train[i,:,:])) for i in range(X_train.shape[0])])
#X_train_rot = np.swapaxes(X_train_rot, 1,2)
R = get_rot_matrix(angle_x,angle_y,angle_z)
X_test_rot = np.array([np.dot(R,np.transpose(X_test[i,:,:])) for i in range(X_test.shape[0])])
X_test_rot = np.swapaxes(X_test_rot, 1,2)

#%% test on rotated
Y_test = to_categorical(y_test-1, num_classes=6)
X_test_reshaped = X_test_rot[:,:,:,np.newaxis]
X_test_reshaped = np.swapaxes(X_test_reshaped, 1,2)
score = model_cnn.evaluate(X_test_reshaped, Y_test)
print(score)

#%%

model_cnn_rot = Sequential()
model_cnn_rot.add(Rotout(low = 0, high = np.pi/6, input_shape = (3,steps,1)))

model_cnn_rot.add(TimeConvLayer(filters = nb_filter1,strides= strides1,
                            kernel_size = kernel_size1, activation='relu', input_shape = (n_ch,steps,1), name = 'layer_1_conv'))
model_cnn_rot.add(TimeMaxPooLayer(pool_size = pool_size1,name = 'layer_1_pool'))
model_cnn.add(GaussianNoise(0.01))
model_cnn_rot.add(BatchNormalization())
#model_cnn_rot.add(NormalizationLayer(name = 'layer_1_norm'))

# second layer
model_cnn_rot.add(TimeConvLayer(filters = nb_filter2,strides= strides2,
                            kernel_size = kernel_size2, activation='relu', name = 'layer_2_conv'))
model_cnn_rot.add(TimeMaxPooLayer(pool_size = pool_size2,name = 'layer_2_pool'))
#model_cnn_rot.add(NormalizationLayer(name = 'layer_2_norm'))
model_cnn_rot.add(BatchNormalization())

#third layer
model_cnn_rot.add(TimeConvLayer(filters = nb_filter3,strides= strides3,
                            kernel_size = kernel_size3, activation='relu', name = 'layer_3_conv'))
model_cnn_rot.add(TimeMaxPooLayer(pool_size = pool_size3, name = 'layer_3_pool'))
#model_cnn_rot.add(NormalizationLayer(name = 'layer_3_norm'))

model_cnn_rot.add(BatchNormalization())

model_cnn_rot.add(Flatten())
model_cnn_rot.add(Dense(units=100, activation='tanh'))
model_cnn_rot.add(Activation('relu'))
#model_cnn_rot.add(NormalizationLayer())
model_cnn_rot.add(BatchNormalization())

model_cnn_rot.add(Dense(units = n_classes, activation='sigmoid'))
model_cnn_rot.summary()

#%% training rotout
Y_train = to_categorical(y_train-1, num_classes=6)
X_train_reshaped = X_train[:,:,:,np.newaxis]
X_train_reshaped = np.swapaxes(X_train_reshaped, 1,2)
epochs = 100
with h5py.File('model_checkpoint_rotout.hdf5','w') as f:
    model_checkpoint = ModelCheckpoint(f.filename, period = 5)
csv_logger = CSVLogger('training_rotout.log')

model_cnn_rot.compile(loss='categorical_crossentropy', 
              optimizer = 'Adam',
              metrics=['accuracy'])

model_cnn_rot.fit(X_train_reshaped, Y_train, batch_size = 32, epochs = epochs, callbacks = [csv_logger,model_checkpoint])


#%%
score_rot = model_cnn_rot.evaluate(X_test_reshaped, Y_test)
print(score_rot)
y_pred = model_cnn_rot.predict_classes(X_test_reshaped)
from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test,y_pred + 1)

