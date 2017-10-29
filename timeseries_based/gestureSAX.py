# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:22:28 2016

@author: matttrd
"""
#import sys
#sys.path.append('/Users/matt/Documents/Utils/python/pysax-master')

import pandas as pd
import numpy as np
from scipy.ndimage import filters as f
import matplotlib.pyplot as plt
import pyAtomicGesture as at
import mvSAX as mv
#%% take input signals
#source_dA = '/Users/matt/Documents/dottorato/ActivityClassifi/gesture_recognition/data/DatasetA/DatasetA.json'
#source_dA = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\DatasetA\\DatasetA.json'
#source_dA = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\DB.json'

# go through pandas (not necessary)
#datasetA  = pd.read_json(source_dA).rename(columns = {"data": "data", "gesture": "ges", "types":"types"})
#datasetA['data'] = datasetA['data'].apply(lambda x: np.array(x))
#dataset = dataset_1[dataset_1.subId != 'AA03']

def get_dataset(name):
    if name == 'XC':
        source_dA = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\DB.json'
        dataset = pd.read_json(source_dA)
        dataset.rename(columns = {'subId':'User'}, inplace = True)

    if name == 'HAR':
        source_dA = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\HAR.json'
        dataset = pd.read_json(source_dA)
        dataset.Acceleration = dataset.Acceleration.apply(lambda x: np.array(x))
        dataset.rename(columns = {'Acceleration':'input'}, inplace = True)
        dataset.rename(columns = {'Activity':'output'}, inplace = True)
    return dataset

dataset = get_dataset('HAR')
N_user = 29
dataset = dataset[dataset.User.isin(range(0,N_user + 1))].reset_index(drop = True)
#dataset = dataset[dataset.User.isin(['AA12'])].reset_index(drop = True)

#%% functions want in input ndim arrays MxN where M is the dimensionality and N the number of samples
signals = np.array(map(lambda x: np.array(x),dataset['input'])).T
label = dataset['output']
user = dataset['User']
del dataset
#%% atomic filter
sigma = 1
N = 150
filt_sig = np.apply_along_axis(lambda x: f.gaussian_filter1d(x,sigma), 1, signals[0:3])
filt_sig = at.normalize(filt_sig,'mean_std')
#filt_sig2 = np.apply_along_axis(lambda x: f.gaussian_filter1d(x,25), 1, signals)


# har -> sigma = 8, XC =
at_gest = at.getMultiVariateAtRepGestsFromTS(filt_sig,isNormalized = False, fs = None, \
    delta = 0.01, sigma = 8, sampling_ratio = None, T_gesture = N,\
    resampling_type = 'Akima',method = 1,peak_method= 2, extr_method = 'connected')

at_gest = at.assignLabels(atom_df = at_gest, y = label)
at_gest = at.assignUsers(atom_df = at_gest, u = user)


#at_gest = at_gest[at_gest['label'] != 4].reset_index(drop = True)
#at_gest.label[at_gest.label == 2] = 1
#at_gest.label[at_gest.label == 3] = 2

at_gest.label[at_gest.label == 2] = 1
#%% symbolize atomic gestures

# if you want only one df to save memory, just use at_gest insted of at_gest.copy()
model = mv.MMDSAX(df = at_gest.copy(), window = 20, stride = 10,
                 nbins = 30, alphabet = 'ABCDEFG')
#del at_gest
model.symbolize_windows(remove_borders = 2)
#%%
model.split_data(frac = 0.7, method = None, stratify_method = 'label',seed = 45)
model.trainModel(method = '1-NN', NoT = 1)
#model.trainModel(method = 'svm-light', NoT = 10)
results = model.classify(distance = 'sum')
results2 = model.classifyActivityThroughFiltering(distance = 'sum', win_length = 7)

print results[1]
print results2[1]

#%%
res_array = []
cur_mean = np.Inf
cur_index = np.Inf
N_t = 20
for i in range(50):
    model.split_data(frac = 0.7,stratified = True, stratify_method = 'label',seed = 1)
    acc = np.empty((N_t,1))
    for j in range(1,N_t):
        model.trainModel(NoT = j)
        results = model.classify(distance = 'sum')
        #acc[j-10] = results[1]
        res_array.append(results[1])
    mean = np.asarray(res_array).mean()
    res_array = []
    if mean < cur_mean:
        cur_mean = mean
        cur_index = i


#%%
import L1model as l1
l1_model = l1.L1model(df = at_gest.copy())
l1_model.split_data(frac = 0.7, method = 'stratified', stratify_method = 'label',seed = 45)
#l1_model.trainModel(method = 'svm-light', NoT = 5)
l1_model.trainModel(method = '1-NN', NoT = 1)

results1 = l1_model.classify(distance = 'sum')
results2 = l1_model.classifyActivityThroughFiltering(distance = 'sum', win_length = 7)

print results1[1]
print results2[1]


#%%
filt_sig2 = np.apply_along_axis(lambda x: f.gaussian_filter1d(x,8), 1, signals[0:1])[0]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

windows = rolling_window(filt_sig2[:5000], 500)

def autocorrelation(x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size/2]/np.sum(xp**2)
corr = np.apply_along_axis(autocorrelation, 1, windows)

def mapper(z):
    z = np.asarray(z)
    tmp = at.peakdet(z,delta = 0.1)
    if len(tmp) > 1:
        max_locs = map(int,np.apply_along_axis(lambda x: x[0],1, tmp[1:]))
        border = at.__getBorders1D(z, max_locs, wantWindows = False)
        if border[0] != []:
            start = np.asarray(border[0])
            stop = np.asarray(border[1])
            diff1 = abs(z[max_locs] - z[start]).min()
            diff2 = abs(z[max_locs] - z[stop]).min()
            return min(diff1,diff2)
        else: return 0
    else:
        return 0

peaks_corr =np.asarray( map(mapper, corr))
plt.plot(f.gaussian_filter1d(peaks_corr[:1900],50))
plt.show()
