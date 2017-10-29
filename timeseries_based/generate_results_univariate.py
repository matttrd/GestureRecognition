# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 18:02:25 2016

@author: Matteo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 09:30:12 2016

Results comparison script
"""


import pandas as pd
import numpy as np
from scipy.ndimage import filters as f
import pyAtomicGesture as at
import mvSAX as mv
import L1model as l1

#%% parameters
N_user = 29
N_gesture = 150
splitting_frac = 0.8
seed = 45
w_ar = [10, 15, 30]
s_ar = [1,2,3,4,5]
alpha_ar = ['ABCD','ABCDE', 'ABCDEF', 'ABCDEFG']
win_length = 3
name = 'HAR'

#%% functions

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


def gestureExtraction(signal = None, user = None, label = None, normalize_before = True, sigma = 1, N_gesture = None):
    filt_sig = np.apply_along_axis(lambda x: f.gaussian_filter1d(x,sigma), 1, signal)
    if normalize_before:
        filt_sig = at.normalize(filt_sig,'mean_std')
        
    at_gest = at.getMultiVariateAtRepGestsFromTS(filt_sig,isNormalized = False, fs = None, \
    delta = 0.01, sigma = 8, sampling_ratio = None, T_gesture = N_gesture,resampling_type = 'Akima', method = 1)
    at_gest = at.assignLabels(atom_df = at_gest, y = label)
    if user is not None:
        at_gest = at.assignUsers(atom_df = at_gest, u = user)
    return at_gest
    
    
#%% HAR DATASET
dataset = get_dataset(name)
dataset = dataset[dataset.User.isin(range(1, N_user+1))].reset_index()

#%%
'''
"cross-validation" of parameters (w,alpha, s) with 1-NN
'''

signals = np.array(map(lambda x: np.array(x),dataset['input'])).T
label = dataset['output']
user = dataset['User']

dims = sorted(map(lambda x: 'dim' + str(x), range(len(signals))))

#%%******************************* sax model*************************************
#%% 3-classes 
res_dict = {'w':[], 'alpha':[],'s':[],'GR':[],'AR':[],'method':[],'dim': []}
for i in range(len(dims)):
    at_gest = gestureExtraction(signal = signals[i:i+1], sigma = 1,user = user,label = label, \
                            normalize_before = True, N_gesture = N_gesture)
    for w in w_ar:
        for alpha in alpha_ar:
            sax_model = mv.MMDSAX(df = at_gest.copy(), window = 20, stride = 10, \
                              nbins = w, alphabet = alpha)
            sax_model.symbolize_windows(remove_borders = 1)
            sax_model.split_data(frac = splitting_frac, method = None, stratify_method = 'label',seed = seed)
            for s in s_ar:
                sax_model.trainModel(method = '1-NN', NoT = s)
                results_GR = sax_model.classify()
                results_AR = sax_model.classifyActivityThroughFiltering(win_length = win_length)
                res_dict['w'].append(w)
                res_dict['alpha'].append(alpha)
                res_dict['s'].append(s)
                res_dict['GR'].append(results_GR)
                res_dict['AR'].append(results_AR)
                res_dict['method'].append('SAX')
                res_dict['dim'].append(dims[i])

            

#%% 2-classes
res_dict2classes = {'w':[], 'alpha':[],'s':[],'GR':[],'AR':[],'method':[],'dim': []}

for i in range(len(dims)):
    at_gest = gestureExtraction(signal = signals[i:i+1], sigma = 1,user = user,label = label, \
                            normalize_before = True, N_gesture = N_gesture)
    at_gest.label[at_gest.label == 2] = 1
    for w in w_ar:
        for alpha in alpha_ar:
            sax_model =mv.MMDSAX(df = at_gest.copy(), window = 20, stride = 10, \
                              nbins = w, alphabet = alpha)
            sax_model.symbolize_windows(remove_borders = 1)
            sax_model.split_data(frac = splitting_frac, method = None, stratify_method = 'label',seed = seed)
            for s in s_ar:
                sax_model.trainModel(method = '1-NN', NoT = s)
                results_GR = sax_model.classify()
                results_AR = sax_model.classifyActivityThroughFiltering(win_length = win_length)
                res_dict2classes['w'].append(w)
                res_dict2classes['alpha'].append(alpha)
                res_dict2classes['s'].append(s)
                res_dict2classes['GR'].append(results_GR)
                res_dict2classes['AR'].append(results_AR)
                res_dict2classes['method'].append('SAX')
                res_dict2classes['dim'].append(dims[i])

#%% *****************************l1 model*********************************************

#%% 3-classes 
res_dictL1 = {'s':[],'GR':[],'AR':[],'method':[],'dim':[]}

for i in range(len(dims)):
    at_gest = gestureExtraction(signal = signals[i:i + 1], sigma = 1,user = user,label = label, \
                            normalize_before = True, N_gesture = N_gesture)

    l1_model = l1.L1model(df = at_gest.copy())
    l1_model.split_data(frac = splitting_frac, method = None, stratify_method = 'label',seed = seed)
    for s in s_ar:
        l1_model.trainModel(method = '1-NN', NoT = s)
        results_GR = l1_model.classify()
        results_AR = l1_model.classifyActivityThroughFiltering(distance = 'sum',win_length = win_length)
        res_dictL1['s'].append(s)
        res_dictL1['GR'].append(results_GR)
        res_dictL1['AR'].append(results_AR)
        res_dictL1['method'].append('L1')
        res_dictL1['dim'].append(dims[i])
            

#%% 2-classes
res_dictL12classes = {'s':[],'GR':[],'AR':[],'method':[],'dim':[]}

for i in range(len(dims)):
    at_gest = gestureExtraction(signal = signals[i:i+1], sigma = 1,user = user,label = label, \
                            normalize_before = True, N_gesture = N_gesture)
    at_gest.label[at_gest.label == 2] = 1
   
    l1_model = l1.L1model(df = at_gest.copy())
    l1_model.split_data(frac = splitting_frac, method = None, stratify_method = 'label',seed = seed)
    for s in s_ar:
        l1_model.trainModel(method = '1-NN', NoT = s)
        results_GR = l1_model.classify()
        results_AR = l1_model.classifyActivityThroughFiltering(distance = 'sum',win_length = win_length)
        res_dictL12classes['s'].append(s)
        res_dictL12classes['GR'].append(results_GR)
        res_dictL12classes['AR'].append(results_AR)
        res_dictL12classes['method'].append('L1')
        res_dictL12classes['dim'].append(dims[i])

#%%
dest = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\'

res_df = pd.DataFrame(res_dict)
res_df2classes = pd.DataFrame(res_dict2classes)
res_df.to_json(dest + 'SAX_results_' + name + '_3classes_univariate.json')
res_df2classes.to_json(dest + 'SAX_results_' + name + '_2classes_univariate.json')

res_dfL1 = pd.DataFrame(res_dictL1)
res_dfL12classes = pd.DataFrame(res_dictL12classes)
res_dfL1.to_json(dest + 'L1_results_' + name + '_3classes_univariate.json')
res_dfL12classes.to_json(dest + 'L1_results_' + name + '_2classes_univariate.json')