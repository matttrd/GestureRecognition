# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 15:31:04 2016

@author: Matteo
"""


import pandas as pd
import numpy as np
from scipy.ndimage import filters as f
import matplotlib.pyplot as plt
import pyAtomicGesture as at
import mvSAX as mv
import L1model as l1
import math
from matplotlib import style
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

source_dA = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\AtGest_XC_AA09.json'
source_dA = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\AtGest_XC_AA09_3axis.json'

at_gest = pd.read_json(source_dA)
dims = ['dim0', 'dim1', 'dim2']

for dim in dims:
    at_gest[dim] = at_gest[dim].apply(np.asarray)
    at_gest[dim] = at_gest[dim].apply(lambda x: at.canonizeFromLTW(s = x,new_length = 150,interp_type = 'Akima'))

at_gest = at_gest[at_gest['label'] != 4].reset_index(drop = True)
at_gest.label[at_gest.label == 2] = 1




dest = 'D:\\Dropbox\\Matteo - Gian - Angelo\\paper\\IFAC 2017\\graphics\\'

s_ar = [1,2,3,4,5]
dist_ar = ['min', 'sum']
dist_index = 1

w = 30
alpha = 'ABCDEFG'
win_length = 7
seed = 45
name = 'XC'
strat_method = 'stratified'
style.use('ggplot')
nbins = 5
title_font_size = 15
dim_argmax = 'dim0' 
################################ multivariate
#SAX
res_dict = {'s':[],'dist':[],'GR':[],'AR':[],'method':[]}

sax_model = mv.MMDSAX(df = at_gest.copy(), window = 20, stride = 10, \
                  nbins = w, alphabet = alpha)
sax_model.symbolize_windows(remove_borders = 1)
sax_model.split_data(frac = 0.7, method = strat_method, stratify_method = 'label',seed = seed)
for s in s_ar:
    sax_model.trainModel(method = '1-NN', NoT = s)
    for dist in dist_ar:
        results_GR = sax_model.classify(distance = dist)
        results_AR = sax_model.classifyActivityThroughFiltering(distance = 'sum', win_length = win_length)
        res_dict['s'].append(s)
        res_dict['dist'].append(dist)
        res_dict['GR'].append(results_GR)
        res_dict['AR'].append(results_AR)
        res_dict['method'].append('SAX')


#L1
l1_model = l1.L1model(df = at_gest.copy())
l1_model.split_data(frac = 0.7, method = strat_method, stratify_method = 'label',seed = seed)
for s in s_ar:
    l1_model.trainModel(method = '1-NN', NoT = s)
    for dist in dist_ar:
        results_GR = l1_model.classify(distance = dist)
        results_AR = l1_model.classifyActivityThroughFiltering(distance = 'sum', win_length = win_length)
        res_dict['s'].append(s)
        res_dict['dist'].append(dist)
        res_dict['GR'].append(results_GR)
        res_dict['AR'].append(results_AR)
        res_dict['method'].append('L1')

res_df_mv = pd.DataFrame(res_dict)

################################ univariate
at_gest.drop('dim1',inplace = True,axis = 1)
at_gest.drop('dim2',inplace = True,axis = 1)

res_dict = {'s':[],'GR':[],'AR':[],'method':[]}

sax_model = mv.MMDSAX(df = at_gest.copy(), window = 20, stride = 10, \
                  nbins = w, alphabet = alpha)
sax_model.symbolize_windows(remove_borders = 1)
sax_model.split_data(frac = 0.7, method = strat_method, stratify_method = 'label',seed = seed)
for s in s_ar:
    sax_model.trainModel(method = '1-NN', NoT = s)
    results_GR = sax_model.classify(distance = dist)
    results_AR = sax_model.classifyActivityThroughFiltering(distance = 'sum', win_length = win_length)
    res_dict['s'].append(s)
    res_dict['GR'].append(results_GR)
    res_dict['AR'].append(results_AR)
    res_dict['method'].append('SAX')


#L1
l1_model = l1.L1model(df = at_gest.copy())
l1_model.split_data(frac = 0.7, method = strat_method, stratify_method = 'label',seed = seed)
for s in s_ar:
    l1_model.trainModel(method = '1-NN', NoT = s)
    results_GR = l1_model.classify(distance = dist)
    results_AR = l1_model.classifyActivityThroughFiltering(distance = 'sum', win_length = win_length)
    res_dict['s'].append(s)
    res_dict['GR'].append(results_GR)
    res_dict['AR'].append(results_AR)
    res_dict['method'].append('L1')

res_df_univar = pd.DataFrame(res_dict)
 


############################### PLOTS

############### MULTIVAR


##### SAX
accuracy_sax_2cl_GR = res_df_mv[(res_df_mv.method == 'SAX') & (res_df_mv.dist == dist_ar[dist_index])]['GR'].apply(lambda x: x[1])
accuracy_sax_2cl_AR = res_df_mv[(res_df_mv.method == 'SAX') & (res_df_mv.dist == dist_ar[dist_index])]['AR'].apply(lambda x: x[1])
#accuracy_sax_2cl_GR = accuracy_sax_2cl_GR[(accuracy_sax_2cl_GR.dist == dist[dist_index]) & (res_df_mv.s < 6)]
#accuracy_sax_2cl_AR = accuracy_sax_2cl_AR[(accuracy_sax_2cl_AR.dist == dist[dist_index]) & (res_df_mv.s < 6)]

fig = plt.figure()
dz = accuracy_sax_2cl_GR
ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of SAX-GR and $\\mathbf{\\theta} = s$', fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar)+1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracy_GR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_GR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

################ AR 

fig = plt.figure()
dz = accuracy_sax_2cl_AR
ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of SAX-AR and $\\mathbf{\\theta} = s$',fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar) + 1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracy_AR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_AR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)
                                          
                                          

##### L1
accuracy_L1_2cl_GR = res_df_mv[(res_df_mv.method == 'L1') & (res_df_mv.dist == dist_ar[dist_index])]['GR'].apply(lambda x: x[1])
accuracy_L1_2cl_AR = res_df_mv[(res_df_mv.method == 'L1') & (res_df_mv.dist == dist_ar[dist_index])]['AR'].apply(lambda x: x[1])
#accuracy_L1_2cl_GR= np.asarray(accuracy_L1_2cl_GR[(res_df_mv.dist == dist[dist_index]) & (res_df_mv.s < 6)])
#accuracy_L1_2cl_AR = np.asarray(accuracy_L1_2cl_AR[(res_df_mv.dist == dist[dist_index]) & (res_df_mv.s < 6)])

fig = plt.figure()
dz = accuracy_L1_2cl_GR

ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of L1-GR and $\\mathbf{\\theta} = s$', fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar)+1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracyL1_GR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_GR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

################ AR 

fig = plt.figure()
dz = accuracy_L1_2cl_AR
ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of L1-AR and $\\mathbf{\\theta} = s$',fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar) + 1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracyL1_AR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_AR_2classes_' + dist_ar[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)


########################### UNIVAR

##### SAX
accuracy_sax_2cl_GR = res_df_univar[(res_df_univar.method == 'SAX')]['GR'].apply(lambda x: x[1])
accuracy_sax_2cl_AR = res_df_univar[(res_df_univar.method == 'SAX')]['AR'].apply(lambda x: x[1])
#accuracy_sax_2cl_GR = accuracy_sax_2cl_GR[(accuracy_sax_2cl_GR.dist == dist[dist_index]) & (res_df_univar.s < 6)]
#accuracy_sax_2cl_AR = accuracy_sax_2cl_AR[(accuracy_sax_2cl_AR.dist == dist[dist_index]) & (res_df_univar.s < 6)]

fig = plt.figure()
dz = accuracy_sax_2cl_GR
ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of 1-D SAX-GR and $\\mathbf{\\theta} = s$', fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar)+1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracy_GR_2classes_' +dim_argmax + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_GR_2classes_' +dim_argmax + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

################ AR 

fig = plt.figure()
dz = accuracy_sax_2cl_AR
ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100 , color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of 1-D SAX-AR and $\\mathbf{\\theta} = s$',fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar) + 1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracy_AR_2classes_' +dim_argmax + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_AR_2classes_' +dim_argmax + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)
                                          
                                          

##### L1
accuracy_L1_2cl_GR = res_df_univar[(res_df_univar.method == 'L1')]['GR'].apply(lambda x: x[1])
accuracy_L1_2cl_AR = res_df_univar[(res_df_univar.method == 'L1')]['AR'].apply(lambda x: x[1])
#accuracy_L1_2cl_GR= np.asarray(accuracy_L1_2cl_GR[(res_df_univar.dist == dist[dist_index]) & (res_df_univar.s < 6)])
#accuracy_L1_2cl_AR = np.asarray(accuracy_L1_2cl_AR[(res_df_univar.dist == dist[dist_index]) & (res_df_univar.s < 6)])

################# GR                                   
fig = plt.figure()
dz = accuracy_L1_2cl_GR

ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of 1-D L1-GR and $\\mathbf{\\theta} = s$', fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar)+1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracyL1_GR_2classes_' +dim_argmax + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_GR_2classes_' +dim_argmax + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

################ AR 

fig = plt.figure()
dz = accuracy_L1_2cl_AR
ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of 1-D L1-AR and $\\mathbf{\\theta} = s$',fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar) + 1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracyL1_AR_2classes_' +dim_argmax + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_AR_2classes_' +dim_argmax + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)