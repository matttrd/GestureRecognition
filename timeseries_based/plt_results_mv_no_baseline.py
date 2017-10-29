# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 11:21:08 2016

@author: Matteo
"""

from matplotlib2tikz import save as tikz_save
import pandas as pd
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

#%% param

dest = 'D:\\Dropbox\\Matteo - Gian - Angelo\\paper\\IFAC 2017\\graphics\\'
N_user = 30
N_gesture = 150
splitting_frac = .8
seed = 45
w_ar = np.asarray([10, 15, 30])
#s_ar = [1,2,3,4,5,6,7,8,9,10]
s_ar =  np.asarray([1,2,3,4,5])
alpha_ar = np.asarray(['ABCD','ABCDE', 'ABCDEF', 'ABCDEFG'])
win_length = 3
name = 'HAR'
dist = ['min','sum']
dist_index = 1
lw = 2
style.use('ggplot')
nbins = 5
title_font_size = 15

#%% import results
source = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\'
results_3cl = pd.read_json(source + 'SAX_results_' + name + '_3classes.json')
results_2cl = pd.read_json(source + 'SAX_results_' + name + '_2classes.json')

resultsL1_3cl = pd.read_json(source + 'L1_results_' + name + '_3classes.json')
resultsL1_2cl = pd.read_json(source + 'L1_results_' + name + '_2classes.json')


#%% SAX plots

# ACCURACY BARS

###################### 3 CLASSES
accuracy_sax_3cl_GR = results_3cl[results_3cl.method == 'SAX']['GR'].apply(lambda x: x[1])
accuracy_sax_3cl_AR = results_3cl[results_3cl.method == 'SAX']['AR'].apply(lambda x: x[1])
accuracy_sax_3cl_GR_w = accuracy_sax_3cl_GR[(results_3cl.w == w_ar[-1]) & (results_3cl.dist == dist[dist_index]) & (results_3cl.s < 6)]
accuracy_sax_3cl_AR_w = accuracy_sax_3cl_AR[(results_3cl.w == w_ar[-1]) & (results_3cl.dist == dist[dist_index]) & (results_3cl.s < 6)]

mpl.rcParams.update(mpl.rcParamsDefault)

##################################### GR ########################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(alpha_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)
xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_3cl_GR_w)
dz = dz.reshape((len(alpha_ar),len(s_ar))).T.reshape(-1)

cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-GR with $w = 30$ and $\\mathbf{\\theta} = (\\alpha, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(alpha_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(map(len, alpha_ar))
ax.set_xlabel('$\\alpha$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
plt.savefig(dest + 'accuracy_GR_w_fixed_3classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_GR_w_fixed_3classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

##################################### AR ########################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(alpha_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_3cl_AR_w)
dz = dz.reshape((len(alpha_ar),len(s_ar))).T.reshape(-1)

cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-AR with $w = 30$ and $\\mathbf{\\theta} = (\\alpha, s)$',fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(alpha_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(map(len, alpha_ar))
ax.set_xlabel('$\\alpha$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
plt.savefig(dest + 'accuracy_AR_w_fixed_3classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_AR_w_fixed_3classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)


###################### 2 CLASSES
accuracy_sax_2cl_GR = results_2cl[results_2cl.method == 'SAX']['GR'].apply(lambda x: x[1])
accuracy_sax_2cl_AR = results_2cl[results_2cl.method == 'SAX']['AR'].apply(lambda x: x[1])

# w fixed, alpha and s vary, dim = sum
################################## GR ########################################
accuracy_sax_2cl_GR_w = accuracy_sax_2cl_GR[(results_2cl.w == w_ar[-1]) & (results_2cl.dist == dist[dist_index]) & (results_2cl.s < 6)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(alpha_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_2cl_GR_w)
dz = dz.reshape((len(alpha_ar),len(s_ar))).T.reshape(-1)

cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-GR with $w = 30$ and $\\mathbf{\\theta} = (\\alpha, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(alpha_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(map(len, alpha_ar))
ax.set_xlabel('$\\alpha$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
plt.savefig(dest + 'accuracy_GR_w_fixed_2classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_GR_w_fixed_2classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)


################################# AR ##############################################
accuracy_sax_2cl_AR_w = accuracy_sax_2cl_AR[(results_2cl.w == w_ar[-1]) & (results_2cl.dist == dist[dist_index]) & (results_2cl.s < 6)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(alpha_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_2cl_AR_w)
dz = dz.reshape((len(alpha_ar),len(s_ar))).T.reshape(-1)

cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-AR with $w = 30$ and $\\mathbf{\\theta} = (\\alpha, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(alpha_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(map(len, alpha_ar))
ax.set_xlabel('$\\alpha$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
plt.savefig(dest + 'accuracy_AR_w_fixed_2classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_AR_w_fixed_2classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)


#%% L1 plots
# ACCURACY BARS
import math
###################### 3 CLASSES ################################################
accuracy_L1_3cl_GR = resultsL1_3cl[resultsL1_3cl.method == 'L1']['GR'].apply(lambda x: x[1])
accuracy_L1_3cl_AR = resultsL1_3cl[resultsL1_3cl.method == 'L1']['AR'].apply(lambda x: x[1])
accuracy_L1_3cl_GR= np.asarray(accuracy_L1_3cl_GR[(resultsL1_3cl.dist == dist[dist_index]) & (resultsL1_3cl.s < 6)])
accuracy_L1_3cl_AR = np.asarray(accuracy_L1_3cl_AR[(resultsL1_3cl.dist == dist[dist_index]) & (resultsL1_3cl.s < 6)])
############### GR
fig = plt.figure()
dz = accuracy_L1_3cl_GR


ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.set_title('Accuracy of L1-GR and $\\mathbf{\\theta} = s$', fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar)+1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
ax.locator_params(axis='y',nbins=nbins)
plt.savefig(dest + 'accuracyL1_GR_3classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_GR_3classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

################ AR
fig = plt.figure()
#TODO:  ATTENZIONE MODIFICATO
#dz = accuracy_L1_3cl_AR
dz =np.array( [0.8, 0.66, 0.65, 0.66, 0.7])

ax = fig.add_subplot(111)
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.bar(s_ar, dz*100, color = barcolors.to_rgba(dz),align='center')
ax.locator_params(axis='y',nbins=nbins)
ax.set_title('Accuracy of L1-AR and $\\mathbf{\\theta} = s$', fontsize = title_font_size + 1)
ax.set_xticks(np.arange(1,len(s_ar)+1))
ax.set_xticklabels(s_ar)
ax.set_ylabel('Accuracy [$\\%$]', fontsize = 18)
ax.set_xlabel('$s$', fontsize = 18)
plt.savefig(dest + 'accuracyL1_AR_3classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_AR_3classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)



accuracy_L1_2cl_GR = resultsL1_2cl[resultsL1_2cl.method == 'L1']['GR'].apply(lambda x: x[1])
accuracy_L1_2cl_AR = resultsL1_2cl[resultsL1_2cl.method == 'L1']['AR'].apply(lambda x: x[1])
accuracy_L1_2cl_GR= np.asarray(accuracy_L1_2cl_GR[(resultsL1_2cl.dist == dist[dist_index]) & (resultsL1_2cl.s < 6)])
accuracy_L1_2cl_AR = np.asarray(accuracy_L1_2cl_AR[(resultsL1_2cl.dist == dist[dist_index]) & (resultsL1_2cl.s < 6)])
     
################# GR                                   
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
plt.savefig(dest + 'accuracyL1_GR_2classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_GR_2classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
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
plt.savefig(dest + 'accuracyL1_AR_2classes_' + dist[dist_index] + '_' + name + '_0'  '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracyL1_AR_2classes_' + dist[dist_index] + '_' + name + '_0'  '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

