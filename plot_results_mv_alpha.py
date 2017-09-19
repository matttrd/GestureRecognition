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
import math

#%% param

dest = 'D:\\Dropbox\\Matteo - Gian - Angelo\\paper\\IFAC 2017\\graphics\\'
N_user = 30
N_gesture = 150
splitting_frac = .8
seed = 45
w_ar = [10, 15, 30]
#s_ar = [1,2,3,4,5,6,7,8,9,10]
s_ar = [1,2,3,4,5]
alpha_ar = ['ABCD','ABCDE', 'ABCDEF', 'ABCDEFG']
win_length = 3
name = 'HAR'
dist = ['min','sum']
dist_index = 1
lw = 2
style.use('ggplot')
nbins = 5
title_font_size = 15
mpl.rcParams.update(mpl.rcParamsDefault)



#%% import results
source = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\'
results_3cl = pd.read_json(source + 'SAX_results_' + name + '_3classes.json')
results_2cl = pd.read_json(source + 'SAX_results_' + name + '_2classes.json')


#%% SAX plots

# ACCURACY BARS

###################### 3 CLASSES
accuracy_sax_3cl_GR = results_3cl[results_3cl.method == 'SAX']['GR'].apply(lambda x: x[1])
accuracy_sax_3cl_AR = results_3cl[results_3cl.method == 'SAX']['GR'].apply(lambda x: x[1])
accuracy_sax_3cl_GR_alpha = accuracy_sax_3cl_GR[(results_3cl.alpha == alpha_ar[-1]) & (results_3cl.dist == dist[dist_index]) & (results_3cl.s < 6)]
accuracy_sax_3cl_AR_alpha = accuracy_sax_3cl_AR[(results_3cl.alpha == alpha_ar[-1]) & (results_3cl.dist == dist[dist_index]) & (results_3cl.s < 6)]

##################################### GR ########################################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(w_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)
dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_3cl_GR_alpha)
dz = dz.reshape((len(w_ar),len(s_ar))).T.reshape(-1)
offset = math.ceil(dz.min()*100) - 10
offset -= offset % 10

cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100 - offset, color=barcolors.to_rgba(dz),alpha = 1, zsort='average')
ax.set_zticklabels(map(int,offset +  ax.get_zticks()))
ax.locator_params(axis='z',nbins=nbins)
ax.set_title('Accuracy of SAX-GR with $\\alpha = 6$ and $\\mathbf{\\theta} = (w, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(w_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(w_ar)
ax.set_xlabel('$w$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.azim = -62
ax.elev = 22
plt.savefig(dest + 'accuracy_GR_alpha_fixed_3classes_' + dist[dist_index] + '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_GR_alpha_fixed_3classes_' + dist[dist_index] + '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)

##################################### AR ########################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(w_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_3cl_AR_alpha)
dz = dz.reshape((len(w_ar),len(s_ar))).T.reshape(-1)


offset = math.ceil(dz.min()*100) - 10
offset -= offset % 10
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-AR  with $\\alpha = 6$ and $\\mathbf{\\theta} = (w, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(w_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(w_ar)
ax.set_xlabel('$w$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100 - offset, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
ax.set_zticklabels(map(int,offset +  ax.get_zticks()))
plt.savefig(dest + 'accuracy_AR_alpha_fixed_3classes_' + dist[dist_index] + '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_AR_alpha_fixed_3classes_' + dist[dist_index] + '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)


###################### 2 CLASSES
accuracy_sax_2cl_GR = results_2cl[results_2cl.method == 'SAX']['GR'].apply(lambda x: x[1])
accuracy_sax_2cl_AR = results_2cl[results_2cl.method == 'SAX']['AR'].apply(lambda x: x[1])

# w fixed, alpha and s vary, dim = sum
################################## GR ########################################
accuracy_sax_2cl_GR_alpha = accuracy_sax_2cl_GR[(results_2cl.alpha == alpha_ar[-1]) & (results_2cl.dist == dist[dist_index]) & (results_2cl.s < 6)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(w_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_2cl_GR_alpha)
dz = dz.reshape((len(w_ar),len(s_ar))).T.reshape(-1)


offset = math.ceil(dz.min()*100) - 10
offset -= offset % 10
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-GR with $\\alpha = 6$ and $\\mathbf{\\theta} = (w, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(w_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(w_ar)
ax.set_xlabel('$w$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100 - offset, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
ax.set_zticklabels(map(int,offset +  ax.get_zticks()))
plt.savefig(dest + 'accuracy_GR_alpha_fixed_2classes_' + dist[dist_index] + '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_GR_alpha_fixed_2classes_' + dist[dist_index] + '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)


################################# AR ##############################################
accuracy_sax_2cl_AR_alpha = accuracy_sax_2cl_AR[(results_2cl.alpha == alpha_ar[-1]) & (results_2cl.dist == dist[dist_index]) & (results_2cl.s < 6)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xedges = np.arange(len(w_ar))
yedges = np.arange(len(s_ar))
xpos, ypos = np.meshgrid(xedges, yedges)

xpos = xpos.flatten() + 0.5
ypos = ypos.flatten() + 0.5
#zpos = np.array(np.zeros_like(xpos))
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos)
dy = np.ones_like(ypos)
dz = np.array(accuracy_sax_2cl_AR_alpha)
dz = dz.reshape((len(w_ar),len(s_ar))).T.reshape(-1)


offset = math.ceil(dz.min()*100) - 10
offset -= offset % 10
cmap = plt.cm.RdYlBu
norm = mpl.colors.Normalize(vmin=0.3, vmax=1)
barcolors = plt.cm.ScalarMappable(norm, cmap)
ax.set_title('Accuracy of SAX-AR with $\\alpha = 6$ and $\\mathbf{\\theta} = (w, s)$', fontsize = title_font_size)
ax.set_xticks(np.arange(1,len(w_ar)+1))
ax.set_yticks(np.arange(1,len(s_ar)+1))
ax.set_yticklabels(s_ar)
ax.set_xticklabels(w_ar)
ax.set_xlabel('$w$', fontsize = 18)
ax.set_ylabel('$s$', fontsize = 18)
ax.set_zlabel('Accuracy [$\\%$]', fontsize = 18)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz*100 - offset, color=barcolors.to_rgba(dz),alpha = 0.9, zsort='average')
ax.locator_params(axis='z',nbins=nbins)
ax.set_zticklabels(map(int,offset +  ax.get_zticks()))
plt.savefig(dest + 'accuracy_AR_alpha_fixed_2classes_' + dist[dist_index] + '.png',bbox_inches='tight')
plt.savefig(dest + 'accuracy_AR_alpha_fixed_2classes_' + dist[dist_index] + '.svg',bbox_inches='tight')
mpl.rcParams.update(mpl.rcParamsDefault)
