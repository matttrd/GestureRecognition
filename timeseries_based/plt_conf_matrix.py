# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:55:47 2016

@author: Matteo
"""
name = 'HAR'
import pandas as pd
import latex_utils as lu

################### UNIVARIATE
source = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\'
results_2cl = pd.read_json(source + 'SAX_results_' + name + '_2classes_univariate.json')
resultsL1_2cl = pd.read_json(source + 'L1_results_' + name + '_2classes_univariate.json')

# SAX 
accuracy_sax_2cl_AR = results_2cl[results_2cl.method == 'SAX']['AR'].apply(lambda x: x[1])
ind = accuracy_sax_2cl_AR.argmax()
tmp = results_2cl.loc[ind]
conf_mat = tmp['AR'][2]
conf_mat = pd.DataFrame(conf_mat, columns = ['WLK','WUS + WDS'])
lu.confMatrix_to_latex(conf_mat)

# L1
accuracy_L1_2cl_AR = resultsL1_2cl[resultsL1_2cl.method == 'L1']['AR'].apply(lambda x: x[1])
ind = accuracy_L1_2cl_AR.argmax()
tmp = resultsL1_2cl.loc[ind]
conf_mat = tmp['AR'][2]
conf_mat = pd.DataFrame(conf_mat, columns = ['WLK','WUS + WDS'])
lu.confMatrix_to_latex(conf_mat)



################### MULTI
source = 'D:\\Dropbox\\Documents\\dottorato\\ActivityClassifi\\gesture_recognition\\data\\'

#results_3cl = pd.read_json(source + 'SAX_results_' + name + '_3classes.json')
results_2cl = pd.read_json(source + 'SAX_results_' + name + '_2classes.json')

#resultsL1_3cl = pd.read_json(source + 'L1_results_' + name + '_3classes.json')
resultsL1_2cl = pd.read_json(source + 'L1_results_' + name + '_2classes.json')


# SAX 
accuracy_sax_2cl_AR = results_2cl[results_2cl.method == 'SAX']['AR'].apply(lambda x: x[1])
ind = accuracy_sax_2cl_AR.argmax()
tmp = results_2cl.loc[ind]
conf_mat = tmp['AR'][2]
conf_mat = pd.DataFrame(conf_mat, columns = ['WLK','WUS + WDS'])
lu.confMatrix_to_latex(conf_mat)

# L1
accuracy_L1_2cl_AR = resultsL1_2cl[resultsL1_2cl.method == 'L1']['AR'].apply(lambda x: x[1])
ind = accuracy_L1_2cl_AR.argmax()
tmp = resultsL1_2cl.loc[ind]
conf_mat = tmp['AR'][2]
conf_mat = pd.DataFrame(conf_mat, columns = ['WLK','WUS + WDS'])
lu.confMatrix_to_latex(conf_mat)



