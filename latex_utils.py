# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 17:30:50 2016

@author: Matteo

"""
import numpy as np
np.set_printoptions(precision = 3)

def confMatrix_to_latex(M = None, columns_format = None):
    '''
    input_matrix must be a dataframe with columns as name of classes
    '''
    classes = M.columns
    size = len(classes) # size x size matrix
    if columns_format is None:
        columns_format = 'l' + 'c' + '| ' + 'c'*size
    else: 
        if len(columns_format) != size + 3:
            raise("Not valid")
        unique_list = np.unique(list(columns_format))
        for t in unique_list:
            if t != 'r' or t != 'c' or t != 'l' or '|':
                raise(str(t) + ' is not valid')
    
    
    M2 = np.asarray(M)
    tot = np.apply_along_axis(sum, 1, M2)
    tot = np.array(tot, dtype = float)
    TOT = tot.repeat(size).reshape((size,size))   
    perc = M2/TOT
    perc = perc * 100
    perc = perc.round(2)
    
    def mapper(row,i): 
        row = list(row)
        l = []
        for j in range(len(row)):
            if j == i:
                l.append('\cellcolor[gray]{0.8}' + str(row[j]) + '&')
            else:
                l.append(str(row[j]) + '&')
       # l = map(lambda x: str(x) + '&', row)
        l[-1] = l[-1][:-1]
        return ''.join(l)

    def perc_mapper(row,i):
        row = list(row)
        l = []
        for j in range(len(row)):
            if j == i:
                l.append('\cellcolor[gray]{0.8}' + '(' + str(row[j]) + '\%)' + '&')
            else:
                l.append('(' + str(row[j]) + '\%)' + '&')

        l[-1] = l[-1][:-1]
        return ''.join(l)

      
    out_string = r'\begin{tabular}{' + columns_format + '}' + '\n' + \
   '& \multicolumn{1}{c|}{} & \multicolumn{' +  str(size) + r'}{c}{' + '\makebox[0pt]{Predicted}}' + r'\\' + '\n' \
    + r'&&' + ''.join([str(cl) + '&' for cl in classes[:-1]]) + str(classes[-1]) +  r'\\' + '\n' \
    + r'\hline' + '\n' + r'\vspace{-2mm}' + '\n' \
    + r'\parbox[t]{0mm}{\multirow{' +  str(size*2 + 1) + r'}{*}{\rotatebox[origin=c]{90}{True}}} && \cellcolor[gray]{0.8}\\' + '\n' \
    
    string = ''
    for i in range(size):
        string = string + \
        ''.join(r'& \multirow{2}{*}{' + str(classes[i]) + r'} & ' + \
                mapper(M.loc[i],i) +  r'\\' + '\n' + r'&&' + perc_mapper(perc[i,:],i) + r'\\'  + '\n')
    
        #        ''.join(r'& \multirow{2}{*}{' + str(classes[i]) + r'} & \cellcolor[gray]{0.8}' + \
#                mapper(M.loc[i],i) +  r'\\' + '\n' + r'&& \cellcolor[gray]{0.8}' + perc_mapper(perc[i,:],i) + r'\\'  + '\n')
    
    out_string = out_string + string + r'\end{tabular}'
    print out_string