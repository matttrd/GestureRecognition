# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 15:52:05 2016

@author: matttrd

This module aims to extend SAX technique to multivariate input signals through 
different methods

METHOD A: Symbolize each dimension (independently) and then take minimum distance:
                Multivariate Minimum Distance SAX (MMDSAX)

METHOD B: Map the multivariate signal in unique symbolic stream  

If the the input signal is univariate it reduces to the normal sax
"""

#import pyAtomicGesture as at
import pysax
import fnmatch
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from joblib import Parallel, delayed
# method A

class MMDSAX(object):
    '''
    Input must be a dictionary
    If keys are strings, method 1 has been used in the multivariate atomic gestures retrivial
    If keys are numbers, method 2    
        
    - Each key is a dimension
    - Dict value is a M-array, where M is the number of dimensions, and in turn each
     1-D array contains arrays representing atomic gestures
    
    '''
    
    def __init__(self, df = None, window = None, stride = None, 
                 nbins = None, alphabet = None):
        self.df = df
        self.window = window
        self.nbins = nbins
        self.stride = stride
        self.alphabet = alphabet 
        self.dim = sorted(fnmatch.filter(df.columns, '*dim*'))
        tmp = fnmatch.filter(df.columns, '*symb*')
        if len(tmp) > 0:
            self.dim = self.dim[:len(self.dim)/2]
        self.NoD = len(self.dim)
        self.templates = None
        self.model = pysax.SAXModel(window , stride, nbins, alphabet)
        self.train = None
        self.test = None
        self.NoT = 1
        self.classes = None
        self.class_dict = None
        self.class_dict_inv = None
        self.class_model = None
        self.method = None
        self.y_pred = None
    

#TODO: we should find templates through clustering before symbolization         


    def stratify(self, y,frac,seed = None):
        '''
        y must be a numpy array
        '''
        np.random.seed(seed)
        strata = Counter(y).most_common()
        freqs = map(lambda x: x[1], strata)
        total = float(sum(freqs))
        #strata = map(lambda x: (x[0], x[1]/total),strata)
        freqs = map(lambda x: x/total, freqs)
        
        categories = map(lambda x: x[0],strata)    
        train_idx = []
        test_idx = []
        for k in categories:
            k_idx = np.flatnonzero(y == k)              
            msk = np.random.rand(len(k_idx)) <= frac
            k_train = k_idx[msk]
            k_test = k_idx[~msk]
            train_idx = train_idx + list(k_train)
            test_idx = test_idx + list(k_test)
        return (np.array(train_idx), np.array(test_idx), np.array(freqs))
        
            
    def split_data(self,frac,df = None, inplace = True, method = 'stratified', \
                  stratify_method = 'label',users = None,seed = None):
        '''
        frac is the normalized percentage of training samples; frac in (0,1)
        '''
        if inplace:  
            self.df.reset_index(drop = True)    
            y = np.array(self.df['label'])
            index = self.df.index.get_values()
            if method == 'stratified':
                if stratify_method == 'user' and users is not None:
                        users = np.asarray(users)
                        (train_idx, test_idx, freqs) = self.stratify(y = self.df.user, frac = frac,seed = seed)
                else:
                    (train_idx, test_idx, freqs) = self.stratify(y = y, frac = frac,seed = seed)
                self.train = train_idx
                self.test = test_idx
                return
            if method == 'random': #random split
                msk = np.random.rand(len(y)) < frac
                self.train = index[msk]
                self.test = index[~msk]
            else:
                msk = index <= len(y)*frac
                self.train = index[msk]
                self.test = index[~msk]
#        else:
#            if df:
#                df.reset_index()    
#                y = np.array(df['label'])
#                index = df.index.get_values()
#                if stratified:
#                    if stratify_method == 'user' and users is not None:
#                        users = np.asarray(users)
#                        (train_idx, test_idx, freqs) = self.stratify(y = self.df.user, frac = frac,seed = seed)
#                    else:
#                        (train_idx, test_idx, freqs) = self.stratify(y = y, frac = frac,seed = seed)
#                    train = train_idx
#                    test = test_idx
#                    return (train, test)
#                else: #random split
#                    msk = np.random.rand(len(y)) < frac
#                    train = index[msk]
#                    test = index[~msk]
#                    return (train, test)
#            else:
#                raise("Invalid Dataframe")
                
    # RETURN DF. HERE WE PREFER TO SAVE INDEXES TO PRESERVE MEMORY            
    #            train_df = pd.DataFrame()
    #            test_df = pd.DataFrame()
    #            
    #            for k in y:
    #                k_df =  self.df[y==k]               
    #                msk = np.random.rand(len(k_df)) < frac
    #                k_train = k_df[msk]
    #                k_test = k_df[~msk]
    #                train_df = pd.concat([train_df, k_train],ignore_index = True)
    #                test_df = pd.concat([test_df, k_test],ignore_index = True)
    #            
    #            self.train = train_df
    #            self.test = test_df
    #        else:
    #            msk = np.random.rand(len(y)) < frac
    #            self.train = self.df[msk]
    #            self.test = self.df[~msk]
    #            return
    #    else:
    #        if df:
    #            msk = np.random.rand(len(df)) < frac
    #            return (df[msk],df[~msk])
    #        else:
    #            msk = np.random.rand(len(self.df)) < frac
    #            return (self.df[msk],self.df[~msk])
                
    
    def MV_SAX_distance(self, l1,l2,distance = 'min'):
        
        if distance == 'min':
            return self.MV_SAX_min_distance(l1,l2)
        if distance == 'sum':
            return self.MV_SAX_sum_distance(l1,l2)
    
            
    def MV_SAX_sum_distance(self, l1,l2):
        '''
        Multivariate extension with sum function: distance is the sum of distance on
        each axis
        '''
        return sum(np.array(map(lambda a,b: self.model.symbol_distance(a,b),l1,l2)) )
       
                
    def MV_SAX_min_distance(self, l1,l2):
        '''
        Multivariate extension with min function: treat each dimension independently
        and return the minimum distance among the m dimensions
        '''
        d_array = np.array(map(lambda a,b: self.model.symbol_distance(a,b),l1,l2))
        return (d_array.argmin(),d_array.min())
    
    
    def MV_SAX_distance_template(self, l1,templates, distance = 'min'):
        if distance == 'min':
            return self.MV_SAX_min_distance_templates(l1,templates)
        if distance == 'sum':
            return self.MV_SAX_sum_distance_templates(l1,templates)
    
    
    def MV_SAX_min_distance_templates(self,l1,templates):
        distance_matrix = self.univariate_SAX_distance_templates(l1,templates)
        return np.apply_along_axis(min,0,distance_matrix)
    
    
    def MV_SAX_sum_distance_templates(self, l1,templates):
        distance_matrix = self.univariate_SAX_distance_templates(l1,templates)
        return np.apply_along_axis(sum,0,distance_matrix)    
    
    
    def univariate_SAX_distance_templates(self, l1,templates):
        distance_matrix = np.empty([self.NoD,len(self.classes)],dtype = object)    
        for k in self.classes:
            for dim in range(self.NoD):
                distance_matrix[dim, self.class_dict[k]] = self.model.word_templates_dist(l1[dim],\
                                                                templates[k][dim])
        return distance_matrix
        #return pd.DataFrame(distance_matrix, columns = self.classes)
    
    def trainDistances(self, df = None, method = 'svm-light',distance = 'min'):
        '''
        
        '''    
        if df is None or df.empty:
            raise("Invalid dataframe")
        
        dist_df = pd.DataFrame()
        data_cols = list(set(df.columns).difference('label'))
        data_cols = fnmatch.filter(data_cols, '*symb*')
        
        np_array = np.array(df[data_cols])
        
        if method == 'svm-light':
            dist_matrix = np.array(map(lambda x: self.univariate_SAX_distance_templates(x, \
                    self.templates).reshape(-1),np_array))
            return dist_matrix
            
        if method == 'mutual_dist':
            #distance between obs i and all other obs
            N = len(np_array)
            dist_matrix = np.zeros([N - 1,N - 1])
            for j in range(N-1):
                for i in range(j+1, N-1):
                    dist_matrix[j,i-1] = self.MV_SAX_distance(np_array[j],\
                                        np_array[i],distance)
                if j != N-2:
                    dist_matrix[j+1:,j] = dist_matrix[j,:-1]
            
        return pd.DataFrame(dist_df,columns = self.dims)
            
    
    def trainTemplates(self,K = None,frac = 0.7, NoT = 1):
        
        '''
        Supervised learning: find the most common representation for each gesture
        Inputs: 
            - K not used so far
            - frac: number in (0,1) identifying the percentage of dataset used to 
                    train the model
            - number of templates per class
        '''
        
        if self.train is None:
            raise("Data splitting is required")
        #if not K:
        K = self.classes    
        if NoT > 1:
            self.NoT = NoT
        else:
            NoT = 1
        
        templates = dict(zip(K,[np.empty([self.NoD, NoT],dtype = object) \
                                for k in range(len(K))])) 
        train_df = self.df.loc[self.train]
        
        for k in K:        
            df_k = train_df[train_df['label'] == k]
            for this_dim in range(len(self.dim)):
                templates[k][this_dim] = map(lambda t: t[0], \
                    Counter(df_k['symbolic_' + self.dim[this_dim]]).most_common(NoT)) 
        return templates
        
    #TODO: extend with other classification techniques
    def trainModel(self, method = '1-NN', NoT = None):
        self.classes = sorted(self.df['label'][self.train].unique())
        self.class_dict = dict(zip(self.classes,range(len(self.classes))))
        self.class_dict_inv = dict(zip(range(len(self.classes)),self.classes))
        self.method = method
        self.templates = None
        if method == '1-NN':
            self.templates = self.trainTemplates(NoT = NoT)
            self.class_model = self.templates
            return self.templates
        
        if method == 'svm-light':
            df = self.df.loc[self.train]
            self.templates = self.trainTemplates()
            X = np.array(self.trainDistances(df,'svm-light'))
            y = np.array(df['label'])
        
        if method == 'svm-full':
            df = self.df.loc[self.train]
            X = np.array(self.trainDistances(df,'mutual_dist'))
        
        C_range = np.logspace(-6, -1, 10)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma = gamma_range, C = C_range)
        grid = GridSearchCV(estimator = SVC(), param_grid = param_grid, n_jobs = 8)
        X = preprocessing.scale(X)
        grid.fit(X,y)    
        self.class_model = grid    
        self.method = method    
        return grid
    
    def _classify(self, test = None, distance = None):            
        y_true = test['label']
        cols = test.columns
        if not 'label' in cols:
            raise("Missing label column")
        data_cols = list(set(cols).difference('label'))
        data_cols = fnmatch.filter(data_cols, '*symb*')        
        
        if len(data_cols) != self.NoD:
            raise("number of dimensions is different from the trained model")
        
        #dist_df = trainDistances(test,method = '')
        np_array = np.array(test[data_cols])
        
        if self.method == '1-NN':
    #    
    #        def f(x):
    #            return np.array(MV_SAX_distance_templates(x, self.templates))   
    #
    #        # prediction and relative distance in tuples
    #        prediction = np.array(map(lambda x: (np.unravel_index(f(x).argmin(),(self.NoD,len(self.classes)))[1],\
    #                                        f(x).min()), np_array))
            if distance is None:
                distance = 'min'
            
            def match_class(x,distance):
                tmp = self.MV_SAX_distance_template(x, self.templates, distance)
                return (self.class_dict_inv[tmp.argmin()],tmp.min())
                                                
            prediction = np.array(map(lambda x: match_class(x,distance), np_array))
            #return (prediction, classification_report(y_true, map(lambda x: x[0],prediction)))  
            y_pred = map(lambda x: x[0],prediction)
            return (y_pred, accuracy_score(y_true, y_pred, normalize = True), confusion_matrix(y_true, y_pred),\
                    precision_score(y_true, y_pred,average = None),recall_score(y_true, y_pred,average = None))
        else:
            if self.method == 'svm-light':
                X_test = np.array(self.trainDistances(test,'svm-light'))
            
            if self.method == 'svm-full':
                X_test = np.array(self.trainDistances(test,'mutual_dist'))
                
            X_test = preprocessing.scale(X_test)
            y_pred = self.class_model.predict(X_test)
            #return (y_pred, classification_report(y_true, y_pred))  
            return (y_pred, accuracy_score(y_true, y_pred, normalize = True), confusion_matrix(y_true, y_pred),\
                    precision_score(y_true, y_pred,average = None),recall_score(y_true, y_pred,average = None))
     
    def classify(self,test = None,distance = None):    
        '''
        Classification phase
        Test (optional): if test is missing use self.test
                        - test can be input dataframe containing only columns with data
                        and the column label. All other columns are supposed to be 
                        different signal dimesions (or in general features)
        '''
        
        if test is not None:
            return self._classify(test,distance)
        else:
            prediction = self._classify(self.df.loc[self.test].reset_index(drop = True),distance)
            self.y_pred = np.asarray(prediction[0])
            return prediction
            
    def classifyActivityThroughFiltering(self, test = None, y_true = None, \
                        y_pred = None, distance = None, win_length = 3, mode = 1):
        if (y_true is None) or (y_pred is None):
            if self.model is None:
                if test is not None:
                    y_true = test['label']
                    y_pred = np.asarray(self.classify(test,distance)[0])
                else:
                    y_true = self.df.loc[self.test].label
                    y_pred = np.asarray(self.classify(distance = distance)[0])
            else:
                y_true = self.df.loc[self.test].label
                y_pred = self.y_pred
        
        delta = int((win_length - 1.0)/2)
        if mode == 1:
            for i in range(delta, len(y_pred) - delta):
                y_pred[i] = Counter(y_pred[i - delta: i + delta + 1]).most_common(1)[0][0]
            return (y_pred, accuracy_score(y_true, y_pred, normalize = True), confusion_matrix(y_true, y_pred),\
                    precision_score(y_true, y_pred,average = None),recall_score(y_true, y_pred,average = None))
#        
#        if mode == 2:
#            y_filt_pred = np.empty(shape = y_pred.shape)
#            for i in range(delta, len(y_pred) - delta):
#                y_filt_pred[i] = Counter(y_pred[i - delta: i + delta + 1]).most_common(1)[0][0]
#           return (y_pred, accuracy_score(y_true, y_pred, normalize = True), confusion_matrix(y_true, y_pred),\
#                    precision_score(y_true, y_pred),recall_score(y_true, y_pred)) 
#                
        
    def symbolize_windows(self, inplace = True, arg = None, parallel = False,\
                          remove_borders = 0):    
        # symbolize windows 
        '''
        arg is optional and must be a M-columns dataframe containing only gestures 
        (not borders, etc)
        '''
        if arg is not None:
            if not arg == pd.core.frame.DataFrame:
                df = pd.Dataframe(arg)
                df = pd.DataFrame()
                
            for this_dim in self.dim:
                df['symbolic_' + this_dim] = df[this_dim].apply(lambda x:\
                    self.model.symbolize_window(x, brd = remove_borders))
            return df
        
        if inplace:
            
            def f_tmp(self, x):
                x.apply(lambda x: self.model.symbolize_window(x,brd =remove_borders))
                
            if parallel:
                return Parallel(n_jobs = 8)(delayed(f_tmp)(self.df[this_dim]) for this_dim in self.dim)
            else:  
                
                #self.df[map(lambda x: 'symbolic_' + x,self.dim)] = self.df.apply(self.model.symbolize_window, axis = 1)
                for this_dim in self.dim:
                    self.df['symbolic_' + this_dim] = self.df[this_dim].apply(lambda x:\
                    self.model.symbolize_window(x,brd = remove_borders))
                return
        else:
            df = pd.DataFrame()
            for this_dim in self.dim:
                df['symbolic_' + this_dim] = df[this_dim].apply(lambda x:\
                    self.model.symbolize_window(x,brd = remove_borders))
            return df
            
    def get_templates(self):
        return self.templates
    
    def get_model(self):
        return self.model
    
    def get_splitting(self):
        return (self.train, self.test)
    
    def get_dimesions(self):
        return self.dim
    
    def get_class_model(self):
        return self.class_model                 
    #TODO: method B


