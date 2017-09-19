
#import pyAtomicGesture as at
import fnmatch
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from itertools import chain
# method A

class L1model(object):
    '''
    Input must be a dictionary
    If keys are strings, method 1 has been used in the multivariate atomic gestures retrivial
    If keys are numbers, method 2    
        
    - Each key is a dimension
    - Dict value is a M-array, where M is the number of dimensions, and in turn each
     1-D array contains arrays representing atomic gestures
    
    '''
    
    def __init__(self, df = None):
        self.df = df
        self.dim = sorted(fnmatch.filter(df.columns, '*dim*'))
        self.NoD = len(self.dim)
        self.templates = None
        self.model = None
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
  
    def distance_templates(self, l1,templates):
        distance_matrix = np.empty([self.NoD,len(self.classes)],dtype = object)    
        for k in self.classes:
            for dim in range(self.NoD):
                distance_matrix[dim, self.class_dict[k]] = np.linalg.norm(l1[dim] - templates[k,self.dim[dim]])
        return distance_matrix
        #return pd.DataFrame(distance_matrix, columns = self.classes)
    
    
    def MV_min_distance_templates(self,l1,templates):
        distance_matrix = self.distance_templates(l1,templates)
        return np.apply_along_axis(min,0,distance_matrix)
    
    
    def MV_sum_distance_templates(self, l1,templates):
        distance_matrix = self.distance_templates(l1,templates)
        return np.apply_along_axis(sum,0,distance_matrix)    
       
    def MV_distance_template(self, l1,templates, distance = 'min'):
        if distance == 'min':
            return self.MV_min_distance_templates(l1,templates)
        if distance == 'sum':
            return self.MV_sum_distance_templates(l1,templates)
            
    def trainDistances(self, df = None, method = 'svm-light'):
        '''
        
        '''    
        if df is None or df.empty:
            raise("Invalid dataframe")
        
        dist_df = pd.DataFrame()
        
        np_array = np.array(df[self.dim])
        
        if method == 'svm-light':
            dist_matrix = np.array(map(lambda x: self.distance_templates(x, \
                    self.templates).reshape(-1),np_array))
            return dist_matrix
            
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
        
#        templates = dict(zip(K,[np.empty([self.NoD, NoT],dtype = object) \
#                                for k in range(len(K))])) 
        templates = dict()
        train_df = self.df.loc[self.train].reset_index(drop = True)
        
        for k in K:
            df_k = train_df[train_df['label'] == k].reset_index(drop = True)
           
            for this_dim in self.dim:
                tmp = df_k[this_dim].apply(lambda x: [list(x)])
                X = np.asarray(list(chain.from_iterable(tmp.values)))
                # S-means clustering 
                kmeans = KMeans(n_clusters = NoT, random_state=0).fit(X)
                templates[k,this_dim] = kmeans.cluster_centers_
      
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
        data_cols = fnmatch.filter(cols, '*dim*')     
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
                tmp = self.MV_distance_template(x, self.templates, distance)
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
        
#        if mode == 2:
#            y_filt_pred = np.empty(shape = y_pred.shape)
#            for i in range(delta, len(y_pred) - delta):
#                y_filt_pred[i] = Counter(y_pred[i - delta: i + delta + 1]).most_common(1)[0][0]
#            return (y_pred, accuracy_score(y_true, y_pred, normalize = True), confusion_matrix(y_true, y_pred),\
#                    precision_score(y_true, y_pred),recall_score(y_true, y_pred))
