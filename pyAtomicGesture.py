# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 10:02:44 2016
Atomic gestures module
@author: matttrd


IMPORTANT NOTES: the input df must have the column "data" in which each row 
is list o array of N_dim x N_samples
"""

# here dataset are supposed to be Pandas df with the only column of data

import sys
from scipy.interpolate import splev, splrep, Akima1DInterpolator
from scipy.signal import gaussian
from scipy.ndimage import filters as f
import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
import pandas as pd
from collections import Counter
#from detect_peaks import detect_peaks
import peak_detection as peak_det

#from sklearn.cluster import k_means
def smooth(x,window_len,window,sigma):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','gaussian']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'gaussian'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        if window == 'gaussian':
            w = gaussian(window_len, sigma)
        else:    
            w = eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
#
def smooth_dataset(df, window_len = 7,window = 'flat',sigma = 2): 
    smoothed = df.copy()
    if smoothed.shape[0] > 1:
        smoothed['data'] = smoothed['data'].apply(lambda z: np.array(map(lambda x_i: smooth(x_i,window_len, window,sigma),z)))
    else:
        smoothed['data'] = smoothed['data'].apply(lambda z: np.array(smooth(z,window_len, window,sigma)))
    return smoothed

# input as pandas dataframe
def time_equalizer(time, x_i):
    if time.size -  x_i.shape[0] > 0:
        time = time[:-1]
    return time


def resampleBsplines(x,y,r = 1,x2 = None):
    '''
    x is the x-axis corrently being used
    y is the f(x)      
    r is the resampling factor
    x2 in the new axis to be used
    '''
    tck = splrep(x,y)
    if not x2:
        if isinstance(r, (int, long, float, complex)):
            r = float(r)
            x2 = np.linspace(0, len(x)-1, r*len(x))
    return splev(x2,tck)    

def resampleAkima(x,y,r = 1,x2 = None):
    '''
    x is the x-axis corrently being used
    y is the f(x)      
    r is the resampling factor
    x2 in the new axis to be used (optional)
    '''
    if x2 is None:
        if isinstance(r, (int, long, float, complex)):
            r = float(r)
            x2 = np.linspace(0, len(x)-1, r*len(x))
    return Akima1DInterpolator(x, y, axis = 0).__call__(x2)
    
def resample_dataset(df, fs, method = 'Akima', r = 1.5,x2 = None):
    fs = float(fs)  
    if isinstance(r, (int, long, float, complex)):
        r = float(r)
    resampled = df.copy()
    if method == 'Bsplines':
        if resampled.shape[0] > 1: 
            resampled['data'] = df['data'].apply(lambda z: np.array(map(lambda x_i:\
            resampleBsplines(time_equalizer(np.arange(0,len(x_i)/fs, 1.0/fs),x_i), x_i,r,x2),z)))  
        else: # 1- dimensional case
            resampled['data'] = df['data'].apply(lambda z: np.array(resampleBsplines(time_equalizer(np.arange(0,len(z)/fs, 1.0/fs),z), z,r,x2)))
    else:
        if resampled.shape[0] > 1: 
            resampled['data'] = df['data'].apply(lambda z: np.array(map(lambda x_i:\
            resampleAkima(time_equalizer(np.arange(0,len(x_i)/fs, 1.0/fs),x_i), x_i,r,x2),z))) 
        else: # 1- dimensional case
            resampled['data'] = df['data'].apply(lambda z: np.array(resampleAkima(time_equalizer(np.arange(0,len(z)/fs, 1.0/fs),z), z, r,x2)))
    return resampled


def peakdet(v, delta = 0.01, x = None, method = 1):
    v = asarray(v)
    if method == 1:
        """
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.
        """
        maxtab = []
        #mintab = []
           
        if x is None:
            x = arange(len(v))
        
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        
        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        
        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN
        
        lookformax = True
        
        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                #mnpos = x[i]
            
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    #mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    #mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        
        return array(maxtab)
    if method == 2:
        pk_object = peak_det.peak_detection(v, flat_method='l', height_type = 'prominence')
        locs = pk_object.findPeaks(minPeakDist = 20, minPeakHeight = 0.1)
        max_vals = v[locs]
        return np.asarray(map(lambda x,y: (x,y), locs, max_vals))
        

def peakdetectFromDf(normalized_df,delta = 0.01):
    '''
    #TODO: to be improved
    Compute peaks on the normalized signal on numpy array.
    '''
    new_df = normalized_df.copy()
    if new_df.shape[0] > 1:    
        #find maxima    
        peaks = normalized_df['data'].apply(lambda z: np.array(map(lambda x_i: peakdet(x_i,delta),z)))
        new_df['max_peaks_loc'] = peaks.apply(lambda z: np.array(map(lambda x: map(lambda x_i: x_i[0],x),z)))
        new_df['max_peaks_val'] = peaks.apply(lambda z: np.array(map(lambda x: map(lambda x_i: x_i[1],x),z)))
        #find minima
        peaks = normalized_df['data'].apply(lambda z: np.array(map(lambda x_i: peakdet(-x_i,delta),z)))
        new_df['min_peaks_loc'] = peaks.apply(lambda z: np.array(map(lambda x: map(lambda x_i: x_i[0],x),z)))
        new_df['min_peaks_val'] = peaks.apply(lambda z: np.array(map(lambda x: map(lambda x_i: -x_i[1],x),z)))
    else: # 1- dimensional case
        #find maxima    
        peaks = normalized_df['data'].apply(lambda z: np.array(peakdet(z,delta)))
        new_df['max_peaks_loc'] = peaks.apply(lambda z: np.array(map(lambda x: x[0]),z))
        new_df['max_peaks_val'] = peaks.apply(lambda z: np.array(map(lambda x: x[1]),z))
        #find minima
        peaks = normalized_df['data'].apply(lambda z: np.array(peakdet(-z,delta),z))
        new_df['min_peaks_loc'] = peaks.apply(lambda z: np.array(map(lambda x: x[0]),z))
        new_df['min_peaks_val'] = peaks.apply(lambda z: np.array(map(lambda x: -x[1]),z))
    return new_df

def normalize(s,fs = 1,norm_typ = 'mean_std'):
      
    # s in a numpy 2dimarray    
    norm_s = s.copy()
    shape = s.shape
    if len(shape) > 1:   
        M = int(shape[0])
    else:      
        M = 1
    
    if M == 1:
        if norm_typ == 'area':
            norm_s = s/np.abs(np.trapz(s, dx = 1.0/fs))
        
        if norm_typ == 'max':
            norm_s = s/np.max(s)
            
        if norm_typ == 'mean_std':
            norm_s = (s - s.mean())/s.std()
                    
    else:
        for j in range(M): 
            if norm_typ == 'area':
                norm_s[j] = s[j]/np.abs(np.trapz(s[j], dx = 1.0/fs))
            
            if norm_typ == 'max':
                norm_s[j] = s[j]/np.max(s[j])
                
            if norm_typ == 'mean_std':
                norm_s[j] = (s[j] - s[j].mean())/s[j].std()     
    return norm_s

def normalize_df(s_df,fs = 1):
    #    
    # normalize the signal area
    s_norm = s_df.copy()
    s_norm['data'] = s_norm['data'].apply(lambda s: normalize(s,fs))
    return s_norm
   
def gaussianFilterFromDf(df,sigma = 1):
    #sigma can be a sequence of scalars
    df_new = df.copy()    
    df_new = df_new['data'].apply(lambda x: f.gaussian_filter(x,sigma))
    return df_new
    

def __getBorders1D(signal, max_locs, wantWindows = True,extr_method = None):
    # signal borders are not considered as maxima points
    tmp = np.r_[0, max_locs, len(signal)-1]
    tmp = map(lambda n: int(n),tmp)
    if tmp[0] == tmp[1]:
        tmp = np.delete(tmp,0)
    if tmp[-1] == tmp[-2]:
        tmp = np.delete(tmp,-1)
      
    start_locs = []
    end_locs = []        
    windows = []
    if extr_method != 'connected':
        for i in range(1,len(tmp)-1):
            start_loc = np.argmin(signal[tmp[i-1]: tmp[i]]) + tmp[i-1]
            end_loc = np.argmin(signal[tmp[i]: tmp[i + 1]]) + tmp[i]
            window = signal[start_loc:end_loc + 1]        
            start_locs.append(start_loc)
            end_locs.append(end_loc)
            windows.append(window)
    else:
        for i in range(1,len(max_locs)-1):
            start_loc = max_locs[i]
            end_loc = max_locs[i + 1]
            window = signal[start_loc:end_loc + 1]        
            start_locs.append(start_loc)
            end_locs.append(end_loc)
            windows.append(window)
            
    if wantWindows :   
        return start_locs,end_locs, windows
    else:
        return start_locs,end_locs

def getBorders(signal, max_locs, wantWindows = True, extr_method = None):
    
    #signal in a ndim numpy array: the number of rows is the dimension        
    if len(signal.shape) == 2:
        N = len(signal)
    else:
        N = 1
    if N == 1:
        if wantWindows:
            (start,end,windows) = __getBorders1D(signal, max_locs, \
                                            wantWindows,extr_method = extr_method)
            return (np.array(start),np.array(end), np.array(windows))
        else:
            (start,end) = __getBorders1D(signal, max_locs, wantWindows,\
                                                extr_method = extr_method)
            return (np.array(start),np.array(end))
    
    start_array = []
    end_array = []
    windows_array = []
    
    for i in range(N):
        if wantWindows:
            (start,end,windows) = __getBorders1D(signal[i],max_locs[i],\
                        wantWindows, extr_method = extr_method)
            windows_array.append(windows)
        else:
            (start,end) = __getBorders1D(signal[i],max_locs[i],\
                        wantWindows, extr_method = extr_method)
        start_array.append(start)
        end_array.append(end) 
     
    if wantWindows:
        return (np.array(start_array), np.array(end_array),np.array(windows_array))
    else:    
        return (np.array(start_array), np.array(end_array))

def __getWindowsFromBorders1D(signal, start_array, end_array):
    windows = []
    for i in range(len(start_array)):
         start_loc = start_array[i]
         end_loc = end_array[i]
         window = signal[start_loc : end_loc + 1]
         windows.append(window)
    return np.array(windows)

def getWindowsFromBorders(signal, start_array, end_array):
    windows_array = []  
    shape = start_array.shape    
    if len(shape) > 1:
        N = len(start_array)
    else:
        N = 1
    
    if N ==  1:
        windows = __getWindowsFromBorders1D(signal, start_array, end_array)
        return windows
    for i in range(N):
        windows = __getWindowsFromBorders1D(signal[i], start_array[i], end_array[i])
        windows_array.append(windows)
    return np.array(windows_array)

    
def getBordersFromDf(df,wantWindows):
    cols = df.columns
    new_df = df.copy()
    borders = [None]*len(df)
    if (not "max_peaks_loc" in cols) and (not "min_peaks_loc" in cols):
        raise('First run peakdetection')
    else:
        if ("max_peaks_loc" in cols) and (not "min_peaks_loc" in cols):
            for i,row in df.iterrows():
                borders[i] = getBorders(row['data'],row['max_peaks_loc'])
            new_df['borders'] = borders
            new_df['max_start'] = new_df['borders'].apply(lambda b: b[0])
            new_df['max_stop'] = new_df['borders'].apply(lambda b: b[1]) 
            
            
        else: 
            
            if (not "max_peaks_loc" in cols) and  ("min_peaks_loc" in cols):
                
                for i,row in df.iterrows():
                    borders[i] = getBorders(row['data'],row['min_peaks_loc'])
                new_df['borders'] = borders
                new_df['min_start'] = new_df['borders'].apply(lambda b: b[0])
                new_df['min_stop'] = new_df['borders'].apply(lambda b: b[1])            
            else:
                for i,row in df.iterrows():                
                    borders[i] = getBorders(row['data'],row['max_peaks_loc'])
                new_df['borders'] = borders
                new_df['max_start'] = new_df['borders'].apply(lambda b: b[0])
                new_df['max_stop'] = new_df['borders'].apply(lambda b: b[1])
                for i,row in df.iterrows(): 
                    borders[i] = getBorders(-row['data'],row['min_peaks_loc'])
                new_df['borders'] = borders
                new_df['min_start'] = new_df['borders'].apply(lambda b: b[0])
                new_df['min_stop'] = new_df['borders'].apply(lambda b: b[1])   
    
    if wantWindows:
        new_df['windows'] = new_df['borders'].apply(lambda b: b[2]) 
    
    new_df.drop('borders', inplace = True, axis = 1)       
    return new_df  


def canonizeFromLTW(s,new_length,interp_type = 'Akima'):
    '''
    Canonize from linear time warping
    s must be a 1D numpy signal
    interp_type must be 'Akima' or 'Bsplines'
    '''
    N_s = len(s)
    x = np.linspace(0,N_s - 1, N_s)
    x2 = np.linspace(0,N_s - 1, new_length)     
    if interp_type == 'Bsplines':
        return resampleBsplines(x = x, y = s, x2 = x2)
    else: 
        return resampleAkima(x = x, y = s, x2 = x2)
            
    
def getAtGestFromDf(df,isNormalized = False,fs = None, delta = None, sigma = 1,\
                    sampling_ratio = None, resampling_type = 'Akima'):
    '''
    Find atomic (non-repetitive) gestures: 
    1) if we are interested in sequencing a period (or quasi-periodic) time-series 
        then the extraction is only based on the maxima points
    2) if we are interested in extracting general elementary signals then we need
        to extract the maxima and the minima points (see the definitions of elementaryGesture).
     
    It return a list with n elements (where n = number of axes considered) and 
    each element contains all the atomic gestures corresponding to a dimension 
    
    #TODO: extend to all elementary gesture types
    '''    
    if sampling_ratio:
        df = resample_dataset(df, fs, method = resampling_type, r = sampling_ratio)
    if not isNormalized:
        df = normalize_df(df) 
        
    df = peakdetectFromDf(df,delta)
        
    #find the start and end position of gestures
    df = getBordersFromDf(df,wantWindows = True)
    dim = len(df['windows'][0])
    atomicGestures_array = [[] for i in range(dim)]
    for i,row in df.iterrows(): 
        for j in range(dim):
            atomicGestures_array[j].append(row['windows'][j])
    
    for j in range(dim):
        atomicGestures_array[j] = [item for sublist in atomicGestures_array[j] for item in sublist]
        
    return atomicGestures_array

def getAtGestsFromTS(signal,isNormalized = False, fs = None,\
    delta = 0.01, sigma = 1, sampling_ratio = None, peak_method = 1, resampling_type = 'Akima'):
    '''
    Get atomic non-repetitive gestures from 1D time series
    '''    
   
    if sampling_ratio:
        samples = np.linspace(0,len(signal)-1, len(signal))            
        signal = map(lambda x: resampleAkima(samples,x,r = sampling_ratio),signal)
        
    if not isNormalized:
        signal = normalize(signal) 
    
    peaks = peakdet(signal, delta, method = peak_method)  
    max_locs = map(lambda l:l[0], peaks)    
    atomicStructure = getBorders(signal, max_locs, wantWindows = True)
    return atomicStructure[-1]

def getAtRepGestsFromTS(signal,isNormalized = False, fs = None, delta = 0.01, \
sigma = 1, sampling_ratio = None, x2 = None, resampling_type = 'Akima', out_struct = 'eco',\
    want_df = False,peak_method = 1, extr_method = None):
    '''
    Get atomic repetitive gestures from 1D time series
    If out_struct = 'eco' outputs only atomic gestures otherwise outputs entire structure 
    '''    

    if sampling_ratio:
        samples = np.linspace(0,len(signal)-1, len(signal))            
        if resampling_type == 'Akima':
            signal = resampleAkima(samples,signal,r = sampling_ratio)
        if resampling_type == 'Bsplines':
            signal = resampleBsplines(samples,signal,r = sampling_ratio)
    if type(x2) == np.ndarray:
        samples = np.linspace(0,len(signal)-1, len(signal)) 
        if resampling_type == 'Akima':
            signal = resampleAkima(samples,signal,x2 = x2)
        if resampling_type == 'Bsplines':
            signal = resampleBsplines(samples,signal,x2 = x2)
    # first apply gaussian filter to find atomic gestures
    filtered_signal = f.gaussian_filter(signal,sigma)
    if not isNormalized:
        filtered_signal = normalize(filtered_signal,norm_typ='mean_std')    
    
    peaks = peakdet(filtered_signal, delta,method = peak_method ) 
    max_locs = map(lambda l:l[0], peaks)
    atomicStructure = getBorders(filtered_signal, max_locs, wantWindows = False, extr_method = extr_method)
#    atomicStructure = getBorders(filtered_signal, max_locs, wantWindows = True, extr_method = None)
    
    # now extract atomic gestures from original signal
    start_array = atomicStructure[0]
    end_array = atomicStructure[1]
#    atomicGestures = atomicStructure[2]
    atomicGestures = getWindowsFromBorders(signal, start_array, end_array)
        
    if want_df:
        return pd.DataFrame([(start_array,end_array),atomicGestures],\
                columns = ['borders','atomic_gestures'])
    else:
        if out_struct == 'eco':
            return atomicGestures
        else:
            return (start_array, end_array, atomicGestures)
   

def getMultiVariateAtRepGestsFromTS(signals,isNormalized = False, fs = None, \
    delta = 0.01, sigma = 1, x2 = None, sampling_ratio = None, T_gesture = None, resampling_type = 'Akima',\
    method = 1,peak_method = 1, extr_method = None):
    '''
    signal is a 2d-array (numpy) MxN. The first dimension must be the number of samples N
    METHOD 1: Take the most "periodic" as reference and use his starts and stops     
    METHOD 2: Take the most "periodic" as reference and match the atomic gestures of the other dimensions
    
    METHOD 1 returns an dictionary whose keys are string indexes (dim1, dim2, ...) 
            from zero to the number of dimensions. Each element is a list with 
            all the atomic gestures for the corresponding dimension
    METHOD 2 returns a dictionary whose keys are indexes from zero to the 
            number of atomic gestures in the reference dimension. Values are the atomic
            gestures that best match each other (from different dimensions).
            Each value is a M-dim list and each element contains the atomic gesture
            for each dimesion 
    '''
    
    def sequencer(signal, start, stop):
        windows_array = []        
        for i in range(len(start)):
            windows_array.append(signal[start[i]:stop[i] + 1])
        return np.array(windows_array)
                       
    shape = signals.shape
    if len(shape) > 1:
        M = shape[0]
        std = np.Inf
        atomicGestures_dict = {'dim' + str(k): [] for k in range(M)}        
        #borders_dict = {'borders_dim' + str(k): [] for k in range(M)}    
        borders_dict = {'borders': []}    
        cols = []
        for m in range(M):
            cols.append('dim' + str(m))
            (start,stop, at_gest) = getAtRepGestsFromTS(signals[m],isNormalized, \
                fs, delta, sigma, sampling_ratio, x2,resampling_type, out_struct = None,peak_method = peak_method)

            #borders_dict['borders_dim' + str(m)] = zip(start,stop)
           
            lim = min(len(start), len(stop))        
            dT = stop[0:lim] - start[0:lim]
            new_std =  dT.std() 
            
            if new_std < std:
                best_signal = m
                best_atom_gest = at_gest
                best_start = start
                best_stop = stop
                std = new_std
        
        atomicGestures_dict['dim' + str(best_signal)] = best_atom_gest
        #borders_dict['borders_dim' + str(best_signal)] = zip(best_start,best_stop)
        borders_dict['borders'] = zip(best_start,best_stop)
        print "Best signal is dim" + str(best_signal)                
        dims = range(M)
        dims.remove(best_signal)
        
        if method == 1:
        #****************************METHOD 1*************************************        
            for i in dims:
                atomicGestures_dict['dim' + str(i)] = sequencer(signals[i], best_start, best_stop)
            
#            return (atomicGestures_dict,borders_dict)    
        df_gest = pd.DataFrame(atomicGestures_dict)
        if T_gesture is not None:
            for col in cols:
#                for i in range(len(df_gest[col])):
#                    df_gest[col][i] = canonizeFromLTW(df_gest[col][i],T_gesture)
                df_gest[col] = df_gest[col].apply(lambda x: canonizeFromLTW(x,T_gesture,resampling_type))
        return pd.concat([df_gest,pd.DataFrame(borders_dict)],axis = 1)

             
##    
#        else:
#        #****************************METHOD 2************************************* 
#            if method == 2:
#                #TODO: aggiungi output borders come in metodo 1
#                NoAtGest = len(best_start)
#                matching_dict = dict(zip(map(lambda x: 'dim' + x,range(NoAtGest)),[[None for k in range(M)] for kk in range(NoAtGest)]))
#                def f(k,v,m,y):
#                    v[m] = y[k]
#                    return v
#                matching_dict = {k: f(k,v,best_signal,best_atom_gest) for k, v in matching_dict.iteritems()}
#                
#                already_assigned = []
#                for i in dims:
#                    borders = getAtRepGestsFromTS(signals[m],isNormalized, fs, delta, sigma, sampling_ratio, resampling_type, out_struct = None,method = 1)
#                    
#                    starts = borders[0]
#                    #for each atomic gesture find the nearest ones from other dimensions
#                    for j in len(best_start):
#                        matched_at_gest = np.argmin(starts - best_start[j])
#                        if not matched_at_gest in already_assigned:                    
#                            already_assigned.append(matched_at_gest)
#                            matching_dict[j][i] = borders[-1][matched_at_gest]
#                    already_assigned = []
#                return pd.DataFrame(matching_dict)
    else:
        raise('Not a numpy array')
    return    
  
def assignLabels(atom_df = None, y = None, yg = None):
    '''
    y is an array or list whose elements are classes of each samples
    yg is an array or list whose elements are classes of each atomic gesture 
    '''
    if type(atom_df) != pd.core.frame.DataFrame or atom_df.empty:
        raise("Input dataframe is not valid")
    label = []
    if yg:
        atom_df['y'] = yg
        return atom_df
    if y is not None:
        #TODO: here we only handle method 1 from multivariate gesture extraction
        (start,stop) = map(lambda x: x[0],atom_df['borders']),map(lambda x:x[1],atom_df['borders'])
        label = map(lambda a,b: Counter(y[a:b]).most_common(1)[0][0], start, stop)
              
    atom_df['label'] = label
    return atom_df
    
def assignUsers(atom_df = None, u = None):
    '''
    y is an array or list whose elements are classes of each samples
    yg is an array or list whose elements are classes of each atomic gesture 
    '''
    if type(atom_df) != pd.core.frame.DataFrame or atom_df.empty:
        raise("Input dataframe is not valid")
    user = []
    if u is not None:
        #TODO: here we only handle method 1 from multivariate gesture extraction
        (start,stop) = map(lambda x: x[0],atom_df['borders']),map(lambda x:x[1],atom_df['borders'])
        user = map(lambda a,b: Counter(u[a:b]).most_common(1)[0][0], start, stop)
              
    atom_df['user'] = user
    return atom_df
#def findElementaryBasis(signal,K):
#    '''
#    Find the basis of "K" elements of the timeseries "signal" 
#    '''
#    
#    #subdivide in N_t time scales
#       
#    
#    
#    clustering = kmean(n_clusters=20, init='k-means++', n_init=10, max_iter=300, tol=0.0001, \
#          precompute_distances='auto', verbose=1, copy_x=False, n_jobs=1)
#    clustering.fit_predict(X, y=None)
