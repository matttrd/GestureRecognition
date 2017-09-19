# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:37:23 2016

@author: Matteo
"""

import numpy as np
from scipy.interpolate import splev, splrep, Akima1DInterpolator

class peak_detection(object):
    '''
    Inputs:
    ------
    signal: list, pd.Series or 1-D numpy array
    
    flat_method = {None, 'l','r','b'}, optional. With a flat peak: if 'r' takes the
             left peak, if 'r' takes the right peak, if 'b' takes both
             if None discard flat peaks
    
    - resampling_method:
    '''
    #TODO: extension to multivariate case
    
    def __init__(self, signal, flat_method = 'l', height_type = 'prominence'):
        if signal is None:
            raise "Not a valid signal"
        signal = np.atleast_1d(signal).astype('float64')  
        self.signal = signal
        self.len = len(signal)
        self.flat_method = flat_method
        self.height_type = height_type
        self.height = None
        self.locs = None
        self.l_b = None
        self.r_b = None
        self.ll_b = None
        self.rr_b = None

    def findPeaks(self, minPeakDist = None, minPeakHeight = None, maxPeakHeight = None, \
                  minPeakTh = None, maxPeakTh = None, height_type = None, remove_trend = False):
        
        y = self.signal
        if remove_trend:
            y = self._removeBaseline(y)
        if height_type is not None:
            self.height_type = height_type
        if self.locs is None:
            locs = self._findInitialPeaks(y)
            self.locs = locs
            if self.height_type == 'prominence':
                height = self._evaluateProminence(self.locs,y)
            if self.height_type == 'local':
                height = self._findHeight(locs, y)
            self.height = height
        else:
            locs = self.locs
            height = self.height
        idx = np.ones_like(locs, dtype = bool)
        if minPeakDist is not None:
            idel = np.zeros(locs.size, dtype=bool)
            # detect small peaks closer than minimum peak distance
            if locs.size and minPeakDist > 1:
                locs = locs[np.argsort(height)][::-1]  # sort ind by peak height
                idel = np.zeros(locs.size, dtype=bool)
                for i in range(locs.size):
                    if not idel[i]:
                        # keep peaks with the same height if kpsh is True
                        idel = idel | (locs >= locs[i] - minPeakDist) & (locs <= locs[i] + minPeakDist)
                        idel[i] = 0  # Keep current peak
        
        tmp = ~idel
        #locs = np.sort(locs[idx]) 
        tmp = np.sort(tmp)
        idx = idx & tmp
        
        if minPeakHeight is not None:
                tmp = height >= minPeakHeight
                idx = idx & tmp
        if minPeakTh is not None:
            tmp = y[locs] >= minPeakTh
            idx = idx & tmp            

        if maxPeakHeight is not None:
            tmp = height >= minPeakHeight
            idx = idx & tmp
        if maxPeakTh is not None:
            tmp = y[locs] <= maxPeakTh
            idx = idx & tmp
        return locs[idx]
            
    def interpolate(self, x, y, r = 1.5, x2 = None, method = 'Akima'):
        if method == 'Akima':
            return self._interpAkima(x, y, r, x2)
        if method == 'Bsplines':
            return self._interpBsplines(x, y, x2 = x2)
            
    
    def _findHeight(self, locs, y):
        height = np.zeros_like(locs, dtype = float)
        l_b = np.zeros_like(locs, dtype = int)
        r_b = np.zeros_like(locs, dtype = int)
    
        aug_locs = np.hstack((0, locs, len(y) - 1))
        for i in range(1, len(aug_locs) - 1):
            left = y[aug_locs[i - 1] : aug_locs[i]]
            right = y[aug_locs[i] + 1 : aug_locs[i + 1] + 1]
            height[i - 1] = min(y[aug_locs[i]] - left.min(), y[aug_locs[i]] - right.min())
            l_b[i-1] = aug_locs[i - 1] + left.argmin()
            r_b[i-1] = aug_locs[i] + right.argmin() + 1
        self.l_b = l_b
        self.r_b = r_b
        self.height = height            
        return 
        
    def _findInitialPeaks(self,y):
        dy = y[1:] - y[:-1]
        ind_none, ind_l, ind_r = np.array([[], [], []], dtype=int)
        if self.flat_method is None:
            ind_none = np.where((np.hstack((dy, 0)) < 0) & (np.hstack((0, dy)) > 0))[0]
        else:
            if self.flat_method.lower() in ['l', 'b']:
                ind_l = np.where((np.hstack((dy, 0)) <= 0) & (np.hstack((0, dy)) > 0))[0]
            if self.flat_method.lower() in ['l', 'b']:
                ind_r = np.where((np.hstack((dy, 0)) < 0) & (np.hstack((0, dy)) >= 0))[0]
        
        locs = np.unique(np.hstack((ind_none, ind_l, ind_r)))    
    
        # first and last values of x cannot be peaks
        if locs.size and locs[0] == 0:
            locs = locs[1:]
        if locs.size and locs[-1] == y.size-1:
            locs = locs[:-1]

        return locs

            
    def _evaluateProminence(self, locs, y):
        
        '''
        Find prominence
        
        For each peak:
        Extend a horizontal line from the peak to the left and right until
        the line does one of the following:
            1) Crosses the signal because there is a higher peak
            2) Reaches the left or right end of the signal
        Find the minimum of the signal in each of the two intervals defined in Step 2. 
        This point is either a valley or one of the signal endpoints.
        The higher of the two interval minima specifies the reference level. 
        The height of the peak above this level is its prominence.
        '''
        llb = np.zeros_like(locs)
        rrb = np.zeros_like(locs)
        prom = np.zeros_like(locs, dtype = float)
        
        for i in range(len(locs)):
            loc = locs[i]
            left = y[:loc]
            right = y[loc+1:]
            idx_cross = np.where(left >= y[loc])[0]
            if not idx_cross.size:
                # no higher peak
                mly = left[0:loc].min()
                mlx = left[0:loc].argmin()
            else:
                mly = left[idx_cross[-1]:loc].min()
                mlx = left[idx_cross[-1]:loc].argmin()
        
                idx_cross = np.where(right >= y[loc])[0]
            if not idx_cross.size:
                # no higher peak
                mry = right.min()
                mrx = right.argmin() + loc + 1
            else:
                mry = right[:idx_cross[1]].min()
                mrx = right[:idx_cross[1]].argmin() + loc + 1
         
            if mly > mry:
                llb[i] = mlx 
                #now find the corresponding point to the right
                rrb[i] = (right <= mly)[0] + loc + 1
                prom[i] = y[loc] - mly
            else:
                rrb[i] = mrx
                llb[i] = (left <= mry)[-1]
                prom[i] = y[loc] - mry

        self.ll_b = llb
        self.rr_b = rrb
        return prom
        
        
    def _removeBaseline(self, y):
        '''
        Remove baseline defined as the mean between top and bottom envelops
        '''
        if self.locs is None:
            locs_max = self._findInitialPeaks(y)
            self.locs = locs_max
        else:
            locs_max = self.locs
        locs_max = np.hstack((0, locs_max, len(y) - 1))
        locs_min = self._findInitialPeaks(-y)
        locs_min = np.hstack((0, locs_min, len(y) - 1))
        y_max = y[locs_max]
        y_min =y[locs_min]
        x2 = np.arange(0,len(y))
        top_env = self.interpolate(locs_max, y_max, x2 = x2, method = 'Bsplines')
        bottom_env = self.interpolate(locs_min, y_min, x2 = x2, method = 'Bsplines')
        mean = (top_env + bottom_env)/2
        return y - mean
        

    def _interpBsplines(self, x, y, r = 1.5, x2 = None):
        '''
        Interpolation using B-splines
        Input:
            - x: current axis being used
            - y: signal to resample
        Optional:
            - r: resampling factor
            - x2: new axis to be used
            
        If x2 is not None, use it as the new axis regarless r
        '''
        tck = splrep(x,y)
        if x2 is None:
            if isinstance(r, (int, long, float, complex)):
                r = float(r)
                x2 = np.linspace(0, len(x) - 1, r*len(x))
#        else:
#            raise "r must be a positive real"
        return splev(x2,tck)    


    def _interpAkima(x,y,r = 1,x2 = None):
        '''
        Interpolation using the Akima's technique
        Input:
            - x: current axis being used
            - y: signal to resample
        Optional:
            - r: resampling factor
            - x2: new axis to be used
            
        If x2 is not None, use it as the new axis regarless r
        '''
        if x2 is None:
            if isinstance(r, (int, long, float, complex)):
                r = float(r)
                x2 = np.linspace(0, len(x)-1, r*len(x))
        return Akima1DInterpolator(x, y, axis = 0).__call__(x2)
