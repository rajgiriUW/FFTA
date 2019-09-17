# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:55:14 2019

@author: Raj
"""

import numpy as np
from scipy import optimize as spo

class GKPixel:
    '''
    Class for processing G-KPFM data
    
    Process:
        At each pixel, fits a parabola against the first few cycles
        Finds the x-intercept for the peak of the parabola
        Assigns that as the CPD
    
    Parameters:
        signal_array : h5Py Dataset or USIDDataset
        This currently only works on data that is one signal per pixel (i.e already averaged/flattened)
         
    ncycles : int
        number of cantilever cycles to average over
    
    Returns:
        CPD : array
            Array of the calculated CPD values over time 
        capacitance : array
            The curvature of the parabola fit
        CPD_mean : float
            Simple average of the CPD trace, useful for plotting
    '''
    
    def __init__(self, signal_array, params, phase, ncycles = 4):
        
        self.signal_array = signal_array
        
        for key, value in params.items():

            setattr(self, key, value)
        
        self.ncycles = ncycles
        self.n_points = signal_array.shape[0]
        
        # single cycle values
        self.pts_per_cycle = int(self.sampling_rate / self.drive_freq)
        self.num_cycles = int(self.n_points / self.pts_per_cycle)
        self.excess = self.n_points % self.pts_per_cycle
        
        # ncycle values
        self.pts_per_ncycle = ncycles * int(self.sampling_rate / self.drive_freq)
        self.num_ncycles = int(self.n_points / self.pts_per_ncycle)
        self.excess_n = self.n_points % self.pts_per_ncycle
        
        # create response waveform
        self.txl = np.linspace(0, self.total_time, self.signal_array.shape[0])
        self.resp_wfm = np.sin(self.txl * 2 * np.pi * self.drive_freq + phase)[:self.pts_per_ncycle]
        
        return
    
    def analyze(self):
        
#        tx = np.arange(0,self.total_time, self.total_time/len(self.signal_array))
#        tx_cycle = np.arange(0, self.total_time, self.ncycles * self.total_time/len(self.signal_array))
        
        CPD = np.zeros(self.num_ncycles)
        capacitance = np.zeros(self.num_ncycles)
        
        for p in np.arange(self.num_ncycles):
            
            st = self.pts_per_ncycle * p
            sp = self.pts_per_ncycle * (p+1)
            
            # calculate phase, since this is offset at each pixel due to software lag
            ph = self.min_phase(self.signal_array[st:sp])
            resp_wfm = np.sin(self.txl * 2 * np.pi * self.drive_freq + ph)[:self.pts_per_ncycle]
            
#            popt, _ = spo.curve_fit(poly2, self.resp_wfm[st:sp], self.signal_array[st:sp])
            popt, _ = spo.curve_fit(poly2, resp_wfm, self.signal_array[st:sp])
            CPD[p] = -0.5 * popt[1] / popt[0]
            capacitance[p] = popt[0]
        
        return CPD, capacitance, np.mean(CPD)
    
    def min_phase(self,signal):
        
        fits = []
        xpts = np.arange(-2*np.pi, 2*np.pi, 0.1)
        for i in xpts:
            
            txl = np.linspace(0, self.total_time, self.signal_array.shape[0])
            resp_wfm = np.sin(txl * 2 * np.pi * self.drive_freq + i)[:signal.shape[0]]
            
            st = int(self.pts_per_cycle / 2)
            
            p1 = np.polyfit(resp_wfm[:st], signal[:st], 2)
            p2 = np.polyfit(resp_wfm[-st:], signal[-st:], 2)
            
            fit1 = -0.5 * p1[1]/p1[0]
            fit2 = -0.5 * p2[1]/p2[0]
            fits.append(np.abs(fit2-fit1))
        
        ph = xpts[np.argmin(fits)]
        
        resp_wfm = np.sin(txl * 2 * np.pi * self.drive_freq + ph)[:signal.shape[0]]
        p1 = np.polyfit(resp_wfm, signal, 2)
        
        if p1[0] > 0:
            ph = ph + np.pi/2
        
        return ph
        
def poly2(t, a, b, c):
    
    return a*t**2 + b*t + c