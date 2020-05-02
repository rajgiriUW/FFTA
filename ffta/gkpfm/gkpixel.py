# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:55:14 2019

@author: Raj
"""

import numpy as np
import numpy.polynomial.polynomial as npPoly
from scipy.optimize import fmin_tnc

import warnings

def poly2(t, a, b, c):
    return a * t ** 2 + b * t + c

def cost_func(resp_wfm, signal):

    cost = lambda p: np.sum((poly2(resp_wfm, *p) - signal) ** 2)

    pinit = [-1 * np.abs(np.max(signal) - np.min(signal)), 0, 0]

    popt, _, _ = fmin_tnc(cost, pinit, approx_grad=True, disp=0,
                          bounds=[(-10, 10),
                                  (-10, 10),
                                  (-10, 10)])

    return popt

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

    def __init__(self, signal_array, params, periods = 2):

        self.signal_array = signal_array

        # This functionality is for single lines
        if len(signal_array.shape) > 1:
            warnings.warn('This function only works on 1D (single lines). Flattening..')
            self.signal_array.flatten()

        for key, value in params.items():
            setattr(self, key, value)

        self.n_points = len(signal_array)
        self.periods = periods
        
        self.pxl_time = self.n_points/self.sampling_rate   # how long each pixel is in time (8.192 ms)
        self.time_per_osc = (1/self.drive_freq) # period of drive frequency 
        self.pnts_per_period = self.sampling_rate * self.time_per_osc  # points in a cycle 
        self.num_periods = int(self.pxl_time/self.time_per_osc)  # number of periods in each pixel
        
        self.num_periods_per_CPD = int(np.floor(self.num_periods / self.periods))  # number of CPD samples, since each CPD takes some number of periods
        self.pnts_per_CPD_sample = int(np.floor(self.pnts_per_period * self.periods))  # points per CPD sample
        self.pnts_per_CPD_total = int(self.n_points / self.pnts_per_CPD_sample)
        self.remainder = int(self.n_points % self.pnts_per_CPD_sample)

        self.txl = np.linspace(0, self.total_time, self.n_points)
        
        self.excitation()
        
        return
    
    def excitation(self, exc_params={}, phase=-np.pi):
        
        self.exc_params = {'ac':1, 'dc':0, 'phase':phase, 'frequency':self.drive_freq}
        
        for k,v in exc_params.items():
            self.exc_params.update({k:v})       

        ac = self.exc_params['ac']
        dc = self.exc_params['dc']
        ph = self.exc_params['phase']
        fr = self.exc_params['frequency']

        self.resp_wfm = (ac*np.sin(self.txl * 2 * np.pi * fr + ph) + dc)
        
        return

    def analyze(self, verbose=False, fft=False, fast=False, deg = 2):

        #        tx = np.arange(0,self.total_time, self.total_time/len(self.signal_array))
        #        tx_cycle = np.arange(0, self.total_time, self.ncycles * self.total_time/len(self.signal_array))

        pnts_per_CPD = self.pnts_per_CPD_total
        pnts = self.pnts_per_CPD_sample
        remainder = self.remainder
        
        tx = np.linspace(0, pnts_per_CPD / self.sampling_rate, pnts_per_CPD)
        
        test_wH = np.zeros((pnts_per_CPD, deg+1))
        
        for p in range(pnts_per_CPD-min(1,remainder)):
        
            resp_x = np.float32(self.signal_array[pnts*p:pnts*(p+1)])
            resp_x -= np.mean(resp_x)
            
            V_per_osc = self.resp_wfm[pnts*p:pnts*(p+1)]
            #V_per_osc = excitation[:decimation] # testing single fit
                    
            popt, _ = npPoly.polyfit(V_per_osc, resp_x, deg, full=True)
            test_wH[p] = popt
       
        if remainder > 0:
            resp_x = np.float32(self.signal_array[(pnts_per_CPD-1)*pnts:])
            resp_x -= np.mean(resp_x)
            
            V_per_osc = self.resp_wfm[(pnts_per_CPD-1)*pnts:]
    
            popt, _ = npPoly.polyfit(V_per_osc, resp_x, deg, full=True)
            
            test_wH[-1,:] = popt
       
        self.test_wH = test_wH
        self.CPD =  -0.5 * test_wH[:,1]/test_wH[:,2]
        self.capacitance = test_wH[:,0]

        return 

    def min_phase(self, signal):

        fits = []
        errors = []
        xpts = np.arange(-2 * np.pi, 2 * np.pi, 0.1)

        for i in xpts:
            txl = np.linspace(0, self.total_time, self.n_points)
            resp_wfm = np.sin(txl * 2 * np.pi * self.drive_freq + i)[:len(signal)]

            mid = int(self.pts_per_cycle / 2)

            # find fits for first half-cycle and second half-cycle
            p1 = cost_func(resp_wfm[:mid], signal[:mid])
            p2 = cost_func(resp_wfm[-mid:], signal[-mid:])

            fit1 = -0.5 * p1[1] / p1[0]
            fit2 = -0.5 * p2[1] / p2[0]

            fits.append(np.abs(fit2 - fit1))

        return xpts[np.argmin(fits)]

    def min_phase_fft(self, signal):

        fits = []
        xpts = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
        fs = np.fft.fft(signal)
        idx = np.argmax(np.abs(fs))
        for i in xpts:
            txl = np.linspace(0, self.total_time, self.n_points)
            resp_wfm = np.sin(txl * 2 * np.pi * self.drive_freq + i)[:len(signal)]

            fr = np.fft.fft(resp_wfm)
            fits.append(np.angle(fr / fs)[idx])

        fits = np.array(fits)
        ph_test = np.abs(fits).argsort()[:6]  # sorted index from least to greatest
        ph_test = np.append(ph_test, np.abs(fits).argsort()[-6:])

        fitsp = []
        for i in ph_test:
            txl = np.linspace(0, self.total_time, self.n_points)
            resp_wfm = np.sin(txl * 2 * np.pi * self.drive_freq + i)[:len(signal)]

            mid = int(self.pts_per_cycle / 2)

            # find fits for first half-cycle and second half-cycle

            p1 = self.cost_func(resp_wfm[:mid], signal[:mid])
            p2 = self.cost_func(resp_wfm[-mid:], signal[-mid:])

            fit1 = -0.5 * p1[1] / p1[0]
            fit2 = -0.5 * p2[1] / p2[0]

            fitsp.append(np.abs(fit2 - fit1))

        ph = xpts[ph_test[np.argmin(fitsp)]]

        return ph

    def min_phase_fast(self, signal):

        fs = np.fft.fft(signal)
        idx = np.argmax(np.abs(fs))
        ph = np.angle(fs)[idx]

        return ph


