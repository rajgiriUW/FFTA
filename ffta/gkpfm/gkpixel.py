# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:55:14 2019

@author: Raj
"""

import numpy as np
from scipy.optimize import fmin_tnc

import warnings


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

    def __init__(self, signal_array, params, phase=0, ncycles=4):

        self.signal_array = signal_array

        # This functionality is for single lines
        if len(signal_array.shape) > 1:
            warnings.warn('This function only works on 1D (single lines). Flattening..')
            self.signal_array.flatten()

        for key, value in params.items():
            setattr(self, key, value)

        self.ncycles = ncycles
        self.n_points = len(signal_array)

        # single cycle values
        self.pts_per_cycle = int(self.sampling_rate / self.drive_freq)
        self.num_cycles = int(self.n_points / self.pts_per_cycle)
        self.excess = self.n_points % self.pts_per_cycle

        # ncycle values
        self.pts_per_ncycle = ncycles * int(self.sampling_rate / self.drive_freq)
        self.num_ncycles = int(self.n_points / self.pts_per_ncycle)
        self.excess_n = self.n_points % self.pts_per_ncycle

        # create response waveform
        self.txl = np.linspace(0, self.total_time, self.n_points)
        self.resp_wfm = np.sin(self.txl * 2 * np.pi * self.drive_freq + phase)[:self.pts_per_ncycle]

        return

    def cost_func(self, resp_wfm, signal):

        cost = lambda p: np.sum((poly2(resp_wfm, *p) - signal) ** 2)

        pinit = [-1 * np.abs(np.max(signal) - np.min(signal)), 0, 0]

        popt, _, _ = fmin_tnc(cost, pinit, approx_grad=True, disp=0,
                              bounds=[(-10, 10),
                                      (-10, 10),
                                      (-10, 10)])

        return popt

    def analyze(self, verbose=False, fft=False, fast=False):

        #        tx = np.arange(0,self.total_time, self.total_time/len(self.signal_array))
        #        tx_cycle = np.arange(0, self.total_time, self.ncycles * self.total_time/len(self.signal_array))

        CPD = np.zeros(self.num_ncycles)
        capacitance = np.zeros(self.num_ncycles)

        import time
        t0 = time.time()

        for p in np.arange(self.num_ncycles):

            if verbose:
                print('Slice #:', p)

            st = self.pts_per_ncycle * p
            sp = self.pts_per_ncycle * (p + 1)

            # calculate phase, since this is offset at each pixel due to software lag
            if fast:
                ph = self.min_phase_fast(self.signal_array[st:sp])
            else:
                if fft:
                    ph = self.min_phase_fft(self.signal_array[st:sp])
                else:
                    ph = self.min_phase(self.signal_array[st:sp])

            resp_wfm = np.sin(self.txl * 2 * np.pi * self.drive_freq + ph)[:self.pts_per_ncycle]

            popt = self.cost_func(resp_wfm, self.signal_array[st:sp])

            # popt, _ = spo.curve_fit(poly2, resp_wfm, self.signal_array[st:sp],bounds=[(-np.inf, -np.inf, -np.inf), (0, np.inf, np.inf)])
            CPD[p] = -0.5 * popt[1] / popt[0]
            capacitance[p] = popt[0]

        t1 = time.time()

        if verbose:
            print('Total time:', t1 - t0)

        self.CPD = CPD
        self.capacitance = capacitance
        self.CPD_avg = np.mean(CPD)

        return CPD, capacitance, np.mean(CPD)

    def min_phase(self, signal):

        fits = []
        xpts = np.arange(-2 * np.pi, 2 * np.pi, 0.1)

        for i in xpts:
            txl = np.linspace(0, self.total_time, self.n_points)
            resp_wfm = np.sin(txl * 2 * np.pi * self.drive_freq + i)[:len(signal)]

            mid = int(self.pts_per_cycle / 2)

            # find fits for first half-cycle and second half-cycle
            p1 = self.cost_func(resp_wfm[:mid], signal[:mid])
            p2 = self.cost_func(resp_wfm[-mid:], signal[-mid:])

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

def poly2(t, a, b, c):
    return a * t ** 2 + b * t + c
