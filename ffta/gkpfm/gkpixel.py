# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:55:14 2019

@author: Raj
"""

import numpy as np
import numpy.polynomial.polynomial as npPoly
from scipy.optimize import fmin_tnc
from matplotlib import pyplot as plt
from ffta.simulation.cantilever import Cantilever
from ffta.pixel_utils.load import cantilever_params
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

    def __init__(self, signal_array, params, TF_norm=[], periods = 2):
        '''
        Class for processing G-KPFM data

        Process:
            At each pixel, fits a parabola against the first few cycles
            Finds the x-intercept for the peak of the parabola
            Assigns that as the CPD

        Parameters:
            signal_array : h5Py Dataset or USIDDataset
                This currently only works on data that is one signal per pixel (i.e already averaged/flattened)
            params : dict
                Specifies parameters for analysis. Should include drive_freq, sampling_rate, total_time.
            TF_norm : array, optional
                Transfer function supplied in Shifted Fourier domain, normalized to desired Q
            periods: int
                Number of periods to average over

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
        
        self.num_CPD = int(np.floor(self.num_periods / self.periods))  # number of CPD samples, since each CPD takes some number of periods
        self.pnts_per_CPD = int(np.floor(self.pnts_per_period * self.periods))  # points used to calculate CPD
        self.remainder = int(self.n_points % self.pnts_per_CPD)

        self.t_ax = np.linspace(0, self.total_time, self.n_points) #time axis
        self.f_ax = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, num=self.n_points)
        
        self.excitation()
        
        self.SIG = np.fft.fftshift(np.fft.fft(signal_array))
        self.TF_norm = []
        if any(TF_norm):
            self.TF_norm = TF_norm
        
        return
    
    def excitation(self, exc_params={}, phase=-np.pi):
        """
        Generates excitation waveform (AC probing bias)
        Parameters:
            exc_params: dict, optional
                Specifies parameters for excitation waveform. Relevant keys are ac (in V), dc (in V),
                phase (in radians), and frequency (in Hz). The default is None, implying an excitation waveform of
                magnitude 1V, with period 1/drive_freq, and 0 DC offset.
            phase: float, optional
                Offset of the excitation waveform in radians. Default is pi.
        """
        self.exc_params = {'ac':1, 'dc':0, 'phase':phase, 'frequency':self.drive_freq}
        
        for k,v in exc_params.items():
            self.exc_params.update({k:v})       

        ac = self.exc_params['ac']
        dc = self.exc_params['dc']
        ph = self.exc_params['phase']
        fr = self.exc_params['frequency']

        self.exc_wfm = (ac*np.sin(self.t_ax * 2 * np.pi * fr + ph) + dc)
        
        return

    def dc_response(self, plot=True):
        """
        Extracts the DC response and plots. For noise-free data this will show
        the expected CPD response
        """
        SIG_DC = np.copy(self.SIG)
        mid = int(len(self.f_ax) / 2)
        
        self.drive_bin = np.searchsorted(self.f_ax[mid:], self.drive_freq) + mid
        delta_freq = self.sampling_rate / self.n_points
        dc_width = 10e3 # 10 kHz around the DC peak
        
        SIG_DC[:mid-int(dc_width / delta_freq) ] = 0
        SIG_DC[mid+int(dc_width / delta_freq):] = 0
        sig_dc = np.real(np.fft.ifft(np.fft.ifftshift(SIG_DC)))
        
        if plot:
            plt.figure()
            plt.plot(self.t_ax, sig_dc)
            plt.title('DC Offset')
        
        self.sig_dc = sig_dc
        
        return

    def generate_tf(self, can_params_dict={}, plot=False):
        """
        Uses the cantilever simulation to generate a tune as the transfer function
        
        can_params_dict : Dict
            use ffta.pixel_utils.load.cantilever_params()
        
        plot : bool
            Plots the time-dependent tune

        Returns
        -------
        None.

        """
        if isinstance(can_params_dict, str):
            can_params_dict = cantilever_params(can_params_dict)
            can_params_dict = can_params_dict['Initial']
        
        can_params = {'amp_invols': 7.5e-08,
                      'def_invols': 6.88e-08,
                      'soft_amp': 0.3,
                      'drive_freq': 309412.0,
                      'res_freq': 309412.0,
                      'k': 43.1,
                      'q_factor': 340.0}
        
        force_params =  {'es_force': 1e-10,
                         'ac_force': 6e-07,
                         'dc_force': 3e-09,
                         'delta_freq': -170.0,
                         'tau': 0.001,
                         'v_dc': 3.0,
                         'v_ac': 2.0,
                         'v_cpd': 1.0,
                         'dCdz': 1e-10,
                         'v_step': 1.0}
        sim_params =  {'trigger': 0.02, 
                       'total_time': 0.05 
                       }
        
        for k, v in can_params_dict.items():
            can_params.update(k= v)
        
        # Update from GKPixel class
        sim_params['trigger'] = self.trigger
        sim_params['sampling_rate'] = self.sampling_rate
        sim_params['total_time'] = self.total_time
        sim_params['sampling_rate'] = self.sampling_rate
        can_params['drive_freq'] = self.drive_freq
        can_params['res_freq'] = self.drive_freq
 
        force_keys = ['es_force']
        can_keys = {'amp_invols': ['amp_invols', 'AMPINVOLS'], 
                    'q': ['q_factor', 'Q'],
                    'k': ['SpringConstant', 'k']}
        
        for f in force_keys:
            if 'Force' in can_params_dict:
                force_params.update(es_force=can_params_dict['Force'])
            elif 'es_force' in can_params_dict:
                force_params.update(es_force=can_params_dict['es_force'])
        
        for c in ['amp_invols', 'q', 'k']:
            for l in can_keys[c]:
                if l in can_params_dict:
                    can_params.update(l = can_params_dict[l])
                        
        if can_params['k'] < 1e-3:
            can_params['k'] *= 1e9 #old code had this off by 1e9
        
        cant = Cantilever(can_params, force_params, sim_params)
        cant.trigger = cant.total_time # don't want a trigger
        Z, _ = cant.simulate()
        Z = Z.flatten()
        if plot:
            plt.figure()
            plt.plot(Z)
            plt.title('Tip response)')

        TF = np.fft.fftshift(np.fft.fft(Z))
        
        Q = can_params['q_factor']
        mid = int(len(self.f_ax) / 2)
        drive_bin = np.searchsorted(self.f_ax[mid:], self.drive_freq) + mid
        TFmax = np.abs(TF[drive_bin])
        
        TF_norm = Q * (TF- np.min(np.abs(TF))) / (TFmax - np.min(np.abs(TF)))

        self.tf = Z
        self.TF = TF
        self.TF_norm = TF_norm

        return

    def force_out(self, plot=False):
        """
        Reconstructs force by dividing by transfer function

        Parameters
        ----------
        plot : bool, optional
            Generates plot of reconstructed force. The default is False.

        Returns
        -------
        None.

        """
        if not any(self.TF_norm):
            raise AttributeError('Supply Transfer Function or use generate_tf()')
            
        self.FORCE = self.SIG / self.TF_norm
        self.force = np.real(np.fft.ifft(np.fft.ifftshift(self.FORCE)))
    
        if plot:
            start = int(0.5 * self.trigger * self.sampling_rate)
            stop = int(1.5 * self.trigger * self.sampling_rate)
            plt.figure()
            plt.plot(self.t_ax[start:stop], self.force[start:stop])
            plt.title('Force (output/TF_norm) vs time')
            plt.xlabel('Time (s)')
            plt.ylabel('Force (N)')
    
        return 
    
    def plot_response(self):
        
        plt.figure()
        plt.semilogy(self.f_ax, np.abs(self.SIG), 'b')
        plt.semilogy(self.f_ax, np.abs(self.TF_norm), 'g')
        plt.semilogy(self.f_ax, np.abs(self.FORCE), 'k')
        plt.xlim((0, 2.5*self.drive_freq))
        plt.legend(labels=['Signal', 'TF-normalized', 'Force_out'])
        plt.title('Frequency Response of the Data')

        return

    def analyze(self, verbose=False, deg = 2):
        """
        Extracts CPD and capacitance gradient from data.
        Parameters:
            verbose: bool

            deg: int
                Degree of polynomial fit. Default is 2, which is a quadratic fit.
        """

        #        tx = np.arange(0,self.total_time, self.total_time/len(self.signal_array))
        #        tx_cycle = np.arange(0, self.total_time, self.ncycles * self.total_time/len(self.signal_array))

        num_CPD = self.num_CPD
        pnts = self.pnts_per_CPD

        self.t_ax_wH = np.linspace(0, self.periods*self.time_per_osc*self.num_CPD, num_CPD) #time ax for CPD/capacitance
        test_wH = np.zeros((num_CPD, deg+1))
        
        for p in range(num_CPD):

            resp_x = np.float32(self.signal_array[pnts*p:pnts*(p+1)])
            resp_x -= np.mean(resp_x)
            
            V_per_osc = self.exc_wfm[pnts*p:pnts*(p+1)]
            #V_per_osc = excitation[:decimation] # testing single fit
                    
            popt, _ = npPoly.polyfit(V_per_osc, resp_x, deg, full=True)
            test_wH[p] = popt.flatten()
       
        self.test_wH = test_wH
        self.CPD =  -0.5 * test_wH[:,1]/test_wH[:,2]
        self.capacitance = test_wH[:,2]

        return 

    def min_phase(self, phases_to_test = [2.0708, 2.1208, 2.1708]):
        """
        Determine the optimal phase shift due to cable lag
        
        Parameters
        ----------
        phases_to_test : list, optional
            Which phases to shift the signal with. The default is [2.0708, 2.1208, 2.1708],
            which is 0.5, 0.55, 0.5 + pi/2

        Returns
        -------
        None.

        """
        # have to iterate this cell many times to find the right phase
        phases_to_test = np.array(phases_to_test)
        
        start = int(0.5 * self.trigger)
        stop = int(1.5 * self.trigger)
        
        mid = int(len(self.f_ax) / 2)
        drive_bin = np.searchsorted(self.f_ax[mid:], self.drive_freq) + mid
        
        # that processes is very slow for large datasets
        noise_floor = px.processing.fft.get_noise_floor(self.signal_array, 1e-6)[0]
        Noiselimit = np.ceil(noise_floor)
        
        fig, a = plt.subplots(nrows=3, figsize=(6, 14))
        
        for x, ph in zip(range(len(phases_to_test)), phases_to_test):
           
            SIG_shifted = self.SIG * np.exp(-1j * self.f_Z/self.f_Z[drive_bin] * ph)
            Gout_shifted = SIG_shifted / self.TF_norm
            gout_shifted = np.real(np.fft.ifft(np.fft.ifftshift(Gout_shifted)))
        
            a[x].plot(self.exc_wfm[start:start+1000], gout_shifted[start:start+1000], 'b')
            a[x].plot(self.exc_wfm[stop:stop+1000], gout_shifted[stop:stop+1000], 'r')

        return

    def min_phase_old(self, signal):

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
