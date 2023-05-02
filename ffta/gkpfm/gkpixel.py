# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:55:14 2019

@author: Raj
"""

import warnings
from math import pi

import numpy as np
import numpy.polynomial.polynomial as npPoly
import pyUSID as usid
from BGlib.be.analysis.utils.be_sho import SHOfunc, SHOfit
from igor2.binarywave import load as loadibw
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc
from scipy.signal import fftconvolve, chirp

from ffta.pixel import Pixel
from ffta.pixel_utils.load import cantilever_params
from ffta.simulation.cantilever import Cantilever
from ..analysis.fft import get_noise_floor, NoiseBandFilter
from ..analysis.gmode_utils import test_filter


class GKPixel(Pixel):

    def __init__(self, signal_array,
                 params,
                 can_params={},
                 fit=True,
                 pycroscopy=False,
                 method='hilbert',
                 fit_form='product',
                 filter_amplitude=False,
                 filter_frequency=False,
                 trigger=None,
                 total_time=None,
                 sampling_rate=None,
                 TF_norm=[],
                 exc_wfm=[],
                 periods=2,
                 phase_shift=0):
        '''
        Class for processing G-KPFM data

        Process:
            At each pixel, fits a parabola against the first few cycles
            Finds the x-intercept for the peak of the parabola
            Assigns that as the CPD

        Parameters
        ----------
        See Pixel of parameter defintions
        
        Returns
        -------
        CPD : array
            Array of the calculated CPD values over time
        capacitance : array
            The curvature of the parabola fit
        CPD_mean : float
            Simple average of the CPD trace, useful for plotting
        
        Additional parameters:
            
        :param TF_norm: Transfer function supplied in Shifted Fourier domain, normalized to desired Q
        :type TF_norm: array, optional
        
        :param periods:Number of periods to average over for CPD calc
        :type periods: int
        
        :param phase_shift: Amount to shift the phase of the deflection by (cable lag)
        :type phase_shift : float
            
        '''

        if len(signal_array.shape) > 1:
            warnings.warn('Be sure you are only sending one pixel of data, not an image')

        self.periods = periods
        self.phase_shift = phase_shift

        super().__init__(signal_array, params, can_params,
                         fit, False, method, fit_form,
                         filter_amplitude, filter_frequency,
                         trigger, total_time, sampling_rate)

        # This functionality is for single lines
        if len(self.signal_array.shape) > 1:
            warnings.warn('This function only works on 1D (single lines). Flattening..')
            self.signal_array.flatten()

        self.n_points = len(self.signal_array)

        self.t_ax = np.linspace(0, self.total_time, self.n_points)  # time axis
        self.f_ax = np.linspace(-self.sampling_rate / 2, self.sampling_rate / 2, num=self.n_points)

        self.SIG = np.fft.fftshift(np.fft.fft(self.signal_array))

        self.TF_norm = []
        if any(TF_norm):
            self.TF_norm = TF_norm

        self.exc_wfm = exc_wfm
        if not any(exc_wfm):
            self.excitation()

        return

    def excitation(self, exc_params={}, phase=-pi):
        """
        Generates excitation waveform (AC probing bias) for CPD vs excitation plots

        :param exc_params: Specifies parameters for excitation waveform. Relevant keys are ac (in V), dc (in V),
                phase (in radians), and frequency (in Hz). The default is None, implying an excitation waveform of
                magnitude 1V, with period 1/drive_freq, and 0 DC offset.
        :type exc_params: dict, optional
               
        :param phase: Offset of the excitation waveform in radians. Default is pi.
        :type phase: float, optional
                
        """
        self.exc_params = {'ac': 1, 'dc': 0, 'phase': phase, 'frequency': self.drive_freq}

        for k, v in exc_params.items():
            self.exc_params.update({k: v})

        ac = self.exc_params['ac']
        dc = self.exc_params['dc']
        ph = self.exc_params['phase']
        fr = self.exc_params['frequency']

        self.exc_wfm = (ac * np.sin(self.t_ax * 2 * pi * fr + ph) + dc)

        return

    def excitation_phase(self, exc_path, exc_params={}, phase_range=[-pi, pi]):
        """
        Generates the excitation waveform based on the input ibw.
        
        Calculates phase and runs self.excitation until they match.
        This process is to compensate for the Gage card having a limited input voltage range.
        
        Returns
        -------
        self.exc_wfm
        
        :param exc_path: .ibw path containing the applied DDS excitation to the cantilever from the experiment
        :type exc_path: string
            
        :param exc_params: Specifies parameters for excitation waveform. Relevant keys are ac (in V), dc (in V),
                phase (in radians), and frequency (in Hz). The default is None, implying an excitation waveform of
                magnitude 1V, with period 1/drive_freq, and 0 DC offset.
        :type exc_params: dict, optional
                
        :param phase_range: The start and stop phases to check
        :type phase_range: list, shape 2, optional
                
        """

        phase_test = np.arange(phase_range[0], phase_range[1], 0.1)

        exc_raw = loadibw(exc_path)['wave']['wData']
        exc_raw = exc_raw[:, 10]  # pick a random slice

        for p in phase_test:

            self.excitation(exc_params, p)

            _pke = np.argmax(self.exc_wfm[:50:1])
            _pki = np.argmax(exc_raw[:50:1])

            if _pke == _pki:
                print('Done matching phase', p)
                plt.figure()
                plt.plot(self.exc_wfm[:50], 'r')
                plt.plot(exc_raw[:50] * 100, 'b')

                break

        return

    def excitation_scale(self, exc_path, exc_params):
        """
        Generates the excitation waveform based on the input ibw. Scaled to exc_params
        
        This process is to compensate for the Gage card having a limited input voltage range.
        
        :param exc_path: .ibw path containing the applied DDS excitation to the cantilever from the experiment
        :type exc_path: string
        
        :param exc_params: Specifies parameters for excitation waveform. Relevant keys are ac (in V), dc (in V),
                phase (in radians), and frequency (in Hz). The default is None, implying an excitation waveform of
                magnitude 1V, with period 1/drive_freq, and 0 DC offset.
                example : exc_params = {'ac': 3, 'dc': 3} for 3 Vdc and 3Vac excitation
        :type exc_params: dict
    
        """

        exc_raw = loadibw(exc_path)['wave']['wData'].mean(axis=1)
        exc_raw *= 10  # resistor divider
        exc_raw = (exc_raw - exc_raw.min()) / (exc_raw.max() - exc_raw.min())
        exc_raw = (exc_raw - 0.5) * 2  # scaled -1 to 1
        exc_raw *= exc_params['ac']
        exc_raw += exc_params['dc']

        self.exc_wfm = exc_raw

        return

    def dc_response(self, plot=True):
        """
        Extracts the DC response and plots. For noise-free data this will show
        the expected CPD response
        
        :param plot:
        :type plot: bool
        """
        SIG_DC = np.copy(self.SIG)
        mid = int(len(self.f_ax) / 2)

        self.drive_bin = np.searchsorted(self.f_ax[mid:], self.drive_freq) + mid
        delta_freq = self.sampling_rate / self.n_points
        dc_width = 10e3  # 10 kHz around the DC peak

        SIG_DC[:mid - int(dc_width / delta_freq)] = 0
        SIG_DC[mid + int(dc_width / delta_freq):] = 0
        sig_dc = np.real(np.fft.ifft(np.fft.ifftshift(SIG_DC)))

        if plot:
            plt.figure()
            plt.plot(self.t_ax, sig_dc, 'b')
            plt.title('DC Offset')

        self.sig_dc = sig_dc

        return

    def load_tf(self, tf_path, tf_excitation_path=[],
                remove_dc=False, adjust_tf_length=False):
        '''
        Process transfer function and broadband excitation from supplied file
        This function does not check shape or length
        
        If no tf_excitation_path is provided, this will generate one via SciPy chirp
        
        :param tf_path:
        :type tf_path: str
        
        :param tf_excitation_path:
        :type tf_excitation_path: list
        
        :param remove_dc: Removes the DC peak from the FFT
        :type remove_dc: bool, optional
           
        :param adjust_tf_length:
        :type adjust_tf_length: bool, optional
            
        '''

        # Load the response
        if isinstance(tf_path, str):
            tf = loadibw(tf_path)['wave']['wData']
        else:
            tf = tf_path

        self.tf = tf
        if len(tf.shape) > 1:
            self.tf = np.mean(tf, axis=1)

        # Load the broadband excitation file, or create one
        if tf_excitation_path:

            if isinstance(tf_excitation_path, str):
                tf_exc = loadibw(tf_excitation_path)['wave']['wData']
            else:
                tf_exc = tf_excitation_path
        else:
            tf_exc = gen_chirp(sampling_rate=self.sampling_rate,
                               length=self.tf.shape / self.sampling_rate)

        if remove_dc:
            self.TF[int(len(tf) / 2)] = 0

        self.tf_exc = tf_exc
        if len(tf_exc.shape) > 1:
            self.tf_exc = np.mean(tf_exc, axis=1)

        if adjust_tf_length:
            if self.n_points > len(self.tf):
                self.tf = np.pad(self.tf, (0, self.n_points - len(self.tf)))
            elif self.n_points < len(self.tf):
                self.tf = self.tf[:self.n_points]

        self.TF = np.fft.fftshift(np.fft.fft(self.tf))
        self.TF_EXC = np.fft.fftshift(np.fft.fft(self.tf_exc))

        self.tf_f_ax = np.linspace(-self.sampling_rate / 2, self.sampling_rate / 2, num=self.tf.shape[0])

    def process_tf(self, resonances=2, width=20e3, exc_floor=10, plot=False):
        '''
        
        :param resonances: Number of resonances to fit SHO to. The default is 2.
        :type resonances: int, optional
        
        :param width:  Width of resonance peaks to fit against. The default is 20e3.
        :type width: int, optional
        
        :param exc_floor: Sets the floor for the transfer function, below that is ignored. 
            The default is 10
        :type exc_floor: float, optional
        
        :param plot: Displays fits. The default is False
        :type plot: bool, optional
            
        '''

        #      
        center = int(len(self.tf_f_ax) / 2)
        df = self.sampling_rate / self.tf.shape[0]
        # drive_bin = int(np.ceil(self.drive_freq / df)) 

        # Reconstruct from SHO
        excited_bins = np.where(np.abs(self.TF_EXC) > exc_floor)[0]
        band_edges = tf_fit_mat(self.drive_freq, resonances, width)

        Q_guesses = [self.Q, 3 * self.Q, 6 * self.Q, 9 * self.Q]
        self.coef_mat = np.zeros((resonances, 4))

        TF = np.zeros(len(self.TF))

        if plot:
            fig = plt.figure(10, figsize=(8, int(4 * resonances)), facecolor='white')

            # Constructs effective SHO
        for n, q_guess in enumerate(Q_guesses[:resonances]):

            # find the bins in frequency axis if above exc_floor threshold
            bin_lo, bin_hi = np.ceil(band_edges[n] / df).astype(int) + center
            _exc = np.intersect1d(excited_bins, np.arange(bin_lo, bin_hi))
            band = self.f_ax[_exc]
            response = self.TF[bin_lo:bin_hi]

            if not any(band):
                _msg = 'Ignoring resonance ' + str(n) + ' outside excitation band'
                warnings.warn(_msg)
                break

            # initial guesses    
            a_guess = np.max(np.abs(response)) / q_guess
            w_guess = band[int(len(band) / 2)]
            phi_guess = -pi
            coef_guess = [np.real(a_guess), w_guess, q_guess, phi_guess]

            # SHO fit
            # coef = SHOestimateGuess(response, band, 10)
            coef = SHOfit(coef_guess, band, response)
            self.coef_mat[n, :] = coef
            response_guess = SHOfunc(coef_guess, band)
            response_fit = SHOfunc(coef, band)
            response_full = SHOfunc(coef, self.tf_f_ax)
            TF = TF + response_full

            if plot:
                plt.subplot(resonances, 2, n + 1)
                plt.plot(band / 1e6, np.abs(response), 'b.-')
                plt.plot(band / 1e6, np.abs(response_guess), 'g')
                plt.plot(band / 1e6, np.abs(response_fit), 'r')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Amplitude (nm)')
                plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

                plt.subplot(resonances, 2, (n + 1) + 2)
                plt.plot(band / 1e6, np.angle(response), '.-')
                plt.plot(band / 1e6, np.angle(response_guess), 'g')
                plt.plot(band / 1e6, np.angle(response_fit), 'r')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Phase (Rad)')
                plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        # Normalize to first resonance Q
        self.TF_norm = self.coef_mat[0, 2] * (TF - np.min(np.abs(TF))) / \
                       (np.max(np.abs(TF)) - np.min(np.abs(TF)))

        return

    def generate_tf(self, can_params_dict={}, plot=False):
        """
        Uses the cantilever simulation to generate a tune as the transfer function
        
        :param can_params_dict: use ffta.pixel_utils.load.cantilever_params()
        :type can_params_dict: Dict
            
        :param plot: Plots the time-dependent tune
        :type plot: bool

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

        force_params = {'es_force': 1e-10,
                        'ac_force': 6e-07,
                        'dc_force': 3e-09,
                        'delta_freq': -170.0,
                        'tau': 0.001,
                        'v_dc': 3.0,
                        'v_ac': 2.0,
                        'v_cpd': 1.0,
                        'dCdz': 1e-10,
                        'v_step': 1.0}
        sim_params = {'trigger': 0.02,
                      'total_time': 0.05
                      }

        for k, v in can_params_dict.items():
            can_params.update(k=v)

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
                    can_params.update(l=can_params_dict[l])

        if can_params['k'] < 1e-3:
            can_params['k'] *= 1e9  # old code had this off by 1e9

        cant = Cantilever(can_params, force_params, sim_params)
        cant.trigger = cant.total_time  # don't want a trigger
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

        TF_norm = Q * (TF - np.min(np.abs(TF))) / (TFmax - np.min(np.abs(TF)))

        self.tf = Z
        self.TF = TF
        self.TF_norm = TF_norm

        self.tf_f_ax = np.linspace(-self.sampling_rate / 2, self.sampling_rate / 2, num=self.tf.shape[0])

        self.tf_exc = gen_chirp(sampling_rate=self.sampling_rate,
                                length=self.tf.shape / self.sampling_rate)
        self.TF_EXC = np.fft.fftshift(np.fft.fft(self.tf_exc))

        return

    def force_out(self, plot=False, noise_tolerance=1e-6, phase_shift=0):
        """
        Reconstructs force by dividing by transfer function

        :param plot: Generates plot of reconstructed force. The default is False. 
        :type plot: bool, optional
        
        :param noise_tolerance: Use to determine noise_floor, The default is 1e-6
        :type noise_tolerance: float, optional
        
        :param phase_shift: Desired phase shift in radians
        :type phase_shift: float, optional

        """
        if not any(self.TF_norm):
            raise AttributeError('Supply Transfer Function or use generate_tf()')

        center = int(len(self.SIG) / 2)
        drive_bin = int(self.drive_freq / (self.sampling_rate / len(self.SIG)))

        SIG = np.copy(self.SIG)

        if phase_shift != 0:
            self.phase_shift = phase_shift

        if self.phase_shift != 0:
            # DFT shift theorem
            period = self.sampling_rate / self.drive_freq
            ph = self.phase_shift * period / (2 * pi)
            SIG = self.SIG * np.exp(-1j * ph * self.f_ax / (0.5 * len(self.f_ax)))

        self.FORCE = np.zeros(len(SIG), dtype=complex)

        noise_limit = np.ceil(get_noise_floor(SIG, noise_tolerance))

        # Only save bins above the noise_limit
        signal_pass = np.where(np.abs(SIG) > noise_limit)[0]

        if 2 * drive_bin + center not in signal_pass:
            warnings.warn('Second resonance not in passband; increase noise_tolerance')

        self.FORCE[signal_pass] = SIG[signal_pass]
        self.FORCE = self.FORCE / self.TF_norm
        self.force = np.real(np.fft.ifft(np.fft.ifftshift(self.FORCE)))

        del SIG

        if plot:
            start = int(0.5 * self.trigger * self.sampling_rate)
            stop = int(1.5 * self.trigger * self.sampling_rate)
            plt.figure()
            plt.plot(self.t_ax[start:stop], self.force[start:stop])
            plt.title('Force (output/TF_norm) vs time, near trigger')
            plt.xlabel('Time (s)')
            plt.ylabel('Force (N)')

            self.plot_response()

        return

    def noise_filter(self, bw=1e3, plot=True, noise_tolerance=1e-6):
        """
        Denoising filter for 50 kHz harmonics (electrical noise in the system)
        
        :param bw: Bandwidth for the notch filters
        :type bw: float, optional
        
        :param plot: Generates plot of reconstructed force. The default is False. 
        :type plot: bool, optional
        
        :param noise_tolerance: Use to determine noise_floor, The default is 1e-6
        :type noise_tolerance: float, optional
        
        """
        nbf = NoiseBandFilter(len(self.force), self.sampling_rate,
                              [2E3, 50E3, 100E3, 150E3, 200E3],
                              [4E3, bw, bw, bw, bw])

        filt_line, _, _ = test_filter(self.force,
                                      frequency_filters=nbf,
                                      noise_threshold=noise_tolerance,
                                      show_plots=plot)
        self.force = np.real(filt_line)
        self.FORCE = np.fft.fftshift(np.fft.fft(self.force))

        return

    def plot_response(self):
        """
        Plots the transfer function and calculated force in frequency space

        """
        plt.figure()
        plt.semilogy(self.f_ax, np.abs(self.SIG), 'b')
        plt.semilogy(self.f_ax, np.abs(self.TF_norm), 'g')
        plt.semilogy(self.f_ax, np.abs(self.FORCE), 'k')
        plt.xlim((0, 2.5 * self.drive_freq))
        plt.legend(labels=['Signal', 'TF-normalized', 'Force_out'])
        plt.title('Frequency Response of the Data')

        return

    def _calc_cpd_params(self, periods=2, return_dict=False):
        """
        Calculates the parameters needed to calculate the CPD

        :param periods: Number of cantilever cycles to average over. The default is 2.
        :type periods: int, optional
        
        :param return_dict: Dictionary of these parameters for debugging purposes
        :type return_dict: bool, optional
        
        :returns: if return_dict == True, return _cpdd
        :rtype:
        
        """

        self.periods = periods

        self.pxl_time = self.n_points / self.sampling_rate  # how long each pixel is in time (8.192 ms)
        self.time_per_osc = (1 / self.drive_freq)  # period of drive frequency
        self.pnts_per_period = self.sampling_rate * self.time_per_osc  # points in a cycle 
        self.num_periods = int(self.pxl_time / self.time_per_osc)  # number of periods in each pixel

        self.num_CPD = int(np.floor(
            self.num_periods / self.periods))  # length of CPD array since each CPD takes some number of periods
        self.pnts_per_CPD = int(np.floor(self.pnts_per_period * self.periods))  # points used to calculate CPD
        self.remainder = int(self.n_points % self.pnts_per_CPD)

        if return_dict:
            _cpdd = {'pxl_time': self.pxl_time,
                     'time_per_osc': self.time_per_osc,
                     'pnts_per_period': self.pnts_per_period,
                     'num_periods': self.num_periods,
                     'num_CPD': self.num_CPD,
                     'pnts_per_CPD': self.pnts_per_CPD,
                     'remainder': self.remainder}
            return _cpdd

        return

    def analyze_cpd(self, verbose=False, deg=2, use_raw=False, periods=2,
                    overlap=False):
        """
        Extracts CPD and capacitance gradient from data.

        :param verbose:
        :type verbose: bool

        :param deg: Degree of polynomial fit. Default is 2, which is a quadratic fit.
            Unless there's a good reason, quadratic is correct to use
        :type deg: int
                
        :param use_raw: Uses the signal_array instead of the reconstructed force
        :type use_raw: bool, optional
                
        :param periods: Numer of cantilever cycles to average over for CPD extraction
        :type periods: int, optional
                
        :param overlap: If False, each CPD is from a separate part of the signal. 
            If True, shifts signal by 1 pixel and recalculates
        
        :type overlap: bool, optional
                
        """

        self._calc_cpd_params(periods)

        pnts = self.pnts_per_CPD
        step = pnts

        if overlap:
            self.t_ax_wH = np.copy(self.t_ax)
            step = 1

        cpd_px = np.arange(0, self.n_points, step)
        test_wH = np.zeros((len(cpd_px), deg + 1))

        for n, p in enumerate(cpd_px):

            if use_raw:

                resp_x = np.float32(self.signal_array[p:p + pnts])

            else:

                resp_x = np.float32(self.force[p:p + pnts])

            resp_x -= np.mean(resp_x)

            V_per_osc = self.exc_wfm[p:p + pnts]

            popt, _ = npPoly.polyfit(V_per_osc, resp_x, deg, full=True)
            test_wH[n] = popt.flatten()

        self.test_wH = test_wH
        self.CPD = -0.5 * test_wH[:, 1] / test_wH[:, 2]
        self.capacitance = test_wH[:, 2]

        if any(np.argwhere(np.isnan(self.CPD))):
            self.CPD[-1] = self.CPD[-2]
            self.capacitance[-1] = self.capacitance[-2]

    def plot_cpd(self, smooth=None):
        '''
        Plots the CPD response

        :param smooth: Boxcar smoothign kernel. Value of 3 is reasonable
        :type smooth: int, optional
            
        :returns:
        :rtype: matplotlib figure axis object

        '''
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        tx = np.linspace(0, self.total_time, self.num_CPD)

        if smooth:
            ax.plot(tx * 1e3, fftconvolve(self.CPD[:self.num_CPD], np.ones(smooth) / 3, mode='same'), 'b')
        else:
            ax.plot(tx * 1e3, self.CPD[:self.num_CPD], 'b')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('CPD (V)')
        ax.set_title('CPD response')

        return ax

    def filter_cpd(self):
        """
        Filters the capacitance based on pixel parameter self.filter_bandwidth
        (typical is 10 kHz, which is somewhat large)
        """
        center = int(len(self.CPD) / 2)
        df = self.sampling_rate / len(self.CPD)
        bin_width = int(self.filter_bandwidth / df)

        return

    def min_phase(self, phases_to_test=[2.0708, 2.1208, 2.1708],
                  noise_tolerance=1e-6, verbose=True):
        """
        Determine the optimal phase shift due to cable lag
        
        :param phases_to_test: Which phases to shift the signal with. The default is [2.0708, 2.1208, 2.1708],
            which is 0.5, 0.55, 0.5 + pi/2
        :type phases_to_test: list, optional
        
        :param noise_tolerance: Use to determine noise_floor, The default is 1e-6
        :type noise_tolerance : float, optional

        """

        # have to iterate this cell many times to find the right phase
        phases_to_test = np.array(phases_to_test)

        numplots = len(phases_to_test)
        fig, ax = plt.subplots(nrows=numplots, figsize=(6, int(4 * numplots)),
                               facecolor='white')

        for x, ph in enumerate(phases_to_test):

            self.phase_shift = ph
            self.force_out(plot=False, noise_tolerance=noise_tolerance)

            if len(phases_to_test) > 1:
                usid.plot_utils.rainbow_plot(ax[x], self.exc_wfm, self.force)
                ax[x].set_title('Phase=' + str(ph))
            else:
                usid.plot_utils.rainbow_plot(ax, self.exc_wfm, self.force)
                ax.set_title('Phase=' + str(ph))

        if verbose:
            print('Set self.phase_shift to match desired phase offset (radians)')

        return

    def min_phase_fft(self, signal):
        """
        :param signal:
        :type signal:
        
        :returns:
        :rtype:
    
        """
        fits = []
        xpts = np.arange(-2 * pi, 2 * pi, 0.1)
        fs = np.fft.fft(signal)
        idx = np.argmax(np.abs(fs))
        for i in xpts:
            txl = np.linspace(0, self.total_time, self.n_points)
            resp_wfm = np.sin(txl * 2 * pi * self.drive_freq + i)[:len(signal)]

            fr = np.fft.fft(resp_wfm)
            fits.append(np.angle(fr / fs)[idx])

        fits = np.array(fits)
        ph_test = np.abs(fits).argsort()[:6]  # sorted index from least to greatest
        ph_test = np.append(ph_test, np.abs(fits).argsort()[-6:])

        fitsp = []
        for i in ph_test:
            txl = np.linspace(0, self.total_time, self.n_points)
            resp_wfm = np.sin(txl * 2 * pi * self.drive_freq + i)[:len(signal)]

            mid = int(self.pts_per_cycle / 2)

            # find fits for first half-cycle and second half-cycle

            p1 = self.cost_func(resp_wfm[:mid], signal[:mid])
            p2 = self.cost_func(resp_wfm[-mid:], signal[-mid:])

            fit1 = -0.5 * p1[1] / p1[0]
            fit2 = -0.5 * p2[1] / p2[0]

            fitsp.append(np.abs(fit2 - fit1))

        ph = xpts[ph_test[np.argmin(fitsp)]]

        return ph


# Support functions
def poly2(t, a, b, c):
    """
    :param t:
    :type t: float
    
    :param a:
    :type a: float
    
    :param b:
    :type b: float
    
    :param c:
    :type b: float
    
    :returns:
    :rtype: float
    """
    return a * t ** 2 + b * t + c


def cost_func(resp_wfm, signal):
    """
    :param resp_wfm:
    :type resp_wfm:
    
    :param signal:
    :type signal:
    
    :returns:
    :rtype:
    """
    cost = lambda p: np.sum((poly2(resp_wfm, *p) - signal) ** 2)

    pinit = [-1 * np.abs(np.max(signal) - np.min(signal)), 0, 0]

    popt, _, _ = fmin_tnc(cost, pinit, approx_grad=True, disp=0,
                          bounds=[(-10, 10),
                                  (-10, 10),
                                  (-10, 10)])

    return popt


def tf_fit_mat(drive_freq, resonances=2, width=20e3):
    """
    :param drive_freq:
    :type drive_freq:
    
    :param resonances:
    :type resonances: int
    
    :param width:
    :type width: float
    
    :returns:
    :rtype: numpy nd.array
    """
    # Cantilever resonances, see Table 1 in doi:10.1016/j.surfrep.2005.08.003
    eigen_factors = [1, 6.255, 17.521, 34.33]

    band_edge_mat = [np.array([drive_freq * e - width, drive_freq * e + width])
                     for e in eigen_factors[:resonances]]

    return np.array(band_edge_mat)


def gen_chirp(f_center=500e3, f_width=450e3, length=1e-2, sampling_rate=1e7):
    '''
    Generates a single broad-frequency signal using scipy chirp, writes to name.dat
    
    :param f_center: Central frequency for the signal
    :type f_center: float
        
    :param f_width: The single-sided width of the chirp. Generates signal from f_center - f_width to f_center + f_width
    :type f_width : float
        
    :param length: the timescale of the signal. Keep this length in mind for data acquisition;
        if your chirp is longer than your data acquisition, you will miss many of 
        the frequencies
    :type length: float
        
    :param sampling_rate: Sampling rate of the chirp, based on length/sampling_rate number of steps
        This rate must be consistent on the wave generator or the frequencies will
        be off 
    :type sampling_rate: int
    
    :returns:
    :rtype:
               
    '''

    if isinstance(sampling_rate, str):
        sampling_rate = float(sampling_rate)

    tx = np.arange(0, length, 1 / sampling_rate)  # fixed 100 MHz sampling rate for 10 ms

    f_hi = f_center + f_width
    f_lo = np.max([f_center - f_width, 1])  # to ensure a positive number

    gen_chirp = chirp(tx, f_lo, tx[-1], f_hi)

    return gen_chirp
