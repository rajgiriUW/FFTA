"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2023"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import time
import warnings

import numpy
import pywt
from matplotlib import pyplot as plt
from scipy import integrate as spg
from scipy import signal as sps

import cupy as cp
# import cusignal

from ffta.nfmd.cuNFMD import CUNFMD
from ffta.pixel_utils import dwavelet
from ffta.pixel_utils import parab
from ffta.pixel_utils import tfp_calc

from ffta.FFTA import FFTA
import cupy as cp
from cupyx.scipy import signal as cps

class CUFFTA(FFTA):

    def __init__(self,
                 signal_array,
                 params={},
                 can_params={},
                 fit=True,
                 pycroscopy=False,
                 method='hilbert',
                 fit_form='product',
                 filter_amplitude=False,
                 filter_frequency=False,
                 recombination=False,
                 trigger=None,
                 total_time=None,
                 sampling_rate=None,
                 roi=None):
        """
        Signal Processing to Extract Time-to-First-Peak. This version enables
        GPU-computation via cuda (cupy and cupyx)

        Extracts Time-to-First-Peak (tFP) from digitized Fast-Free Time-Resolved
        Electrostatic Force Microscopy (FF-trEFM) signals [1-2]. It includes a few
        types of frequency analysis:

        a) Hilbert Transform
        b) Wavelet Transform
        c) Short-Time Fourier Transform (STFT)
        d) Non-stationary Fourier mode decomposition (NFMD)
            
        """
        super().__init__(signal_array,
                                   params,
                                   can_params,
                                   fit,
                                   pycroscopy,
                                   method,
                                   fit_form,
                                   filter_amplitude,
                                   filter_frequency,
                                   recombination,
                                   trigger,
                                   total_time,
                                   sampling_rate,
                                   roi)
        
        # Creature CUDA arrays
        self.signal_array = cp.array(self.signal_array)
        self.signal = cp.array(self.signal)
        
        return

    def apply_window(self):
        """
        Applies the window given in parameters.
        """

        self.signal *= cps.get_window(self.window, self.n_points)

        return

    def fir_filter(self):
        """
        Filters signal with a FIR bandpass filter.
        """

        # Calculate bandpass region from given parameters.
        nyq_rate = 0.5 * self.sampling_rate
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / nyq_rate
        freq_high = (self.drive_freq + bw_half) / nyq_rate

        band = [freq_low, freq_high]

        # Create taps using window method.
        try:
            taps = cps.firwin(int(self.n_taps), band, pass_zero=False,
                                   window='blackman')
        except:
            print('band=', band)
            print('nyq=', nyq_rate)
            print('drive=', self.drive_freq)

        self.signal = cps.fftconvolve(self.signal, taps, mode='same')

        # Shifts trigger due to causal nature of FIR filter
        self.tidx -= (self.n_taps - 1) / 2

        return

    def amplitude_filter(self):
        '''
        Filters the drive signal out of the amplitude response
        '''
        AMP = cp.fft.fftshift(cp.fft.fft(self.amplitude))

        DRIVE = self.drive_freq / (self.sampling_rate / self.n_points)  # drive location in frequency space
        center = int(len(AMP) / 2)

        # crude boxcar
        AMP[:center - int(DRIVE / 2) + 1] = 0
        AMP[center + int(DRIVE / 2) - 1:] = 0

        self.amplitude = cp.abs(cp.fft.ifft(cp.fft.ifftshift(AMP)))

        return

    def frequency_filter(self):
        '''
        Filters the instantaneous frequency around DC peak to remove noise
        Uses self.filter_bandwidth for the frequency filter
        '''
        FREQ = cp.fft.fftshift(cp.fft.fft(self.inst_freq))

        center = int(len(FREQ) / 2)

        df = self.sampling_rate / self.n_points
        drive_bin = int(cp.ceil(self.drive_freq / df))
        bin_width = int(self.filter_bandwidth / df)

        if bin_width > drive_bin:
            print('width exceeds first resonance')
            bin_width = drive_bin - 1

        FREQ[:center - bin_width] = 0
        FREQ[center + bin_width:] = 0

        self.inst_freq = cp.real(cp.fft.ifft(cp.fft.ifftshift(FREQ)))

        return

    def frequency_harmonic_filter(self, width=5):
        '''
        Filters the instantaneous frequency to remove noise
        Defaults to DC and then every multiple harmonic up to sampling
        
        :param width: Size of the boxcar around the various peaks
        :type width: int, optional
            
        '''
        FREQ = cp.fft.fftshift(cp.fft.fft(self.inst_freq))

        center = int(len(FREQ) / 2)

        # Find drive_bin
        df = self.sampling_rate / self.n_points
        drive_bin = int(cp.ceil(self.drive_freq / df))
        bins = cp.arange(len(FREQ) / 2)[::drive_bin]
        bins = cp.append(center - bins, center + bins)

        FREQ_filt = cp.zeros(len(FREQ), dtype='complex128')
        for b in bins:
            FREQ_filt[int(b) - width:int(b) + width] = FREQ[int(b) - width:int(b) + width]

        self.inst_freq = cp.real(cp.fft.ifft(cp.fft.ifftshift(FREQ)))

        return

    def hilbert(self):
        """
        Analytical signal and calculate phase/frequency via Hilbert transform
        """

        self.hilbert_transform()
        self.calculate_amplitude()
        self.calculate_phase()
        self.calculate_inst_freq()

        return

    def hilbert_transform(self):
        """
        Gets the analytical signal doing a Hilbert transform.
        """

        self.signal = cps.hilbert(self.signal)

        return

    def calculate_amplitude(self):
        """
        Calculates the amplitude of the analytic signal. Uses pre-filter
        signal to do this.
        """
        #
        if self.n_signals != 1:
            signal_orig = self.signal_array.mean(axis=1)
        else:
            signal_orig = self.signal_array

        self.amplitude = cp.abs(cps.hilbert(signal_orig))

        if not cp.isnan(self.AMPINVOLS):
            self.amplitude *= self.AMPINVOLS

        return

    def calculate_power_dissipation(self):
        """
        Calculates the power dissipation using amplitude, phase, and frequency
        and the Cleveland eqn (see DOI:10.1063/1.121434)
        """

        phase = self.phase  # + np.pi/2 #offsets the phase to be pi/2 at resonance

        A = self.k / self.Q * self.amplitude ** 2 * (self.inst_freq + self.drive_freq)
        B = self.Q * self.DriveAmplitude * cp.sin(phase) / self.amplitude
        C = self.inst_freq / self.drive_freq

        self.power_dissipated = A * (B - C)

        return

    def calculate_phase(self, correct_slope=False):
        """
        Gets the phase of the signal and correct the slope by removing
        the drive phase.
        
        :param correct_slope:
        :type correct_slope: bool, optional
        """

        # Unwrap the phase.
        self.phase = cp.unwrap(cp.angle(self.signal))

        if correct_slope:
            start = int(0.3 * self.tidx)
            end = int(0.7 * self.tidx)
            fit = self.phase[start:end]

            xfit = cp.polyfit(cp.arange(start, end), fit, 1)

            # Remove the fit from phase.
            self.phase -= (xfit[0] * cp.arange(self.n_points)) + xfit[1]
        else:
            self.phase -= (2 * cp.pi * self.drive_freq *
                           cp.arange(self.n_points) / self.sampling_rate)

        self.phase = -self.phase  # need to correct for negative in DDHO solution

        self.phase += cp.pi / 2  # corrects to be at resonance pre-trigger

        return

    def calculate_inst_freq(self):
        """
        Calculates the first derivative of the phase using Savitzky-Golay
        filter.
        """

        dtime = 1 / self.sampling_rate  # Time step.

        _inst_freq_raw = sps.savgol_filter(-self.phase.get(), 5, 1, deriv=1,
                                           delta=dtime)

        # Bring trigger to zero.
        self.tidx = int(self.tidx)
        self.inst_freq = cp.array(_inst_freq_raw - _inst_freq_raw[self.tidx])

        return

    def calculate_cwt(self, f_center=None, verbose=False):
        '''
        Calculate instantaneous frequency using continuous wavelet transfer
        
        wavelet specified in self.wavelet. See PyWavelets CWT documentation
        
        :param f_center:
        :type f_center:
        
        :param verbose:
        :type verbose: bool, optional
        
        :param optimize: Currently placeholder for iteratively determining wavelet scales
        :type optimize: bool, optionals
            
        :param fit: Whether to curve-fit for ridge finding or use parabolic approximation
        :type fit: bool, optional
            
        :param calc_phase: Calculates teh Phase (not usually needed)
        :type calc_phase : bool, optional
            
        '''

        # determine if scales will capture the relevant frequency
        if not f_center:
            f_center = self.drive_freq

        dt = 1 / self.sampling_rate
        sc = pywt.scale2frequency(self.wavelet, self.scales) / dt

        if verbose:
            print('Wavelet scale from', numpy.min(sc), 'to', numpy.max(sc))

        if f_center < numpy.min(sc) or f_center > numpy.max(sc):
            raise ValueError('Choose a scale that captures frequency of interest')

        spectrogram, freq = pywt.cwt(self.signal.get(), self.scales, self.wavelet, sampling_period=dt)

        inst_freq, amplitude, _ = parab.ridge_finder(numpy.abs(spectrogram),
                                                         numpy.arange(len(freq)))

        # rescale to correct frequency 
        inst_freq = pywt.scale2frequency(self.wavelet, inst_freq + self.scales[0]) / dt

        tidx = int(self.tidx * len(inst_freq) / self.n_points)

        self.amplitude = amplitude
        self.inst_freq_raw = inst_freq
        self.inst_freq = -1 * (inst_freq - inst_freq[tidx])  # -1 due to way scales are ordered
        self.spectrogram = cp.abs(spectrogram)
        self.wavelet_freq = freq  # the wavelet frequencies

        self.phase = cp.zeros(len(inst_freq))
        self.inst_freq = cp.array(self.inst_freq)
        self.phase = cp.array(self.phase)
        self.amplitude = cp.array(self.amplitude)

        return

    def calculate_stft(self, nfft=200):
        '''
        Sliding FFT approach
        
        :param nfft: Length of FFT calculated in the spectrogram. More points gets much slower
            but the longer the FFT the finer the frequency bin spacing
        :type nfft: int
               
        :param calc_phase: Calculates teh Phase (not usually needed)
        :type calc_phase: bool, optional
            
        '''

        pts_per_ncycle = int(self.fft_time_res * self.sampling_rate)

        if nfft < pts_per_ncycle:
            nfft = pts_per_ncycle

        if pts_per_ncycle > len(self.signal):
            pts_per_ncycle = len(self.signal)

        # drivebin = int(self.drive_freq / (self.sampling_rate / nfft ))
        freq, times, spectrogram = cps.spectrogram(self.signal,
                                                        self.sampling_rate,
                                                        nperseg=pts_per_ncycle,
                                                        noverlap=pts_per_ncycle - 1,
                                                        nfft=nfft,
                                                        window=self.window,
                                                        mode='magnitude')

        # Parabolic ridge finder
        inst_freq, amplitude, _ = parab.cu_ridge_finder(spectrogram, freq)

        # Correctly pad the signals
        _pts = self.n_points - len(inst_freq)
        _pre = int(cp.floor(_pts / 2))
        _post = int(cp.ceil(_pts / 2))

        inst_freq = cp.pad(inst_freq, (_pre, _post))
        amplitude = cp.pad(amplitude, (_pre, _post))

        tidx = int(self.tidx * len(inst_freq) / self.n_points)

        self.amplitude = amplitude
        self.inst_freq_raw = inst_freq
        self.inst_freq = inst_freq - inst_freq[tidx]
        self.spectrogram = spectrogram
        self.stft_freq = freq
        self.stft_times = times
        self.phase = cp.zeros(len(inst_freq))

        return

    def calculate_nfmd(self, calc_phase=False, override_window=True, verbose=False):
        '''
        Nonstationary Fourier Mode Decomposition Approach
        
        :param calc_phase: Calculates the Phase (not usually needed)
        :type calc_phase: bool, optional
            
        :param override_window: Automatically adjusts window to be integer number of cycles 
        :type override_window: bool, optional
            
        :param verbose: Console feedback
        :type verbose: bool, optional
            
    
        '''
        if not self.signal.any():
            self.signal = cp.copy(self.signal_array)
            self.average()

        z = self.signal

        if override_window:

            win_size_cycle = int(self.sampling_rate / self.drive_freq)
            self.window_size = (self.window_size // win_size_cycle) * win_size_cycle

            if verbose:
                print('window size automatically adjusted to ', self.window_size)

        nfmd = CUNFMD(z / cp.std(z),
                      num_freqs=self.num_freqs,
                      window_size=self.window_size,
                      optimizer_opts=self.optimizer_opts,
                      max_iters=self.max_iters,
                      target_loss=self.target_loss,
                      device=self.device)

        if verbose:
            freqs, A, losses, indices = nfmd.decompose_signal(self.update_freq)
        else:
            freqs, A, losses, indices = nfmd.decompose_signal()

        dt = 1 / self.sampling_rate
        self.n_freqs = nfmd.correct_frequencies(dt=dt)
        self.n_amps = nfmd.compute_amps()
        self.n_mean = nfmd.compute_mean()

        self.inst_freq = self.n_freqs[:, 0]
        self.amplitude = self.n_amps[:, 0]

        if calc_phase:
            phase = spg.cumtrapz(self.inst_freq)
            self.phase = cp.append(phase, phase[-1])
        else:
            self.phase = self.n_mean

        return

    def generate_inst_freq(self, dc=True):
        """
        Generates the instantaneous frequency
        
        :param timing: prints the time to execute (for debugging)
        :type timing: bool, optional
            
        :returns: tuple (inst_freq, amplitude, phase)
            WHERE
            array_like inst_freq is instantaneous frequency of the signal. in the format (n_points,)
            [type] amplitude is...
            [type] phase is...
        """

        self.remove_dc()
        self.average()

        if self.method == 'wavelet':

            self.calculate_cwt(**self.wavelet_params)

        elif self.method == 'stft':

            self.calculate_stft(**self.fft_params)

        elif self.method == 'nfmd':

            self.calculate_nfmd()

        elif self.method == 'hilbert':

            self.apply_window()
            self.fir_filter()
            self.hilbert()

            # Filter out oscillatory noise from amplitude
            if self.filter_amplitude:
                self.amplitude_filter()

        else:
            raise ValueError('Invalid analysis method! Valid options: hilbert, wavelet, fft, nfmd')

        # Filter out oscillatory noise from instantaneous frequency
        if self.filter_frequency:
            self.frequency_filter()

        return self.inst_freq, self.amplitude, self.phase

    def analyze(self):
        """
        Analyzes the pixel with the given method.

        :returns: tuple (tfp, shift, inst_freq)
            WHERE
            float tfp is time from trigger to first-peak, in seconds.
            float shift is frequency shift from trigger to first-peak, in Hz.
            array_like inst_freq is instantenous frequency of the signal in format (n_points,)
            
        """

        self.inst_freq, self.amplitude, self.phase = self.generate_inst_freq()

        # If it's a recombination image invert it to find minimum.
        if self.recombination:
            self.inst_freq = self.inst_freq * -1

        # Find where the minimum is.
        self.find_tfp()

        # Restore the length due to FIR filter being causal
        if self.method == 'hilbert':
            self.restore_signal()

        # If it's a recombination image invert it to find minimum.
        if self.recombination:
            self.inst_freq = self.inst_freq * -1
            self.best_fit = self.best_fit * -1
            self.cut = self.cut * -1

        if self.phase_fitting:

            return self.tfp, self.shift, self.phase

        else:

            return self.tfp, self.shift, self.inst_freq
        
    def find_tfp(self):
            """
            Calculate tfp and shift based self.fit_form and self.fit selection
            """
            ridx = int(self.roi * self.sampling_rate)
            cut = self.inst_freq[self.tidx:(self.tidx + ridx)] - self.inst_freq[self.tidx]
            self.cut = cut
            t = cp.arange(cut.shape[0]) / self.sampling_rate

            try:

                tfp_calc.fit_freq_product(self, cut, t)
                
            except:

                self.tfp = cp.nan
                self.shift = cp.nan
                self.best_fit = cp.zeros(cut.shape[0])
                print('error with fitting')

            if not (self.method == 'nfmd' and self.fit_form == 'exp'):
                self.cut += self.inst_freq[self.tidx]
                self.best_fit += self.inst_freq[self.tidx]
            else:
                self.cut += self.phase[self.tidx]
                self.best_fit += self.phase[self.tidx]

            del cut

            return
