"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
import logging
from numba import jit
from ffta.utils import noise
from ffta.utils import cwavelet
from ffta.utils import parab
from scipy import signal as sps
from scipy import optimize as spo
from scipy import interpolate as spi


class Pixel(object):
    """Signal Processing to Extract Time-to-First-Peak.

    Extracts time-to-first-peak (tFP) from digitized Fast-Free time-resolved
    Electrostatic Force Microscopy signal. Pixel class uses Morlet Wavelet
    transform to extract tFP.

    Parameters
    ----------
    signal_array : array, [n_points, n_signals]
        2D real-valued signal array, corresponds to a pixel.

    params : dict
        Includes parameters for processing. Here is a list of parameters:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)
        drive_freq = float (in Hz)

        roi = float (in seconds)
        window = string (see documentation of pixel.apply_window)
        bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
        filter_bandwidth = float (in Hz)
        n_taps = integer (default: 999)
        wavelet = bool (exists: wavelet method, otherwise: Hilbert method)

    Attributes
    ----------
    `tidx` : int
        Index of trigger in time-domain.

    `n_points` : int
        Number of points in a signal.

    `n_signals` : int
        Number of signals to be averaged in a pixel.

    `signal_array` : array, shape = [n_points, n_signals]
        Array that contains signals from a single pixel.

    `signal` : array, shape = [n_points]
        The signal after phase-locking and averaging in time domain.

    `phase` : array, shape = [n_points]
        The phase information of the signal.

    `inst_freq` : array, shape = [n_points]
        Instantenous frequency of the signal, depends on the method used.

    `window` : string
        Possible windows are: boxcar, triang, blackman, hamming, hann,
        bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann.

    `tfp` : float
        Time-to-First-Peak, in seconds.

    `shift` : float
        Frequency shift from steady-state to first-peak, in Hz.



    """

    def __init__(self, signal_array, params, fit=False):

        # Create parameter attributes from parameters dictionary.
        for key, value in params.iteritems():

            setattr(self, key, value)

        # Assign values from inputs.
        self.signal_array = signal_array
        self.tidx = int(self.trigger * self.sampling_rate)
        self.n_points, self.n_signals = signal_array.shape
        self.n_points_orig = signal_array.shape[0]
        self.fit = fit

        # Initialize attributes.
        self.signal = None
        self.phase = None
        self.inst_freq = None
        self.tfp = None
        self.shift = None

        # Check if there is n_taps, if not assign default.
        if not hasattr(self, 'n_taps'):

            self.n_taps = 999

        # Check if there is wavelet_analysis, if not assign default.
        if not hasattr(self, 'wavelet_analysis'):

            self.wavelet_analysis = False
        
        return

    def remove_dc(self):
        """Remove DC components from signals."""

        self.signal_array -= self.signal_array.mean(axis=0)

        return

    def phase_lock(self):
        """Phase-lock signals in the signal array. This also cuts signals."""

        # Phase-lock signals.
        self.signal_array, self.tidx = noise.phase_lock(
            self.signal_array, self.tidx,
            np.ceil(self.sampling_rate / self.drive_freq))

        # Update number of points after phase-locking.
        self.n_points = self.signal_array.shape[0]

        return

    def average(self):
        """Average signals."""

        self.signal = self.signal_array.mean(axis=1)

        return

    def check_drive_freq(self):
        """Calculate drive frequency of averaged signals, and check against
           the given drive frequency."""

        n_fft = 2 ** int(np.log2(self.tidx))  # For FFT, power of 2.
        dfreq = self.sampling_rate / n_fft  # Frequency separation.

        # Calculate drive frequency from maximum power of the FFT spectrum.
        signal = self.signal[:n_fft]
        fft_amplitude = np.abs(np.fft.rfft(signal))
        drive_freq = fft_amplitude.argmax() * dfreq

        # Difference between given and calculated drive frequencies.
        difference = np.abs(drive_freq - self.drive_freq)

        # If difference is too big, reassign. Otherwise, continue.
        if difference >= dfreq:

            self.drive_freq = drive_freq

        return

    def apply_window(self):
        """Apply the window given in parameters."""

        self.signal *= sps.get_window(self.window, self.n_points)

        return

    def fir_filter(self):
        """Filters signal with a FIR bandpass filter."""

        # Calculate bandpass region from given parameters.
        nyq_rate = 0.5 * self.sampling_rate
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / nyq_rate
        freq_high = (self.drive_freq + bw_half) / nyq_rate

        band = [freq_low, freq_high]

        # Create taps using window method.
        taps = sps.firwin(self.n_taps, band, pass_zero=False, window='parzen')

        self.signal = sps.fftconvolve(self.signal, taps, mode='same')
        self.tidx -= (self.n_taps - 1) / 2

        return

    def iir_filter(self):
        """Filters signal with two Butterworth filters (one lowpass,
        one highpass) using filtfilt. This method has linear phase and no
        time delay. Do not use for production."""

        # Calculate bandpass region from given parameters.
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / (0.5 * self.sampling_rate)
        freq_high = (self.drive_freq + bw_half) / (0.5 * self.sampling_rate)

        # Do a high-pass filtfilt operation.
        b, a = sps.butter(9, freq_low, btype='high')
        self.signal = sps.filtfilt(b, a, self.signal)

        # Do a low-pass filtfilt operation.
        b, a = sps.butter(9, freq_high, btype='low')
        self.signal = sps.filtfilt(b, a, self.signal)

        return

    def hilbert_transform(self):
        """Get the analytical signal doing a Hilbert transform."""

        self.signal = sps.hilbert(self.signal)

        return

    def calculate_phase(self, correct_slope=True):
        """Get the phase of the signal and correct the slope by removing
        the drive phase."""

        # Unwrap the phase.
        self.phase = np.unwrap(np.angle(self.signal))

        if correct_slope:

            # Remove the drive from phase.
            self.phase -= (2 * np.pi * self.drive_freq *
                           np.arange(self.n_points) / self.sampling_rate)

            # A curve fit on the initial part to make sure that it worked.
            start = int(0.3 * self.tidx)
            end = int(0.7 * self.tidx)
            fit = self.phase[start:end]

            xfit = np.polyfit(np.arange(start, end), fit, 1)

            # Remove the fit from phase.
            self.phase -= (xfit[0] * np.arange(self.n_points)) + xfit[1]

        return

    def calculate_inst_freq(self):
        """Calculates the first derivative of the phase using Savitzky-Golay
        filter."""

        dtime = 1 / self.sampling_rate  # Time step.

        # Do a Savitzky-Golay smoothing derivative.
        self.inst_freq = sps.savgol_filter(self.phase, 5, 1,
                                           deriv=1,
                                           delta=dtime)

        return

    def find_minimum(self):
        """Find when the minimum of instantenous frequency happens."""

        # Cut the signal into region of interest.
        ridx = int(self.roi * self.sampling_rate)
        cut = self.inst_freq[self.tidx:(self.tidx + ridx)]

        # Define a spline to be used in finding minimum and find minimum.
        x = np.arange(ridx)
        y = cut
        func = spi.UnivariateSpline(x, y, ext=3)

        idx = spo.fmin_powell(func, cut.argmin(), disp=0)

        # Do index to time conversion and find shift.
        self.tfp = idx / self.sampling_rate
        self.shift = func(0) - func(idx)

        return

    @staticmethod
    @jit
    def __decay__(t, tau, tau2, dfz):

        decay = dfz * np.expm1(-t / tau)
        relax = -2 * dfz * np.expm1(-t / tau2)

        return (decay + relax)

    def fit_minimum(self):

        ridx = int(self.roi * self.sampling_rate)
        fit_time = np.arange(0, ridx) / self.sampling_rate
        fit_inst_freq = self.inst_freq[self.tidx:(self.tidx + ridx)]

        initial_guess = (100e-6, 5.28e-4, 1000.)

        popt, pcov = spo.curve_fit(self.__decay__, fit_time, fit_inst_freq,
                                   p0=initial_guess)

        tau = popt[0]
        tau2 = popt[1]

        self.tfp = tau * tau2 * np.log(tau / (2 * tau2)) / (tau2 - tau)
        self.shift = self.__decay__(self.tfp, *popt)

        return

    def restore_length(self):
        """Restores the length of instantenous frequency array to original,
        keeping trigger at the center."""

        # Decide how much cut there is going to be from the ends.
        cut = np.min([(self.n_points - self.tidx), self.tidx])
        new = self.inst_freq[(self.tidx - cut):(self.tidx + cut)]  # Cut

        self.tidx = self.n_points_orig / 2
        padding = int(self.tidx - cut)  # How much padding there is.
        self.inst_freq = np.pad(new, (padding, padding), 'edge')

        return

    def __get_cwt__(self):
        """Generates the CWT using Morlet wavelet. Returns a 2D Matrix."""

        w0 = 5  # Default for Morlet
        wavelet_increment = 0.1  # Reducing this has little benefit.\

        cwt_scale = ((w0 + np.sqrt(2 + w0 ** 2)) /
                     (4 * np.pi * self.drive_freq / self.sampling_rate))

        widths = np.arange(cwt_scale * 0.9, cwt_scale * 1.1, wavelet_increment)

        cwt_matrix = cwavelet.cwt(self.signal, dt=1, scales=widths, p=5)
        self.cwt_matrix = np.abs(cwt_matrix)

        return w0, wavelet_increment, cwt_scale

    def calculate_cwt_freq(self):
        """Fits a curve to each column in the CWT matrix to generate the
        frequency."""

        # Generate necessary tools for wavelet transform.
        w0, wavelet_increment, cwt_scale = self.__get_cwt__()

        n_scales, n_points = np.shape(self.cwt_matrix)
        inst_freq = np.empty(n_points)

<<<<<<< HEAD

        x = np.arange(n_scales)

        for i in range(n_points):

            cut = self.cwt_matrix[:, i]
            #inst_freq[i] = -cut.max()
            inst_freq[i], gb = parab.fit(cut, np.argmax(cut) )
            #y = -1 * cut  # Inverting it to use minimization algorithms.

            #func = spi.UnivariateSpline(x, y, ext=3)
            #inst_freq[i] = spo.fmin_powell(func, cut.argmin(), disp=0)
=======
        for i in xrange(n_points):

            cut = self.cwt_matrix[:, i]
            #inst_freq[i] = -cut.max()
            inst_freq[i], gb = parab.fit(cut, np.argmax(cut))
>>>>>>> e23a411b4940de2301b1d981b19ba47b45a13ac8

        inst_freq = (inst_freq * wavelet_increment + 0.9 * cwt_scale)
        inst_freq = ((w0 + np.sqrt(2 + w0 ** 2)) /
                     (4 * np.pi * inst_freq[:] / self.sampling_rate))

        self.inst_freq = inst_freq - inst_freq[self.tidx]

        return

    def analyze(self):
        """Hilbert method: Runs the analysis for the pixel and outputs tFP,
        shift and instantenous frequency."""

        try:
            # Remove DC component, first.
            self.remove_dc()

            # Phase-lock signals.
            self.phase_lock()

            # Average signals.
            self.average()

            # Remove DC component again, introduced by phase-locking.
            self.remove_dc()

            # Check the drive frequency.
            self.check_drive_freq()

            if self.wavelet_analysis:

                # Calculate instantenous frequency using wavelet transform.
                self.calculate_cwt_freq()

            else:

                # Apply window.
                self.apply_window()

                # Filter the signal with a filter, if wanted.
                if self.bandpass_filter == 1:

                    self.fir_filter()

                elif self.bandpass_filter == 2:

                    self.iir_filter()

                # Get the analytical signal doing a Hilbert transform.
                self.hilbert_transform()

                # Calculate the phase from analytic signal.
                self.calculate_phase()

                # Calculate instantenous frequency.
                self.calculate_inst_freq()

            # Find where the minimum is.
            if self.fit:

                self.fit_minimum()

            else:

                self.find_minimum()

            # Restore the length.
            self.restore_length()

        except Exception as e:

            self.tfp = 0
            self.shift = 0
            self.inst_freq = np.zeros(self.n_points_orig)

            logging.exception(e, exc_info=True)

        return (self.tfp, self.shift, self.inst_freq)