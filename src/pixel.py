"""pixel.py: Contains pixel class."""
#pylint: disable=E1101,R0902
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
from utils import cwavelet, noise, utils
from scipy import signal as sps


class Pixel(object):
    """Digital Signal Process to Extract Time-to-First-Peak.

    Extracts time-to-first-peak (tFP) from digitized Fast-Free time-resolved
    Electrostatic Force Microscopy signal. Pixel class contains three different
    methods to extract tFP: Hilbert-Huang transform, Continuous Wavelet
    Transform and Synthetic Lock-in Amplifier.

    Parameters
    ----------
    params : dict
        Includes parameters for processing. Here is a list of parameters:

        trigger = float
        total_time = float
        sampling_rate = int
        drive_freq = float
        bandwidth = float
        smooth_time = float
        window_size = float
        noise_reduction = boolean
        bandpass_filter = boolean
        window = boolean
        smooth = boolean

    Attributes
    ----------
    `tidx` : int
        Index of trigger in time-domain.

    `n_points` : int
        Number of points in time-domain.

    `n_signals` : int
        Number of signals to be averaged in a pixel.

    `signal_array` : array, shape = [n_points, n_signals]
        Array that contains signals from a single pixel.

    `signal` : array, shape = [n_points]
        The signal after phase-locking and averaging in time domain.

    `phase` : array, shape = [n_points]
        The phase information of the signal.

    `cwt_matrix` : array, shape = [n_scales, n_points]
        Continuous wavelet transform of the signal by Morlet wavelet.

    `inst_freq` : array, shape = [n_points]
        Instantenous frequency of the signal, depends on the method used.

    `windows_type` : string, (default='blackman')
        Possible windows are: boxcar, triang, blackman, hamming, hann,
        bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann.

    """
    def __init__(self, params):

        # Create parameter attributes from parameters dictionary.
        for key, value in params.iteritems():

            setattr(self, key, value)

        self.signal_array = None
        self.signal = None
        self.phase = None
        self.cwt_matrix = None
        self.inst_freq = None

        self.tidx = self.trigger * self.sampling_rate
        self.n_points = None
        self.n_signals = None

        if self.window:

            self.window_type = 'blackman'

        else:

            self.window_type = 'boxcar'

        return

    def load_signals(self, signal_array):
        """
        Load signals from given signal array.

        Parameters
        ----------
        signal_array : array, [n_points, n_signals]
            2D real-valued signal array.

        """

        self.signal_array = signal_array
        self.n_points, self.n_signals = signal_array.shape

        return

    def remove_dc(self):
        """Remove DC components from signals."""

        self.signal_array -= self.signal_array.mean(axis=0)

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

            print "Calculated and given drive frequencies are not matching!"
            print "Calculated drive frequency will be used for calculations."
            print "Calculated drive frequency: {0:.0f} Hz.".format(drive_freq)

            self.drive_freq = drive_freq

            return

        else:

            return

    def filter_signal(self, n_taps=1999):
        """
        Filters signal with a FIR filter.

        Parameters
        ----------
        n_taps : integer, (default=1999)
            Number of taps for FIR filter.

        """

        bw_half = self.bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / (0.5 * self.sampling_rate)
        freq_high = (self.drive_freq + bw_half) / (0.5 * self.sampling_rate)

        bandpass = [freq_low, freq_high]

        taps = sps.firwin(n_taps, bandpass, pass_zero=False, window='parzen')

        self.signal = sps.fftconvolve(self.signal, taps, mode='same')

        return

    def correct_phase_slope(self):
        """Correct the slope of the phase by removing the drive phase."""

        self.phase -= (2 * np.pi * self.drive_freq * np.arange(self.n_points) /
                       self.sampling_rate)

        # A curve fit on the initial part to make sure that it worked.
        start = int(0.2 * self.tidx)
        end = int(0.8 * self.tidx)
        fit = self.phase[start:end]

        xfit = np.polyfit(np.arange(start, end) / self.sampling_rate, fit, 1)
        self.phase -= (xfit[0] * np.arange(self.n_points) /
                       self.sampling_rate) + xfit[1]

        # Drive frequency is corrected through curve fitting.
        self.drive_freq += xfit[0]

        return

    def run_hilbert(self):

        # Remove DC component, first.
        self.remove_dc()
        # Phase-lock signals.
        self.signal_array, self.tidx = \
            noise.phase_lock(self.signal_array, self.tidx)
        self.n_points = self.signal_array.shape[0]
        # Average signals.
        self.signal = self.signal_array.mean(axis=1)
        # Remove DC component again, introduced by phase-locking.
        self.remove_dc()
        # Check the drive frequency.
        self.check_drive_freq()
        # Apply window.
        self.signal *= sps.get_window(self.window_type, self.n_points)
        # Filter the signal with an FIR filter, if wanted.
        if self.bandpass_filter:

            self.filter_signal()

        # Get the analytical signal doing a Hilbert transform.
        self.signal = sps.hilbert(self.signal)
        # Calculate the phase from analytic signal.
        self.phase = np.unwrap(np.angle(self.signal))
        # Correct the slope of the phase.
        self.correct_phase_slope()

        if self.smooth:

            smooth_length = int(self.smooth_time * self.sampling_rate)
            boxcar = np.ones(smooth_length) / smooth_length
            self.phase = sps.fftconvolve(self.phase, boxcar, mode='same')

        self.inst_freq = np.gradient(self.phase, 1 / self.sampling_rate)

    def run_cwt(self):

        # Remove DC component, first.
        self.remove_dc()
        # Phase-lock signals.
        self.signal_array, self.tidx = \
            noise.phase_lock(self.signal_array, self.tidx)
        self.n_points = self.signal_array.shape[0]
        # Average signals.
        self.signal = self.signal_array.mean(axis=1)
        # Remove DC component again, introduced by phase-locking.
        self.remove_dc()
        # Check the drive frequency.
        self.check_drive_freq()

        # Calculate scale and width, then apply CWT.
        scale = (5 + np.sqrt(27)) /\
                (4 * np.pi * self.drive_freq / self.sampling_rate)
        scales = np.arange(scale * 0.8, scale * 1.2, 0.5)

        self.cwt_matrix = cwavelet.cwt(self.signal, dt=1, scales=scales, w0=5)
        freq = utils.max_2d_fit(np.abs(self.cwt_matrix))

        self.inst_freq = -1*freq[:] + freq[0]
