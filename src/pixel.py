"""pixel.py: Contains pixel class."""
#pylint: disable=E1101,R0902
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
from utils import noise
from scipy import signal as sps
from scipy import optimize as spo


class Pixel(object):
    """Signal Processing to Extract Time-to-First-Peak.

    Extracts time-to-first-peak (tFP) from digitized Fast-Free time-resolved
    Electrostatic Force Microscopy signal. Pixel class uses Hilbert transform
    to extract tFP.

    Parameters
    ----------
    signal_array : array, [n_points, n_signals]
        2D real-valued signal array.

    params : dict
        Includes parameters for processing. Here is a list of parameters:

        trigger = float
        total_time = float
        sampling_rate = int
        drive_freq = float

        window = boolean
        bandpass_filter = boolean
        filter_bandwidth = float

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

    `inst_freq` : array, shape = [n_points]
        Instantenous frequency of the signal, depends on the method used.

    `window_type` : string, (default='blackman')
        Possible windows are: boxcar, triang, blackman, hamming, hann,
        bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann.

    `tfp` : float
        Time-to-First-Peak.

    """
    def __init__(self, signal_array, params):

        # Create parameter attributes from parameters dictionary.
        for key, value in params.iteritems():

            setattr(self, key, value)

        self.signal_array = signal_array
        self.signal = None
        self.phase = None
        self.inst_freq = None
        self.tfp = None
        self.shift = None

        self.tidx = int(self.trigger * self.sampling_rate)
        self.n_points, self.n_signals = signal_array.shape

        if self.window:

            self.window_type = 'blackman'

        else:

            self.window_type = 'boxcar'

        return

    def remove_dc(self):
        """Remove DC components from signals."""

        self.signal_array -= self.signal_array.mean(axis=0)

        return

    def phase_lock(self):
        """Phase-lock signals in the signal array. This also cuts signals."""

        self.signal_array, self.tidx = \
            noise.phase_lock(self.signal_array, self.tidx)

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

            #print "Calculated and given drive frequencies are not matching!"
            #print "New drive frequency: {0:.0f} Hz.".format(drive_freq)

            self.drive_freq = drive_freq

        return

    def apply_window(self):
        """Apply the window defined in parameters."""

        self.signal *= sps.get_window(self.window_type, self.n_points)

        return

    def fir_filter(self, n_taps=1999):
        """
        Filters signal with a FIR bandpass filter.

        Parameters
        ----------
        n_taps : integer, (default=1999)
            Number of taps for FIR filter.

        """

        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / (0.5 * self.sampling_rate)
        freq_high = (self.drive_freq + bw_half) / (0.5 * self.sampling_rate)

        band = [freq_low, freq_high]

        taps = sps.firwin(n_taps, band, pass_zero=False)

        self.signal = sps.fftconvolve(self.signal, taps, mode='same')
        self.tidx = self.tidx - int(n_taps / 2)

        return

    def butterworth_filter(self):
        """
        Filters signal with a Butterworth bandpass filter using lfiltfilt.
        This method has linear phase and no time delay.
        """
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / (0.5 * self.sampling_rate)
        freq_high = (self.drive_freq + bw_half) / (0.5 * self.sampling_rate)

        band = [freq_low, freq_high]
        b, a = sps.butter(5, band, btype='bandpass')

        self.signal = sps.filtfilt(b, a, self.signal)

    def hilbert_transform(self):
        """Get the analytical signal doing a Hilbert transform."""

        self.signal = sps.hilbert(self.signal)

        return

    def calculate_phase(self, correct_slope=True):
        """Get the phase of the signal and correct the slope by removing
        the drive phase."""

        self.phase = np.unwrap(np.angle(self.signal))

        if correct_slope:

            self.phase -= (2 * np.pi * self.drive_freq
                           * np.arange(self.n_points) / self.sampling_rate)

            # A curve fit on the initial part to make sure that it worked.
            start = int(0.2 * self.tidx)
            end = int(0.8 * self.tidx)
            fit = self.phase[start:end]

            xfit = np.polyfit(np.arange(start, end) /
                              self.sampling_rate, fit, 1)
            self.phase -= (xfit[0] * np.arange(self.n_points) /
                           self.sampling_rate) + xfit[1]

            # Drive frequency is corrected through curve fitting.
            self.drive_freq += xfit[0]

        return

    def calculate_inst_freq(self):
        """Calculates the first derivative of the phase using Savitzky-Golay
        filter."""

        delta = 1 / self.sampling_rate
        self.inst_freq = sps.savgol_filter(self.phase, 9, 2,
                                           deriv=1, delta=delta)

        return

    def find_minimum(self):
        """Find when the minimum of instantenous frequency happens."""

        ridx = int(self.roi * self.sampling_rate)
        cut = self.inst_freq[self.tidx:(self.tidx + ridx)]

        func = lambda idx: cut[idx]
        idx = int(spo.fminbound(func, 0, ridx))

        self.tfp = idx / self.sampling_rate
        self.shift = -self.inst_freq[idx + self.tidx]

        return

    def get_tfp(self):
        """Runs the analysis for the pixel and outputs tFP."""

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

        # Filter the signal with an FIR filter, if wanted.
        if self.bandpass_filter:

            self.fir_filter()

        # Apply window.
        self.apply_window()

        # Get the analytical signal doing a Hilbert transform.
        self.hilbert_transform()

        # Calculate the phase from analytic signal.
        self.calculate_phase(correct_slope=True)

        # Calculate instantenous frequency.
        self.calculate_inst_freq()

        # Find where the minimum is.
        self.find_minimum()

        return (self.tfp, self.shift)