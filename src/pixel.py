#!/usr/bin/env python
from __future__ import division

"""pixel.py: Contains pixel class."""

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

    def __init__(self, parameters):

        # Create parameter attributes from parameters dictionary.
        for key, value in parameters.iteritems():

            setattr(self, key, value)

        self.tidx = self.trigger * self.sampling_rate

        if self.window:

            self.set_window('blackman')

        else:

            self.set_window('boxcar')

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

    def remove_DC(self):
        """Remove DC components from signals."""

        self.signal_array -= self.signal_array.mean(axis=0)

        return

    def check_drive_freq(self):
        """Calculate drive frequency of averaged signals, and check against
           the given drive frequency."""

        n = 2 ** int(np.log2(self.tidx))  # For FFT, choose n as a power of 2.
        dfreq = self.sampling_rate / n  # Frequency separation.

        # Calculate drive frequency from maximum power of the FFT spectrum.
        signal = self.signal[:n]
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

    def set_window(self, window_type):

        self.window_type = window_type

        return

    def filter_signal(self, n_taps=1999):

        bw_half = self.bandwidth / 2

        fL = (self.drive_freq - bw_half) / (0.5 * self.sampling_rate)
        fH = (self.drive_freq + bw_half) / (0.5 * self.sampling_rate)

        taps = sps.firwin(n_taps, [fL, fH], pass_zero=False, window='parzen')

        self.signal = sps.fftconvolve(self.signal, taps, mode='same')

        return

    def correct_phase_slope(self):

        self.phase -= (2 * np.pi * self.drive_freq * np.arange(self.n_points) /
                       self.sampling_rate)

        # A curve fit on the initial part to make sure that it worked.
        start = int(0.2 * self.tidx)
        end = int(0.8 * self.tidx)
        fit = self.phase[start:end]

        line = lambda x, m, b: m * x + b

        xfit, xn = spo.curve_fit(line,
                                 np.arange(start, end) / self.sampling_rate,
                                 fit)
        self.phase -= (xfit[0] * np.arange(self.n_points) /
                       self.sampling_rate) + xfit[1]

        # Drive frequency is corrected through curve fitting.
        self.drive_freq += xfit[0]

        return

    def run_hilbert(self):

        # Remove DC component, first.
        self.remove_DC()
        # Phase-lock signals.
        self.signal_array, self.tidx = \
            noise.phase_lock(self.signal_array, self.tidx)
        self.n_points = self.signal_array.shape[0]
        # Average signals.
        self.signal = self.signal_array.mean(axis=1)
        # Remove DC component again, introduced by phase-locking.
        self.remove_DC()
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



