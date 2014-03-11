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


class Pixel(object):

    def __init__(self, parameters):

        # Create parameter attributes from parameters dictionary.
        for key, value in parameters.iteritems():

            setattr(self, key, value)

        self.tidx = self.trigger * self.sampling_rate

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
        signal = self.signal_array.mean(axis=1)[:n]
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

    def cwt(self):

        scale = (5 + np.sqrt(27)) / \
                (self.drive_freq / self.sampling_rate * 4 * np.pi)
        wavelet_inc = 0.5
        widths = np.arange(scale * 0.9, scale * 1.2, wavelet_inc)
        cwt_matrix = sps.cwt(self.signal, sps.morlet, widths)

    def run_cwt(self):

        # Remove DC component, first.
        self.remove_DC()

        # Phase-lock signals.
        self.signal_array, self.tidx = \
            noise.phase_lock(self.signal_array, self.tidx)

        # Remove DC component again, introduced by phase-locking.
        self.remove_DC()

        # Check the drive frequency.
        self.check_drive_freq()

        # Average signals.
        self.signal = self.signal_array.mean(axis=1)
