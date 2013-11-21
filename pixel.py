#!/usr/bin/env python

"""pixel.py: Contains pixel class."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"

from calc import *


class Pixel:

    def __init__(self, signals):

        self.signals = signals

    def setParams(self, parameters=None):

        if parameters is None:

            self.fs = self.params['SampleRate']
            self.tidx = int(self.params['Trigger'] * self.fs)

        else:

            self.fs = parameters['SampleRate']
            self.tidx = int(parameters['Trigger'] * self.fs)

    def phaseLock(self):

            self.signals, self.tidx = noise.phaselock(self.signals, self.tidx)

    def findNoise(self, k=2):

            self.noisy = noise.discard(self.signals, k)

    def getResult(self):

            self.phaseLock()
            self.findNoise()

            return self.noisy
