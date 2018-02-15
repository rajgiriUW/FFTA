"""loadHDF5.py: Includes routines for loading into HDF5 files."""

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

#import ConfigParser
import sys
from igor.binarywave import load as loadibw
from numpy.lib.npyio import loadtxt
from os.path import splitext

import pycroscopy as py
import h5py

from ffta.utils.load import signal


def loadHDF5():
    tran = px.io.NumpyTranslator()