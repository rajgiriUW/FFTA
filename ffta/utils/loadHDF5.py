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
import os

import pycroscopy as py
import h5py

from ffta.utils import load


def loadHDF5(path):
    """
    Loads .ibw folder into an HDF5 format.

    Parameters
    ----------
    path : string
        Path to signal file.

    Returns
    -------
    

    """
    filelist = os.listdir(path)

    data_files = [os.path.join(path, name)
                  for name in filelist if name[-3:] == 'ibw']

    config_file = [os.path.join(path, name)
                   for name in filelist if name[-3:] == 'cfg'][0]
    
    #file_name = os.path.
    
    
    tran = px.io.NumpyTranslator()
    
#    h5_path = os.path.join(folder_path, file_name + '.h5')
#
#
#h5_path = tran.translate(h5_path, raw_data_2d, num_rows, num_cols,
#                         qty_name='Current', data_unit='nA', spec_name='Bias',
#                         spec_unit='V', spec_val=volt_vec, scan_height=100,
#                         scan_width=200, spatial_unit='nm', data_type='STS',
#                         translator_name='ASC', parms_dict=parm_dict)
#    
    