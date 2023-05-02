"""loadHDF5.py: Includes routines for loading into HDF5 files."""

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import os

import h5py
import numpy as np
import pyUSID as usid
from igor2.binarywave import load as loadibw


def load_exc(ibw_folder='', pixels=64, scale=1, offset=1, verbose=False):
    """
    Loads excitation files into a single .h5 file

    :param ibw_folder: The folder containing the excitation files to load
    :type ibw_folder: string, optional

    :param pixels: How many pixels per line to divide the excitation ibw
    :type pixels: int, optional

    :param scale: The AC signal scaling, assuming the input excitation is wrong
    :type scale: float, optional

    :param offset: The DC offset for the excitation
    :type offset: float, optional

    :param verbose:
    :type verbose: bool, optional

    :returns:
    :rtype: str
    """
    if not any(ibw_folder):
        exc_file_path = usid.io_utils.file_dialog(caption='Select any file in destination folder',
                                                  file_filter='IBW Files (*.ibw)')
        exc_file_path = '/'.join(exc_file_path.split('/')[:-1])

    filelist = os.listdir(exc_file_path)

    data_files = [os.path.join(exc_file_path, name)
                  for name in filelist if name[-3:] == 'ibw']

    h5_path = os.path.join(exc_file_path, 'excitation') + '.h5'

    h5_file = h5py.File(h5_path, 'w')

    _line0 = loadibw(data_files[0])['wave']['wData']
    _pix0 = np.split(_line0, pixels, axis=1)[0].mean(axis=1)
    _pix0 = scale * ((_pix0 - np.min(_pix0)) / (np.max(_line0) - np.min(_line0)) - 0.5) + offset

    data = h5_file.create_dataset('excitation',
                                  shape=(len(data_files), pixels, len(_pix0)),
                                  compression='gzip')

    for n, d in enumerate(data_files):

        if verbose:
            print('### Loading', d.split('/')[-1], '###')

        _line = loadibw(d)['wave']['wData']

        for m, l in enumerate(np.split(_line, pixels, axis=1)):
            _pix = np.mean(l, axis=1)
            data[n, m, :] = _pix

    # self.h5_tfp = self.h5_results_grp.create_dataset('tfp', data=_arr, dtype=np.float32)

    return h5_path
