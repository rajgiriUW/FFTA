# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:19:11 2022

@author: Raj
"""

import warnings

import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from sidpy.hdf.hdf_utils import get_attributes
from sidpy.viz import plot_utils

from ffta.hdf_utils.process import FFtrEFM


def select(data, scale_tfp=1e6, **kwargs):
    '''
    Chooses a pixel and then displays the instantaneous frequency 
    
    :param data: The dataset to plot
    :type data: USID Dataset or FFtrEFM class object (inherits Process) or the parent Group

    :param scale_tfp: Amount to multiply the tFP values by (1e6 converts to microseconds)
    :type scale_tfp: float
    
    :param kwargs: pyplot arguments
    :type kwargs: dict
    
    :returns: tuple (filt_line, freq_filts, fig_filt, axes_filt)
		WHERE
		numpy.ndarray filt_line is filtered signal of hdf_file
		list freq_filts is the filter parameters to be passed to SignalFilter
	    matplotlib controls fig_filt, axes_filt - Only functional if show_plots is on
    
    '''

    fig, ax = plt.subplots(facecolor='white', figsize=(8, 6))

    if isinstance(data, h5py.Dataset):
        data = data.parent
        warnings.warn('Should specify the parent group, not a dataset')

    if isinstance(data, FFtrEFM):

        img_length = data.parm_dict['FastScanSize']
        img_height = data.parm_dict['SlowScanSize']

        num_cols = data.parm_dict['num_cols']
        num_rows = data.parm_dict['num_rows']

        tfp = data.h5_tfp[()]

    elif isinstance(data, h5py.Group):

        attr = get_attributes(data['Inst_Freq'])
        img_length = attr['FastScanSize']
        img_height = attr['SlowScanSize']

        num_cols = attr['num_cols']
        num_rows = attr['num_rows']

        tfp = data['tfp']

    elif isinstance(data, h5py.Dataset):

        attr = get_attributes(data)
        img_length = attr['FastScanSize']
        img_height = attr['SlowScanSize']

        num_cols = attr['num_cols']
        num_rows = attr['num_rows']

        tfp = data

    tfp = tfp[()]  # numpy array

    kwarg = {'origin': 'lower', 'x_vec': img_length * 1e6,
             'y_vec': img_height * 1e6, 'num_ticks': 5, 'stdevs': 3, 'show_cbar': True}

    ax.set_title('tFP Image')

    for k, v in kwarg.items():
        if k not in kwargs:
            kwargs.update({k: v})

    tfp_image, cbar_tfp = plot_utils.plot_map(ax, tfp * scale_tfp,
                                              cmap='inferno', **kwargs)
    cbar_tfp.set_label('Time (us)', rotation=270, labelpad=16)
    cursor = Cursor(ax, linewidth=2, vertOn=False, horizOn=False)
    plt.show()
