# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:16:05 2018

@author: Raj
"""

import os
import sys
import time
import multiprocessing
import logging
import argparse as ap
import numpy as np
import ffta.line as line
from ffta.utils import load
import badpixels

# Plotting imports
import matplotlib as mpl
#mpl.use('WxAgg')
from matplotlib import pyplot as plt

from ffta.utils import hdf_utils
import pycroscopy as px
from pycroscopy.io.io_utils import getTimeStamp

"""
Analyzes an HDF_5 format trEFM data set and writes the result into that file
"""

def process(h5_path):
    """Main function of the executable file."""
#    logging.basicConfig(filename='error.log', level=logging.INFO)

    # Initialize file and read parameters
#    h5_path = args.h5_path
    hdf = px.ioHDF5(h5_path)
    h5_main = hdf.file
    p = px.hdf_utils.findH5group(h5_main, 'FF')[0]
    parameters = hdf_utils.get_params(p)

    n_pixels = parameters['num_cols']
    num_rows = parameters['num_rows']
    pnts_per_pixel = parameters['pnts_per_pixel']

    print('Recombination: ', parameters['recombination'])
    print( 'ROI: ', parameters['roi'])

    # Initialize arrays.
    tfp = np.zeros([num_rows, n_pixels])
    shift = np.zeros([num_rows, n_pixels])

    # Initialize plotting.
    plt.ion()

    fig, a = plt.subplots(nrows=2, figsize=(13, 6))

#    fig = plt.figure(figsize=(12, 6), tight_layout=True)
#    grid = gs.GridSpec(1, 2)
    tfp_ax = a[0]
    shift_ax = a[1]

    tfp_ax.set_title('tFP Image')
    shift_ax.set_title('Shift Image')

    img_length = parameters['width']
    img_height = parameters['height']

    kwargs = {'origin': 'lower', 'aspect': 'equal', 'x_size':img_length,
              'y_size':img_height, 'num_ticks': 5, 'stdevs': 3}

    tfp_image, cbar = px.plot_utils.plot_map(tfp_ax, tfp * 1e6, cmap='inferno', **kwargs)
    cbar.set_label('Time (us)', rotation=270, labelpad=16)
    shift_image, cbar = px.plot_utils.plot_map(shift_ax, shift, cmap='inferno', **kwargs)
    cbar.set_label('Frequency Shift (Hz)', rotation=270, labelpad=16)
    text = tfp_ax.text(n_pixels/2,num_rows+3, '')
    
    plt.show()

    # Load every file in the file list one by one.
    for i in range(num_rows):

        line_inst = hdf_utils.get_line(hdf.file, i)
        
        tfp[i, :], shift[i, :], _ = line_inst.analyze()

        tfp_image, _ = px.plot_utils.plot_map(tfp_ax, tfp * 1e6, cmap='inferno', show_cbar=False, **kwargs)
        shift_image, _ = px.plot_utils.plot_map(shift_ax, shift, cmap='inferno', show_cbar=False, **kwargs)

        tfp_sc = tfp[tfp.nonzero()] * 1e6
        tfp_image.set_clim(vmin=tfp_sc.min(), vmax=tfp_sc.max())

        shift_sc = shift[shift.nonzero()]
        shift_image.set_clim(vmin=shift_sc.min(), vmax=shift_sc.max())

        tfpmean = 1e6 * tfp[i, :].mean()
        tfpstd = 1e6 * tfp[i, :].std()

        string = ("Line {0:.0f}, average tFP (us) ="
                  " {1:.2f} +/- {2:.2f}".format(i + 1, tfpmean, tfpstd))
        print(string)

        text.remove()
        text = tfp_ax.text((n_pixels-len(string))/2,num_rows+4, string)

        plt.draw()
        plt.pause(0.0001)

        del line_inst  # Delete the instance to open up memory.

    # Filter bad pixels
    tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
    tfp_fixed = np.array(tfp_fixed)

    # Save to HDF5 file. 
    
    # create unique suffix
    names = hdf_utils.h5_list(hdf.file['/FFtrEFM_Group'], 'processed')
    suffix = names[-1][-4:]
    suffix = str(int(suffix)+1).zfill(4)
    
    grp_name = p.name.split('/')[-1] + '-processed-' + suffix
    grp_tr = px.MicroDataGroup(grp_name, p.name + '/')

    tfp_px = px.MicroDataset('tfp', tfp, parent=p.name)
    shift_px = px.MicroDataset('shift', shift, parent=p.name)
    tfp_fixed_px = px.MicroDataset('tfp_filtered', tfp_fixed, parent=p.name)
    grp_tr.attrs['timestamp'] = getTimeStamp()

    grp_tr.addChildren([tfp_px])
    grp_tr.addChildren([shift_px])
    grp_tr.addChildren([tfp_fixed_px])
    
    hdf.writeData(grp_tr, print_log=True)

    hdf.flush()
    hdf.close()

    return

#
#if __name__ == '__main__':
#
#    sys.exit(main(sys.argv[1:]))
