# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:05:42 2020

@author: Raj
"""

import os

import matplotlib.font_manager as fm
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# from ffta.hdf_utils import get_utils
from ffta.load import get_utils
from ffta.pixel_utils import badpixels


def plot_tfps(h5_file, h5_path='/', append='', savefig=True, stdevs=2, scale=None):
    """
    Plots the relevant tfp, inst_freq, and shift values as separate image files
    
    :param h5_file:
    :type h5_file: h5Py File
    
    :param h5_path: Location of the relevant datasets to be saved/plotted. e.g. h5_rb.name
    :type h5_path: str, optional
        
    :param append: A string to include in the saved figure filename
    :type append: str, optional
        
    :param savefig: Whether or not to save the image
    :type savefig: bool, optional
        
    :param stdevs: Number of standard deviations to display
    :type stdevs: int, optional
        
    :param scale: Scale bar size, in microns
    :type scale: float, optional
        
    :returns: tuple (fig, ax)
        WHERE
        [type] fig is...
        [type] ax is...
    """

    try:
        h5_if = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Inst_Freq')[0]
    except:
        h5_if = usid.hdf_utils.find_dataset(h5_file[h5_path], 'inst_freq')[0]

    parm_dict = get_utils.get_params(h5_if)
    #  parm_dict = usid.hdf_utils.get_attributes(h5_file[h5_path])

    if 'Dataset' in str(type(h5_file[h5_path])):
        h5_path = h5_file[h5_path].parent.name

    tfp = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp')[0][()]
    shift = usid.hdf_utils.find_dataset(h5_file[h5_path], 'shift')[0][()]

    if tfp.shape[1] == 1:
        # Forgot to reconvert and/or needs to be converted
        [ysz, xsz] = h5_if.pos_dim_sizes
        tfp = np.reshape(tfp, [ysz, xsz])
        shift = np.reshape(shift, [ysz, xsz])

    try:
        tfp_fixed = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp_fixed')[0][()]
    except:
        tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
        tfp_fixed = np.array(tfp_fixed)

    xs = parm_dict['FastScanSize']
    ys = parm_dict['SlowScanSize']
    asp = ys / xs
    if asp != 1:
        asp = asp * 2

    fig, ax = plt.subplots(nrows=3, figsize=(8, 9), facecolor='white')

    [vmint, vmaxt] = np.mean(tfp) - 2 * np.std(tfp), np.mean(tfp) - 2 * np.std(tfp)
    [vmins, vmaxs] = np.mean(shift) - 2 * np.std(shift), np.mean(shift) - 2 * np.std(shift)

    _, cbar_t = usid.viz.plot_utils.plot_map(ax[0], tfp_fixed * 1e6, x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)
    _, cbar_r = usid.viz.plot_utils.plot_map(ax[1], 1 / (1e3 * tfp_fixed), x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)
    _, cbar_s = usid.viz.plot_utils.plot_map(ax[2], shift, x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)

    cbar_t.set_label('tfp (us)', rotation=270, labelpad=16)
    ax[0].set_title('tfp', fontsize=12)

    if scale:
        sz = np.floor(xs * 1e6 / scale)
        ratio = int(tfp_fixed.shape[1] / sz)
        if scale > 1:
            sz = str(scale) + ' $\mu m$'
        else:
            sz = str(int(scale * 1000)) + ' nm'

        fontprops = fm.FontProperties(size=14, weight='bold')
        bar1 = AnchoredSizeBar(ax[0].transData, ratio, sz, frameon=False,
                               loc='lower right', size_vertical=2,
                               pad=0.2, color='white', fontproperties=fontprops)

        ax[0].add_artist(bar1)

    cbar_r.set_label('Rate (kHz)', rotation=270, labelpad=16)
    ax[1].set_title('1/tfp', fontsize=12)

    if scale:
        bar1 = AnchoredSizeBar(ax[0].transData, ratio, sz, frameon=False,
                               loc='lower right', size_vertical=2,
                               pad=0.2, color='white', fontproperties=fontprops)
        ax[1].add_artist(bar1)

    cbar_s.set_label('shift (Hz)', rotation=270, labelpad=16)
    ax[2].set_title('shift', fontsize=12)

    if scale:
        bar1 = AnchoredSizeBar(ax[0].transData, ratio, sz, frameon=False,
                               loc='lower right', size_vertical=2,
                               pad=0.2, color='white', fontproperties=fontprops)
        ax[2].add_artist(bar1)

    fig.tight_layout()

    if savefig:
        path = h5_file.file.filename.replace('\\', '/')
        path = '/'.join(path.split('/')[:-1]) + '/'
        os.chdir(path)
        fig.savefig('tfp_shift_' + append + '_.tif', format='tiff')

    return fig, ax


def get_scale(target_size, x_size):
    """
    
    :param target_size:
    :type target_size:
    
    :param x_size:
    :type x_size:
    
    :returns:
    :rtype:
    
    """
    return
