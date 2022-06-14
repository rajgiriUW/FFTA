# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:04:39 2018

@author: Raj
"""

import h5py
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt
from sidpy.viz import plot_utils

from ffta.load import get_utils
from .svd_utils import SVD, rebuild_svd

"""
Wrapper to SVD functions, specific to ffta Class.

Typical usage:
    >> h5_svd = svd.test_svd(h5_avg)
    >> clean = [0,1,2,3] # to filter to only first 4 components
    >> h5_rb = svd.svd_filter(h5_avg, clean_components=clean)

"""


def test_svd(h5_main, num_components=128, show_plots=True, override=True, verbose=True):
    """

    :param h5_main: Main dataset to filter
    :type h5_main: h5Py Dataset
        
    :param num_components: Number of SVD components. Increasing this lenghtens computation
    :type num_components: int, optional
        
    :param show_plots: If True displays skree, abundance_maps, and data loops
    :type show_plots: bool, optional
        
    :param override: Force SVD.Compute to reprocess data no matter what   
    :type override: bool, optional
    
    :param verbose: Print out component ratio values
    :type verbose: bool, optional
        
    :returns: Group containing the h5_svd data
    :rtype: h5Py Group
        
    """

    if not (isinstance(h5_main, usid.USIDataset)):
        h5_main = usid.USIDataset(h5_main)

    h5_svd = SVD(h5_main, num_components=num_components)

    [num_rows, num_cols] = h5_main.pos_dim_sizes

    # performs SVD
    h5_svd_group = h5_svd.compute(override=override)

    h5_S = h5_svd_group['S']

    if verbose:
        skree_sum = np.zeros(h5_S.shape)
        for i in range(h5_S.shape[0]):
            skree_sum[i] = np.sum(h5_S[:i]) / np.sum(h5_S)

        print('Need', skree_sum[skree_sum < 0.8].shape[0], 'components for 80%')
        print('Need', skree_sum[skree_sum < 0.9].shape[0], 'components for 90%')
        print('Need', skree_sum[skree_sum < 0.95].shape[0], 'components for 95%')
        print('Need', skree_sum[skree_sum < 0.99].shape[0], 'components for 99%')

    if show_plots:
        plot_svd(h5_svd_group)

    return h5_svd_group


def svd_filter(h5_main, clean_components=None):
    """
    Filters data given the array clean_components
    
    :param h5_main: Dataset to be filtered and reconstructed.
        This must be the same as where SVD was performed
    :type h5_main: h5py
    
    :param clean_components:
        Clean_components has 2 components will filter from start to finish
        Clean_components has 3+ components will use those individual components
    :type clean_components:
    
    :returns:
    :rtype:
    
    """
    if not (isinstance(h5_main, usid.USIDataset)):
        h5_main = usid.USIDataset(h5_main)

    h5_rb = rebuild_svd(h5_main, components=clean_components)

    parameters = get_utils.get_params(h5_main)

    for key in parameters:
        if key not in h5_rb.attrs:
            h5_rb.attrs[key] = parameters[key]

    return h5_rb


def plot_svd(h5_main, savefig=False, num_plots=16, **kwargs):
    '''
    Replots the SVD showing the skree, abundance maps, and eigenvectors.
    If h5_main is the results group, then it will plot the values for that group.
    If h5_main is a Dataset, it will default to the most recent SVD group.
    
    :param h5_main:
    :type h5_main: USIDataset or h5py Dataset or h5py Group
    
    :param savefig: Saves the figures to disk
    :type savefig: bool, optional
        
    :param num_plots: Default number of eigenvectors and abundance plots to show
    :type num_plots: int
        
    :param kwargs: keyword arguments for svd filtering
    :type kwargs: dict, optional
        
    '''

    if isinstance(h5_main, h5py.Group):

        _U = usid.hdf_utils.find_dataset(h5_main, 'U')[-1]
        _V = usid.hdf_utils.find_dataset(h5_main, 'V')[-1]
        units = 'arbitrary (a.u.)'
        h5_spec_vals = np.arange(_V.shape[1])
        h5_svd_group = _U.parent

    else:

        h5_svd_group = usid.hdf_utils.find_results_groups(h5_main, 'SVD')[-1]
        units = h5_main.attrs['quantity']
        h5_spec_vals = h5_main.get_spec_values('Time')

    h5_U = h5_svd_group['U']
    h5_V = h5_svd_group['V']
    h5_S = h5_svd_group['S']

    _U = usid.USIDataset(h5_U)
    [num_rows, num_cols] = _U.pos_dim_sizes

    abun_maps = np.reshape(h5_U[:, :16], (num_rows, num_cols, -1))
    eigen_vecs = h5_V[:16, :]

    skree_sum = np.zeros(h5_S.shape)
    for i in range(h5_S.shape[0]):
        skree_sum[i] = np.sum(h5_S[:i]) / np.sum(h5_S)

    plt.figure()
    plt.plot(skree_sum, 'bo')
    plt.title('Cumulative Variance')
    plt.xlabel('Total Components')
    plt.ylabel('Total variance ratio (a.u.)')

    if savefig:
        plt.savefig('Cumulative_variance_plot.png')

    try:
        fig_skree, axes = usid.viz.plot_utils.plot_scree(h5_S, title='Scree plot')
        fig_skree.tight_layout()
    except:
        pass

    if savefig:
        plt.savefig('Scree_plot.png')

    fig_abun, axes = plot_utils.plot_map_stack(abun_maps, num_comps=num_plots, title='SVD Abundance Maps',
                                               color_bar_mode='single', cmap='inferno', reverse_dims=True,
                                               fig_mult=(3.5, 3.5), facecolor='white', **kwargs)
    fig_abun.tight_layout()
    if savefig:
        plt.savefig('Abundance_maps.png')

    fig_eigvec, axes = plot_utils.plot_curves(h5_spec_vals * 1e3, eigen_vecs, use_rainbow_plots=False,
                                              x_label='Time (ms)', y_label=units,
                                              num_plots=num_plots, subtitle_prefix='Component',
                                              title='SVD Eigenvectors', evenly_spaced=False, **kwargs)
    fig_eigvec.tight_layout()
    if savefig:
        plt.savefig('Eigenvectors.png')

    return
