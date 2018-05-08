# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:04:39 2018

@author: Raj
"""

import pycroscopy as px
import numpy as np

from pycroscopy.processing.svd_utils import SVD

from matplotlib import pyplot as plt

from . import hdf_utils

"""
Wrapper to SVD functions, specific to ffta Class

"""

def FF_SVD(h5_main, num_components=128, show_plots=True):
    """
    h5_main : h5Py Dataset
        Main dataset to filter
        
    num_components : int, optional
        Number of SVD components. Increasing this lenghtens computation
        
    show_plots : bool, optional
        If True displays skree, abundance_maps, and data loops
        
    Returns
    -------
    h5_rb_gp : h5Py Group
        Group containing the h5_svd data
    
    """
    h5_svd = SVD(h5_main, num_components=num_components)
    
    parm_dict = hdf_utils.get_params(h5_main)
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    
    # performs SVD
    h5_svd_group = h5_svd.compute()
    
    h5_U = h5_svd_group['U']
    h5_V = h5_svd_group['V']
    h5_S = h5_svd_group['S']
    
    skree_sum = np.zeros(h5_S.shape)

    # abundance maps (eigenvalues) and eigenvectors    
    abun_maps = np.reshape(h5_U[:,:25], (num_rows, num_cols,-1))
    eigen_vecs = h5_V[:9, :]
    h5_spec_vals = px.hdf_utils.getAuxData(h5_main, 'Spectroscopic_Values')[0].value[0,:]
    
    if show_plots:
        for i in range(h5_S.shape[0]):
            skree_sum[i] = np.sum(h5_S[:i])/np.sum(h5_S)
    
        plt.figure()
        plt.plot(skree_sum, 'o')
        print('Need', skree_sum[skree_sum<0.8].shape[0],'components for 80%')
        print('Need', skree_sum[skree_sum<0.9].shape[0],'components for 90%')
        print('Need', skree_sum[skree_sum<0.95].shape[0],'components for 95%')
        print('Need', skree_sum[skree_sum<0.99].shape[0],'components for 99%')
        
        fig_skree, axes =px.plot_utils.plot_scree(h5_S, title='Skree plot')
        
        fig_abun, axes = px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                                                      color_bar_mode='single', cmap='inferno')


        fig_eigvec, axes = px.plot_utils.plot_loops(h5_spec_vals*1e3, eigen_vecs, use_rainbow_plots=False, 
                                                    x_label='Time (ms)', y_label='Displacement (a.u.)', 
                                                    plots_on_side=3, subtitle_prefix='Component', 
                                                    title='SVD Eigenvectors', evenly_spaced=False)
    
    px.hdf_utils.copyAttributes(h5_main.parent, h5_svd_group)
    
    return h5_svd_group

def FF_SVD_filter(h5_main, clean_components=None):
    """
    Filters data given the array clean_components
    
    Clean_components has 2 components will filter from start to finish
    Clean_components has 3+ components will use those individual components
    
    h5_main : h5Py
        Dataset to be filtered and reconstructed.
        This must be the same as where SVD was performed
    
    """
    
    h5_rb = px.svd_utils.rebuild_svd(h5_main, components=clean_components)
    px.hdf_utils.copyAttributes(h5_main.parent, h5_rb)
    
    return h5_rb