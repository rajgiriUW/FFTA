# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:15:36 2018

@author: Raj
"""

import pycroscopy as px
from ffta.utils import hdf_utils
import numpy as np

def FFT_filter(hdf_file, parameters={}, DC=True, linenum = 0, show_plots = True, 
               narrowband = False, noise_tolerance = 5e-7):
    """
    Applies FFT Filter to the file and displays the result
    
    hdf_file : h5Py file or Nx1 NumPy array
        hdf_file to work on, e.g. hdf.file['/FF-raw'] if that's a Dataset
        if ndarray, uses passed or default parameters
        Use ndarray.flatten() to ensure correct dimensions
        
    DC : bool, optional
        Determines whether to remove mean (DC-offset)
        
    parameters : dict, optional
        Contains parameters in FF-raw file for constructing filters
        Must contain num_pts and samp_rate to be functional
    
    linenum : int, optional
        For extracting a specific line to do FFT Filtering on
        
    show_plots : bool, optional
        Turns on FFT plots from Pycroscopy
    
    narrowband : bool, optional
        Sets noiseband filter to be a narrow pass filter centered on the drive_frequency
    
    noise_tolerance : float 0 to 1
        Amount of noise below which signal is set to 0
    
    Returns:
    
    filt_line : numpy.ndarray
        Filtered signal of hdf_file
    
    fig_filt, axes_filt: matplotlib controls
        Only functional if show_plots is on
    """
    
    reshape = False
    
    if 'h5py' in str(type(hdf_file)):   #hdf file
        
        parameters = hdf_utils.get_params(hdf_file)
        hdf_file = hdf_utils.get_line(hdf_file, linenum, array_form=True, transpose=False)
        hdf_file = hdf_file.flatten()
    
    if len(hdf_file.shape) == 2:
    
        sh = hdf_file.shape
        reshape = True
        hdf_file = hdf_file.flatten() 
    
    num_pts = hdf_file.shape[0]
    drive = parameters['drive_freq']
    lpf_cutoff = np.round(drive/1e5, decimals=0)*2*1e5     # 2times the drive frequency, round up
    samp_rate = parameters['sampling_rate']

    #default filtering, note the bandwidths --> DC filtering and certain noise peaks
    lpf = px.processing.fft.LowPassFilter(num_pts, samp_rate, lpf_cutoff)
    nbf = px.processing.fft.NoiseBandFilter(num_pts, samp_rate, 
                                            [10E3, 50E3, 100E3, 150E3, 200E3],
                                            [20E3, 1E3, 1E3, 1E3, 1E3])

    freq_filts = [lpf, nbf]
    
    # Remove DC Offset
    if DC == True:
        
        hdf_file -= hdf_file.mean(axis=0)
    
    # Generate narrowband signal
    if narrowband == True:
        
        try:
            bw = parameters['filter_bandwidth']
        except:
            bw = 2500
            
        nbf = px.processing.fft.HarmonicPassFilter(num_pts, samp_rate, drive, bw, 5)
        freq_filts = [nbf]

#    noise_tolerance = 5e-7

    # Test filter on a single line:
    filt_line, fig_filt, axes_filt = px.processing.gmode_utils.test_filter(hdf_file,
                                                                           frequency_filters=freq_filts,
                                                                           noise_threshold=noise_tolerance,
                                                                           show_plots=show_plots)
    
    # If need to reshape
    if reshape == True:
        filt_line = np.reshape(filt_line, sh)
        
    return filt_line, fig_filt, axes_filt