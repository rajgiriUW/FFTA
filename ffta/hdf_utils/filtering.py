# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:15:36 2018

@author: Raj
"""

import pycroscopy as px
from pycroscopy.processing.fft import FrequencyFilter
import pyUSID as usid
import numpy as np

from ffta.hdf_utils import get_utils

from ffta import pixel
from matplotlib import pyplot as plt

import warnings


def FFT_testfilter(hdf_file, parameters={}, DC=True, pixelnum=[0, 0], show_plots=True,
                   narrowband=False, noise_tolerance=5e-7, bandwidth=-1, check_filter=True):
    """
    Applies FFT Filter to the file at a specific line and displays the result
    
    Usage:
    >> h5_ll = hdf_utils.get_line(h5_file, linenum, avg=True)
    >> filt_sig, freq_filts, _,_ = filtering.FFT_testfilter(h5_ll, 
                                                            parameters, 
                                                            narrowband=True, 
                                                            noise_tolerance=1e-5, 
                                                            show_plots=True)
    
    This works on any line. However, you should have the filter work on a single signal
    
    hdf_file : h5Py file or Nx1 NumPy array (preferred is NumPy array)
        hdf_file to work on, e.g. hdf.file['/FF-raw'] if that's a Dataset
        if ndarray, uses passed or default parameters
        Use ndarray.flatten() to ensure correct dimensions
        
    DC : bool, optional
        Determines whether to remove mean (DC-offset)
        
    parameters : dict, optional
        Contains parameters in FF-raw file for constructing filters. Automatic if a Dataset/File
        Must contain num_pts and samp_rate to be functional
    
    pixelnum : int, optional
        For extracting a specific pixel to do FFT Filtering on
        
    show_plots : bool, optional
        Turns on FFT plots from Pycroscopy
    
    narrowband : bool, optional
        Sets noiseband filter to be a narrow pass filter centered on the drive_frequency
        Sometimes fails for very large datasets for unknown reasons.
    
    noise_tolerance : float 0 to 1
        Amount of noise below which signal is set to 0
        
    bandwidth : int, optional
        Total bandwidth (each side is half-bandwidth) is capped at 2500 Hz for computational reasons
        This overrides the parameters value
        
    Returns
    -------
    filt_line : numpy.ndarray
        Filtered signal of hdf_file
    
    freq_filts : list
        The filter parameters to be passed to SignalFilter
    
    fig_filt, axes_filt: matplotlib controls
        Only functional if show_plots is on
    """

    reshape = False
    ftype = str(type(hdf_file))
    if ('h5py' in ftype) or ('Dataset' in ftype):  # hdf file

        parameters = get_utils.get_params(hdf_file)
        # hdf_file = get_utils.get_line(hdf_file, linenum, array_form=True, transpose=False)
        hdf_file = get_utils.get_pixel(hdf_file, [pixelnum[0], pixelnum[1]], array_form=True, transpose=False)
        hdf_file = hdf_file.flatten()

    if len(hdf_file.shape) == 2:
        sh = hdf_file.shape
        reshape = True
        hdf_file = hdf_file.flatten()

    num_pts = hdf_file.shape[0]
    drive = parameters['drive_freq']
    lpf_cutoff = np.round(drive / 1e5, decimals=0) * 2 * 1e5  # 2times the drive frequency, round up
    samp_rate = parameters['sampling_rate']

    # default filtering, note the bandwidths --> DC filtering and certain noise peaks
    lpf = px.processing.fft.LowPassFilter(num_pts, samp_rate, lpf_cutoff)
    nbf = px.processing.fft.NoiseBandFilter(num_pts, samp_rate,
                                            [10E3, 50E3, 100E3, 150E3, 200E3],
                                            [20E3, 1E3, 1E3, 1E3, 1E3])

    freq_filts = [lpf, nbf]

    # Remove DC Offset
    if DC:
        hdf_file -= hdf_file.mean(axis=0)

    # Generate narrowband signal
    if narrowband:

        if bandwidth == -1:  # unspecified
            try:
                bandwidth = parameters['filter_bandwidth']
                if bandwidth > 2500:
                    warnings.warn('Bandwidth of that level might cause errors')
                    bandwidth = 2500
            except:
                print('No bandwidth parameters')
                bandwidth = 2500

        nbf = px.processing.fft.HarmonicPassFilter(num_pts, samp_rate, drive, bandwidth, 7)
        freq_filts = [nbf]

    #    composite_filter = px.fft.build_composite_freq_filter(freq_filts)
    #    print('Composite filter of len:', len(composite_filter))

    # Test filter on a single line:
    filt_line, fig_filt, axes_filt = px.processing.gmode_utils.test_filter(hdf_file,
                                                                           frequency_filters=freq_filts,
                                                                           noise_threshold=noise_tolerance,
                                                                           show_plots=show_plots)

    # If need to reshape
    if reshape:
        filt_line = np.reshape(filt_line, sh)

    # Test filter out in Pixel
    if check_filter:
        plt.figure()
        plt.plot(hdf_file, 'b')
        plt.plot(filt_line, 'k')

        h5_px_filt = pixel.Pixel(filt_line, parameters)
        h5_px_filt.clear_filter_flags()
        h5_px_filt.analyze()
        h5_px_filt.plot(newplot=True, c1='b', c2='k')

        h5_px_raw = pixel.Pixel(hdf_file, parameters)
        h5_px_raw.analyze()
        h5_px_raw.plot(newplot=True, c1='b', c2='k')

    #    h5_px_raw_unfilt = pixel.Pixel(hdf_file, parameters)
    #    h5_px_raw_unfilt.clear_filter_flags()
    #    h5_px_raw_unfilt.analyze()
    #    h5_px_raw_unfilt.plot(newplot=False,c1='y', c2='c')

    return filt_line, freq_filts, fig_filt, axes_filt


def FFT_filter(h5_main, freq_filts, noise_tolerance=5e-7, make_new=False, verbose=False):
    """
    Stub for applying filter above to the entire FF image set
    
    h5_main : h5py.Dataset object
        Dataset to work on, e.g. h5_main = px.hdf_utils.getDataSet(hdf.file, 'FF_raw')[0]
    
    freq_filts : list
        List of frequency filters usually generated in test_line above
        
    noise_tolerance : float, optional
        Level below which data are set to 0. Higher values = more noise (more tolerant)
    
    make_new : bool, optional
        Allows for re-filtering the data by creating a new folder
    
    Returns
    -------
    
    h5_filt : Dataset
        Filtered dataset within latest -FFT_Filtering Group
        
    """

    h5_filt_grp = usid.hdf_utils.check_for_old(h5_main, 'FFT_Filtering')

    if make_new == True or not any(h5_filt_grp):

        sig_filt = px.processing.SignalFilter(h5_main, frequency_filters=freq_filts,
                                              noise_threshold=noise_tolerance,
                                              write_filtered=True, write_condensed=False,
                                              num_pix=1, verbose=verbose, cores=2, max_mem_mb=512)

        h5_filt_grp = sig_filt.compute()

    else:
        print('Taking previously computed results')
        h5_filt = h5_filt_grp[0]['Filtered_Data']

    h5_filt = h5_filt_grp['Filtered_Data']
    usid.hdf_utils.copy_attributes(h5_main.parent, h5_filt)
    usid.hdf_utils.copy_attributes(h5_main.parent, h5_filt.parent)

    return h5_filt


class BandPassFilter(FrequencyFilter):
    def __init__(self, signal_length, samp_rate, f_center, f_width, roll_off=0.05, fir=False):
        """
        Builds a low pass filter

        Parameters
        ----------
        signal_length : unsigned int
            Points in the FFT. Assuming Signal in frequency space (ie - after FFT shifting)
        samp_rate : unsigned integer
            Sampling rate
        f_cutoff : unsigned integer
            Cutoff frequency for filter
        roll_off : 0 < float < 1
            Frequency band over which the filter rolls off. rol off = 0.05 on a
            100 kHz low pass filter -> roll off from 95 kHz (1) to 100 kHz (0)

        Returns
        -------
        BPF : 1D numpy array describing the low pass filter

        """

        if f_center >= 0.5 * samp_rate:
            raise ValueError('Filter cutoff exceeds Nyquist rate')

        self.f_center = f_center
        self.f_width = f_width

        super(BandPassFilter, self).__init__(signal_length, samp_rate)

        cent = int(round(0.5 * signal_length))

        # very simple boxcar
        ind = int(round(signal_length * (f_center / samp_rate)))
        sz = int(round(cent * f_width / samp_rate))

        bpf = np.zeros(signal_length, dtype=np.float32)

        # FIR filter or not; note that FIR filters are causal in nature
        if not fir:
            bpf[cent - ind - sz:cent - ind + sz + 1] = 1
            bpf[cent + ind - sz:cent + ind + sz + 1] = 1
        else:

            freq_low = (f_center - f_width) / nyq_rate
            freq_high = (f_center + f_width) / nyq_rate

            band = [freq_low, freq_high]

            taps = sps.firwin(int(1999), band, pass_zero=False,
                                  window='blackman')
            bpf = np.abs(np.fft.fftshift(np.fft.fft(taps, n=signal_length)))

        self.value = bpf

    def get_parms(self):
        basic_parms = super(BandPassFilter, self).get_parms()
        prefix = 'band_pass_'
        this_parms = {prefix+'start_freq': self.f_center, prefix+'band_width': self.f_width}
        this_parms.update(basic_parms)
        return this_parms
