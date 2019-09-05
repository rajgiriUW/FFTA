# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:10:38 2019

@author: Raj
"""

import pyUSID as usid
from pyUSID.io.write_utils import  Dimension

from igor import binarywave as bw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal as sg


def transfer_function(h5_file, tf_file = '', params_file = '', 
                      psd_freq=1e6, offset = 0.0016, sample_freq = 10e6,
                      plot=False):
    '''
    Reads in the transfer function .ibw, then creates two datasets within
    a parent folder 'Transfer_Function'
    
    This will destructively overwrite an existing Transfer Function in there
    
    1) TF (transfer function)
    2) Freq (frequency axis for computing Fourier Transforms)
    
    tf_file : ibw
        Transfer Function .ibw File
        
    params_file : string
        The filepath in string format for the parameters file containing
            Q, AMPINVOLS, etc.
    
    psd_freq : float
        The maximum range of the Power Spectral Density.
        For Asylum Thermal Tunes, this is often 1 MHz on MFPs and 2 MHz on Cyphers
        
    offset : float
        To avoid divide-by-zero effects since we will divide by the transfer function
            when generating GKPFM data
            
    sample_freq : float
        The desired output sampling. This should match your data.   
            
    Returns:
        h5_file['Transfer_Function'] : the Transfer Function group
    '''
    if not any(tf_file):
        tf_file = usid.io_utils.file_dialog(caption='Select Transfer Function file ',
                                            file_filter='IBW Files (*.ibw)')
    data = bw.load(tf_file)
    tf = data.get('wave').get('wData')
    
    if 'Transfer_Function' in h5_file:
        del h5_file['/Transfer_Function']
    h5_file.create_group('Transfer_Function')
    h5_file['Transfer_Function'].create_dataset('TF', data = tf)
    
    freq = np.linspace(0, psd_freq, len(tf))
    h5_file['Transfer_Function'].create_dataset('Freq', data = freq)
    
    parms = params_list(params_file, psd_freq=psd_freq)
    
    for k in parms:
        h5_file['Transfer_Function'].attrs[k] = float(parms[k])

    tfnorm = float(parms['Q']) * (tf - np.min(tf))/ (np.max(tf) - np.min(tf)) 
    tfnorm += offset
    h5_file['Transfer_Function'].create_dataset('TFnorm', data = tfnorm)
    
    TFN_RS, FQ_RS = resample_tf(h5_file, psd_freq=psd_freq, sample_freq=sample_freq)
    TFN_RS = float(parms['Q']) * (TFN_RS - np.min(TFN_RS))/ (np.max(TFN_RS) - np.min(TFN_RS))
    TFN_RS += offset
    
    h5_file['Transfer_Function'].create_dataset('TFnorm_resampled', data = TFN_RS)
    h5_file['Transfer_Function'].create_dataset('Freq_resampled', data = FQ_RS)
    
    if plot:
        plt.figure()
        plt.plot(freq, tfnorm, 'b')
        plt.plot(FQ_RS, TFN_RS, 'r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (m)')
        plt.yscale('log')
        plt.title('Transfer Function')
    
    return h5_file['Transfer_Function']

def resample_tf(h5_file, psd_freq = 1e6, sample_freq = 10e6):
    '''
    Resamples the Transfer Function based on the desired target frequency
    
    This is important for dividing the transfer function elements together
    
    psd_freq : float
        The maximum range of the Power Spectral Density.
        For Asylum Thermal Tunes, this is often 1 MHz on MFPs and 2 MHz on Cyphers
        
    sample_freq : float
        The desired output sampling. This should match your data.   
    
    '''
    TFN = h5_file['Transfer_Function/TFnorm'][()]
    #FQ = h5_file['Transfer_Function/Freq'][()]
    
    # Generate the iFFT from the thermal tune data
    tfn = np.fft.ifft(TFN)
    #tq = np.linspace(0, 1/np.abs(FQ[1] - FQ[0]), len(tfn))
    
    # Resample
    scale = int(sample_freq / psd_freq)
    tfn_rs = sg.resample(tfn, len(tfn)*scale)  # from 1 MHz to 10 MHz
    TFN_RS = np.fft.fft(tfn_rs)
    FQ_RS = np.linspace(0, sample_freq, len(tfn_rs))
    
    return TFN_RS, FQ_RS

def params_list(path = '', psd_freq=1e6, lift=50):
    '''
    Reads in a Parameters file as saved in Igor as a dictionary
    
    For use in creating attributes of transfer Function
    
    '''
    if not any(path):
        path = usid.io.io_utils.file_dialog(caption='Select Parameters Files ',
                                            file_filter='Text (*.txt)')
    
    df = pd.read_csv(path, sep='\t', header=1)
    df = df.set_index(df['Unnamed: 0'])
    df = df.drop(columns='Unnamed: 0')
    
    parm_dict = df.to_dict()['Initial']
    parm_dict['PSDFreq'] = psd_freq
    parm_dict['Lift'] = lift
    
    return parm_dict

def Y_calc(h5_main, tf=None, resampled=True, verbose=True):
    '''
    Divides the response by the transfer function
    
    This processes every pixel, so it takes awhile
    
    h5_main : h5py dataset of USIDataset
    
    tf : transfer function, optional
        This can be the resampled or normal transfer function
        For best results, use the "normalized" transfer function
        "None" will default to /Transfer_Function folder
        
    resampled : bool, optional
        Whether to use the upsampled Transfer Function or the original
        
    verbose: bool, optional
        Gives user feedback during processing
    
    '''
    Yout = np.zeros(h5_main.shape)  # frequency domain
    yout = np.zeros(h5_main.shape)          # time domain
    parm_dict = usid.hdf_utils.get_attributes(h5_main)
    ds = h5_main[()]
    
    rows = parm_dict['num_rows']
    cols = parm_dict['num_cols']
    
    # Create frequency axis for the pixel
    samp = parm_dict['sampling_rate']
    fq_y = np.linspace(0, samp,  Yout.shape[1]) 
    
    # Get the transfer function and transfer function frequency values
    fq_tf = h5_main.file['Transfer_Function/Freq']
    if not tf:
        if resampled:
    
            tf = h5_main.file['Transfer_Function/TFnorm_resampled'][()]
            fq_tf = h5_main.file['Transfer_Function/Freq_resampled'][()]
        
        else:
            
            tf = h5_main.file['Transfer_Function/TFnorm'][()]
    
    # FFT of that pixel
    # For each frequency in the FFT
    #   a) Find frequency in the transfer function
    #   b) Divide the results at that frequency
    #   c) move to next frequency
    # FFT division is Yout
    # iFFT of the results, that is yout
    import time
    t0 = time.time()

    for c in np.arange(h5_main.shape[0]):
        
        if verbose:
            if c%100 == 0:
                print('Pixel:', c)
        
        Ypxl = np.fft.fft(ds[c,:])
        
        for f, x in zip(fq_y, np.arange(int(len(fq_y)/2))):
        
            xx = np.searchsorted(fq_tf, f)
            Yout[c,x] = Ypxl[x] / tf[xx] 
            Yout[c,-x] = Ypxl[x] / tf[xx] 
        
        yout[c,:] = np.fft.ifft(Yout[c,:])
    
    t1 = time.time()
    
    print('Time for pixels:', t1 - t0)

    '''    
    # 9/4: The below line-based method is only ~10 seconds faster for the entire image. 
    # 204 s vs 214 s.

    # This is for comparing line-processing speed
    Yout2 = np.zeros([rows, h5_main.shape[1]*cols])  # frequency domain
    yout2 = np.zeros([rows, h5_main.shape[1]*cols])          # time domain
    
    t0 = time.time()
    
    for c in np.arange(parm_dict['num_rows']):
        if verbose:
            if c%10 == 0:
                print('Pixel:', c)
        ll = get_utils.get_line(h5_main, c, array_form=True)
        ll = np.ravel(ll)
        LL = np.fft.fft(ll)
        fq_y = np.linspace(0, samp,  len(LL))
        
        for f, x in zip(fq_y, np.arange(int(len(fq_y)/2))):
        
            xx = np.searchsorted(fq_tf, f)
            Yout2[c,x] = LL[x] / tf[xx] 
            Yout2[c,-x] = LL[x] / tf[xx] 
        
        yout2[c,:] = np.fft.ifft(Yout2[c,:])
    
    Yout = np.vstack(np.split(Yout2.ravel(), rows*cols, axis=0))
    yout = np.vstack(np.split(yout2.ravel(), rows*cols, axis=0))         # time domain
    
    t1 = time.time()
    
    print('Time for pixels:', t1 - t0)
    '''
    return Yout, yout

def save_Yout(h5_main, Yout, yout):

    parm_dict = usid.hdf_utils.get_attributes(h5_main)
    
    # Get relevant parameters
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_avg = parm_dict['pnts_per_avg']

    h5_gp = h5_main.parent
    h5_meas_group = usid.hdf_utils.create_indexed_group(h5_gp, 'GKPFM_Frequency')

    # Create dimensions
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

    #ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
    spec_desc = [Dimension('Frequency', 'Hz',np.linspace(0, parm_dict['sampling_rate'], pnts_per_avg))]
    #ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)

    # Writes main dataset
    h5_y = usid.hdf_utils.write_main_dataset(h5_meas_group,  
                                             Yout,
                                             'Y',  # Name of main dataset
                                             'Deflection',  # Physical quantity contained in Main dataset
                                             'V',  # Units for the physical quantity
                                             pos_desc,  # Position dimensions
                                             spec_desc,  # Spectroscopic dimensions
                                             dtype=np.float32,  # data type / precision
                                             main_dset_attrs=parm_dict)

    usid.hdf_utils.copy_attributes(h5_y, h5_gp)
    
    h5_meas_group = usid.hdf_utils.create_indexed_group(h5_gp, 'GKPFM_Time')
    spec_desc = [Dimension('Time', 's',np.linspace(0, parm_dict['total_time'], pnts_per_avg))]
    h5_y = usid.hdf_utils.write_main_dataset(h5_meas_group,  
                                             yout,
                                             'y_time',  # Name of main dataset
                                             'Deflection',  # Physical quantity contained in Main dataset
                                             'V',  # Units for the physical quantity
                                             pos_desc,  # Position dimensions
                                             spec_desc,  # Spectroscopic dimensions
                                             dtype=np.float32,  # data type / precision
                                             main_dset_attrs=parm_dict)

    usid.hdf_utils.copy_attributes(h5_y, h5_gp)

    h5_y.file.flush()
    
    return

def check_response(h5_main, pixel=0, ph=0):
    
    parm_dict = usid.hdf_utils.get_attributes(h5_main)
    freq = parm_dict['drive_freq']
    txl = np.linspace(0, parm_dict['total_time'], h5_main[pixel,:].shape[0])
    
    resp_wfm = np.sin(txl * 2 * np.pi * freq + ph)

    plt.figure()
    plt.plot(resp_wfm, h5_main[()][pixel,:])
    
    return

