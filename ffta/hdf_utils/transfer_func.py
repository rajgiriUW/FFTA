# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:10:38 2019

@author: Raj
"""

import pycroscopy as px
import pyUSID as usid

from igor import binarywave as bw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def transfer_function(h5_file, tf_file = '', params_file = '', 
                      psd_freq=1e6, plot=False):
    '''
    Reads in the transfer function .ibw, then creates two datasets within
    a parent folder 'Transfer_Function'
    
    This will destructively overwrite an existing Transfer Function in there
    
    1) TF (transfer function)
    2) Freq (frequency axis for computing Fourier Transforms)
    
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
    h5_file['Transfer_Function'].create_dataset('TFnorm', data = tfnorm)
    
    if plot:
        plt.figure()
        plt.plot(freq, data)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (m)')
        plt.title('Transfer Function')
    
    return h5_file['Transfer_Function']


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