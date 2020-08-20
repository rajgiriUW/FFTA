# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:35:38 2020

@author: Raj
"""

#Script that loads, analyzes, and plots fast free point scan with fit
from ffta.pixel import Pixel
from ffta.pixel_utils.load import signal
from ffta.pixel_utils.load import configuration
import matplotlib.pyplot as plt


def analyze_pixel(ibw_file, param_file):
    '''
    Analyzes a single pixel
    
    Parameters
    ----------
    ibw_file : str
        path to \*.ibw file
    param_file : str
        path to parameters.cfg file
        
    Returns
    -------
    pixel : Pixel
        The pixel object read and analyzed
    '''
    signal_array = signal(ibw_file)
    n_pixels, params = configuration(param_file)
    pixel = Pixel(signal_array, params=params)
    
    pixel.analyze()
    pixel.plot() 
    plt.xlabel('Time Step')
    plt.ylabel('Freq Shift (Hz)')
    
    print('tFP is', pixel.tfp, 's')
    
    return pixel.tfp

