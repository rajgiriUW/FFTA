# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:35:38 2020

@author: Raj
"""

import matplotlib.pyplot as plt

# Script that loads, analyzes, and plots fast free point scan with fit
from ffta.pixel import Pixel
from ffta.pixel_utils.load import configuration
from ffta.pixel_utils.load import signal


def analyze_pixel(ibw_file, param_file):
    '''
    Analyzes a single pixel
    
    :param path to \*.ibw file
    :type ibw_file: str
        
    :param param_file: path to parameters.cfg file
    :type param_file: str
        
    :returns: The pixel object read and analyzed
    :rtype: Pixel
        
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
