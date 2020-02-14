# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:07:06 2020

@author: Raj
"""

import pyUSID as usid
import pycroscopy as px
import ffta
from ffta.hdf_utils import hdf_utils, get_utils

class FFtrEFM(usid.Process):
    '''
    Implements the pixel-by-pixel processing using ffta.pixel routines
    Abstracted using the Process class for parallel processing
    '''
    def __init__(self, h5_main):
        
        self.params = get_utils.get_params(h5_main)
        return
    
    def test(self, pixel_ind):
        
        return
    
    def _create_results_datasets(self):
        
        return
    
    
    def _write_results_chunk(self):
        
        return
    
    @staticmethod
    def _map_function(line):
        
        return