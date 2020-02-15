# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:07:06 2020

@author: Raj
"""

import pyUSID as usid
import pycroscopy as px
import ffta
import numpy as np
from ffta.hdf_utils import hdf_utils, get_utils
from pyUSID.io.write_utils import Dimension
from pyUSID.io.hdf_utils import write_main_dataset, create_results_group, create_empty_dataset
'''
To do : allow custom parameters dictionary and input to the pixel test function

'''

class FFtrEFM(usid.Process):
    '''
    Implements the pixel-by-pixel processing using ffta.pixel routines
    Abstracted using the Process class for parallel processing
    '''
    def __init__(self, h5_main, **kwargs):
        '''
        Parameters
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        '''
        
        self.parm_dict = get_utils.get_params(h5_main)
        super(FFtrEFM, self).__init__(h5_main, 'Fast_Free', parms_dict=self.parm_dict, **kwargs)
        
        return
    
    def test(self, pixel_ind):
        '''
        Test the algorithm on a single pixel

        Parameters
        ----------
        pixel_ind : uint or list
            Index of the pixel in the dataset that the process needs to be tested on.
            If a list it is read as [row, column]
            
            
        '''
        # First read the HDF5 dataset to get the deflection for this pixel
        
        if type(pixel_ind) is not list:
            col = pixel_ind % self.parm_dict['num_rows']
            row = np.floor(pixel_ind % self.parm_dict['num_rows'])
            pixel_ind = [row, col]
        
        # as an array, not an ffta.Pixel
        defl = get_utils.get_pixel(self.h5_main, pixel_ind, array_form=True)

        return self._map_function(defl, self.parm_dict)
    
    def _create_results_datasets(self):
        '''
        Creates the datasets an Groups necessary to store the results.
        There are only THREE operations happening in this function:
        1. Creation of HDF5 group to hold results
        2. Writing relevant metadata to this HDF5 group
        3. Creation of a HDF5 dataset to hold results

        Please see examples on utilities for writing h5USID files for more information
        '''

        # Get relevant parameters
        num_rows = self.parm_dict['num_rows']
        num_cols = self.parm_dict['num_cols']
        pnts_per_avg = self.parm_dict['pnts_per_avg']
        
        ds_shape = [num_rows * num_cols, pnts_per_avg]
    
        h5_meas_group = usid.hdf_utils.create_indexed_group(self.h5_main, self.process_name)
        usid.hdf_utils.copy_attributes(self.h5_results_grp, h5_meas_group)
    
        # Create dimensions
        pos_desc = [Dimension('X', 'm', np.linspace(0, self.parm_dict['FastScanSize'], num_cols)),
                    Dimension('Y', 'm', np.linspace(0, self.parm_dict['SlowScanSize'], num_rows))]
    
        # ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
        spec_desc = [Dimension('Time', 's', np.linspace(0, self.parm_dict['total_time'], pnts_per_avg))]
        # ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)
    
        # Writes main dataset
        self.h5_if = usid.hdf_utils.write_main_dataset(h5_meas_group,
                                                       ds_shape,
                                                       'Inst_Freq',  # Name of main dataset
                                                       'Frequency',  # Physical quantity contained in Main dataset
                                                       'Hz',  # Units for the physical quantity
                                                       pos_desc,  # Position dimensions
                                                       spec_desc,  # Spectroscopic dimensions
                                                       dtype=np.float32,  # data type / precision
                                                       main_dset_attrs=self.parm_dict)
    
        self.h5_if.file.flush()
        
        # Create the other datasets, tfp, shift, tfp_fixed
        self.h5_tfp = create_empty_dataset(self.h5_if, np.float32, 'tfp')
        self.h5_shift = create_empty_dataset(self.h5_if, np.float32, 'shift')
    
  
        return
    
    
    def _write_results_chunk(self):
        '''
        Write the computed results back to the H5
        In this case, there isn't any more additional post-processing required
        '''
        # Find out the positions to write to:
        pos_in_batch = self._get_pixels_in_current_batch()

        # write the results to the file
        self.h5_if[pos_in_batch, :] = np.array(self._results)
        
        #self.h5_tfp[]
        
        return
    
    @staticmethod
    def _map_function(defl, parm_dict):
        
        pix = ffta.pixel.Pixel(defl, parm_dict)
        
        tfp, shift, inst_freq = pix.analyze()
        
        return inst_freq