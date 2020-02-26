# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:07:06 2020

@author: Raj
"""

import pyUSID as usid
import ffta
import numpy as np
from ffta.hdf_utils import get_utils
from pyUSID.processing.comp_utils import parallel_compute
from pyUSID.io.write_utils import Dimension
from pyUSID.io.hdf_utils import create_results_group
'''
To do:
    
    Separate the instantaneous frequency and tFP/shift calculations

'''

class FFtrEFM(usid.Process):
    '''
    Implements the pixel-by-pixel processing using ffta.pixel routines
    Abstracted using the Process class for parallel processing
    
    Example usage:
    
        >> data = FFtrEFM(h5_main)
        >> data.test([1,2]) # tests on pixel 1,2 in row, column
        >> data.compute()
        >> data.reshape() # reshapes the tFP, shift data
    '''
    def __init__(self, h5_main, if_only=False, **kwargs):
        '''
        Parameters
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        
        if_only : bool, optional
            If True, only calculates the instantaneous frequency
            
        kwargs : dictionary or variable=
            Keyword pairs to pass to Process constructor
        '''
        
        self.parm_dict = get_utils.get_params(h5_main)
        self.parm_dict.update({'if_only': if_only})
        super(FFtrEFM, self).__init__(h5_main, 'Fast_Free', parms_dict=self.parm_dict, **kwargs)
        
        return
    
    def update_parm(self, **kwargs):
        '''
        Update the parameters, see ffta.pixel.Pixel for details on what to update
        e.g. to switch from default Hilbert to Wavelets, for example
        '''
        self.parm_dict.update(kwargs)
        
        return
    
    def test(self, pixel_ind):
        '''
        Test the Pixel analysis of a single pixel

        Parameters
        ----------
        pixel_ind : uint or list
            Index of the pixel in the dataset that the process needs to be tested on.
            If a list it is read as [row, column]
        
        Returns
        -------
        [inst_freq, tfp, shift] : List
            inst_freq : array
                the instantaneous frequency array for that pixel
            tfp : float
                the time to first peak
            shift : float
                the frequency shift at time t=tfp (i.e. maximum frequency shift)
            
        '''
        # First read the HDF5 dataset to get the deflection for this pixel
        
        if type(pixel_ind) is not list:
            col = int(pixel_ind % self.parm_dict['num_rows'])
            row = int(np.floor(pixel_ind % self.parm_dict['num_rows']))
            pixel_ind = [row, col]
        
        # as an array, not an ffta.Pixel
        defl = get_utils.get_pixel(self.h5_main, pixel_ind, array_form=True)

        pix = ffta.pixel.Pixel(defl, self.parm_dict)
        
        tfp, shift, inst_freq = pix.analyze()
        pix.plot()

        return self._map_function(defl, self.parm_dict)
    
    def _create_results_datasets(self):
        '''
        Creates the datasets an Groups necessary to store the results.

        h5_if : 'Inst_Freq' h5 Dataset
            Contains the Instantaneous Frequencies
            
        tfp : 'tfp' h5 Dataset
            Contains the time-to-first-peak data as a 1D matrix
            
        shift : 'shift' h5 Dataset
            Contains the frequency shift data as a 1D matrix
        '''

        # Get relevant parameters
        num_rows = self.parm_dict['num_rows']
        num_cols = self.parm_dict['num_cols']
        pnts_per_avg = self.parm_dict['pnts_per_avg']
        
        ds_shape = [num_rows * num_cols, pnts_per_avg]
    
        self.h5_results_grp = usid.hdf_utils.create_results_group(self.h5_main, self.process_name)
    
        #h5_meas_group = usid.hdf_utils.create_indexed_group(self.h5_main.parent, self.process_name)
        usid.hdf_utils.copy_attributes(self.h5_main.parent, self.h5_results_grp)
    
        # Create dimensions
        pos_desc = [Dimension('X', 'm', np.linspace(0, self.parm_dict['FastScanSize'], num_cols)),
                    Dimension('Y', 'm', np.linspace(0, self.parm_dict['SlowScanSize'], num_rows))]
    
        # ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
        spec_desc = [Dimension('Time', 's', np.linspace(0, self.parm_dict['total_time'], pnts_per_avg))]
        # ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)
    
        # Writes main dataset
        self.h5_if = usid.hdf_utils.write_main_dataset(self.h5_results_grp,
                                                       ds_shape,
                                                       'Inst_Freq',  # Name of main dataset
                                                       'Frequency',  # Physical quantity contained in Main dataset
                                                       'Hz',  # Units for the physical quantity
                                                       pos_desc,  # Position dimensions
                                                       spec_desc,  # Spectroscopic dimensions
                                                       dtype=np.float32,  # data type / precision
                                                       main_dset_attrs=self.parm_dict)
    
        self.h5_if.file.flush()
        
        # Create the other datasets, tfp, shift
        # pos_desc_tfp = [Dimension('X', 'm', np.linspace(0, self.parm_dict['FastScanSize'], num_cols)),
        #                 Dimension('Y', 'm', np.linspace(0, self.parm_dict['SlowScanSize'], num_rows))]
        # spec_desc_tfp = [Dimension('Time', 's', 1),Dimension('Frequency', 'Hz', 2)]
        
        # self.h5_tfp_results_grp = usid.hdf_utils.create_results_group(self.h5_if, 'tfp-calc')
        
        # self.h5_tfp = usid.hdf_utils.write_main_dataset(self.h5_tfp_results_grp,
        #                                                 [num_rows * num_cols, 2],
        #                                                 'processed',  # Name of main dataset
        #                                                 'Time',  # Physical quantity contained in Main dataset
        #                                                 's',  # Units for the physical quantity
        #                                                 pos_desc_tfp,  # Position dimensions
        #                                                 spec_desc_tfp,  # Spectroscopic dimensions
        #                                                 dtype=np.float32,  # data type / precision
        #                                                 main_dset_attrs=self.parm_dict)
        

        _arr = np.zeros([num_rows * num_cols, 1])
        self.h5_tfp = self.h5_results_grp.create_dataset('tfp', data=_arr, dtype=np.float32)
        self.h5_shift = self.h5_results_grp.create_dataset('shift', data=_arr, dtype=np.float32)
  
        return
    
    def reshape(self):
        '''
        Reshapes the tFP and shift data to be a matrix, then saves that dataset instead of the 1D
        '''
        
        h5_tfp = self.h5_tfp[()]
        h5_shift = self.h5_shift[()]
        
        num_rows = self.parm_dict['num_rows']
        num_cols = self.parm_dict['num_cols']
        
        h5_tfp = np.reshape(h5_tfp, [num_rows, num_cols])
        h5_shift = np.reshape(h5_shift, [num_rows, num_cols])
        
        del self.h5_tfp.file[self.h5_tfp.name]
        del self.h5_tfp.file[self.h5_shift.name]
        
        self.h5_tfp = self.h5_results_grp.create_dataset('tfp', data=h5_tfp, dtype=np.float32)
        self.h5_shift = self.h5_results_grp.create_dataset('shift', data=h5_shift, dtype=np.float32)
        
        return
    
    def _write_results_chunk(self):
        '''
        Write the computed results back to the H5
        In this case, there isn't any more additional post-processing required
        '''
        # Find out the positions to write to:
        pos_in_batch = self._get_pixels_in_current_batch()

        # unflatten the list of results, which are [inst_freq array, tfp, shift]
        _results = np.array([j for i in self._results for j in i[:1]])
        _tfp = np.array([j for i in self._results for j in i[1:2]])
        _shift = np.array([j for i in self._results for j in i[2:]])
        
        # write the results to the file
        self.h5_if[pos_in_batch, :] = _results        
        self.h5_tfp[pos_in_batch, 0] = _tfp
        self.h5_shift[pos_in_batch, 0] = _shift
        
        return
   
    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets that hold the results
        """
        self.h5_results_grp = usid.hdf_utils.find_dataset(self.h5_main.parent, 'Inst_Freq')[0].parent
        self.h5_new_spec_vals = self.h5_results_grp['Spectroscopic_Values']
        self.h5_tfp = self.h5_results_grp['tfp']
        self.h5_shift = self.h5_results_grp['shift']
        self.h5_if = self.h5_results_grp['Inst_Freq']

        return
    
    def _unit_computation(self, *args, **kwargs):
        """
        The unit computation that is performed per data chunk. This allows room for any data pre / post-processing
        as well as multiple calls to parallel_compute if necessary
        """
        # TODO: Try to use the functools.partials to preconfigure the map function
        # cores = number of processes / rank here
        
        args = [self.parm_dict]
        
        if self.verbose and self.mpi_rank == 0:
            print("Rank {} at Process class' default _unit_computation() that "
                  "will call parallel_compute()".format(self.mpi_rank))
        self._results = parallel_compute(self.data, self._map_function, cores=self._cores,
                                         lengthy_computation=False,
                                         func_args=args, func_kwargs=kwargs,
                                         verbose=self.verbose)

    @staticmethod
    def _map_function(defl, *args, **kwargs):
        
        parm_dict = args[0]
        
        pix = ffta.pixel.Pixel(defl, parm_dict)
        
        if parm_dict['if_only']:
            inst_freq = pix.generate_inst_freq()
            tfp = 0
            shift = 0
        else:
           tfp, shift, inst_freq = pix.analyze()
        
        return [inst_freq, tfp, shift]