# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:07:06 2020

@author: Raj
"""

import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from pyUSID import Dimension
from pyUSID import Process
from pyUSID.io.hdf_utils import write_main_dataset, find_dataset, create_results_group
from sidpy.hdf.hdf_utils import get_attributes, copy_attributes
from sidpy.proc.comp_utils import parallel_compute
from sidpy.viz import plot_utils
from skimage import restoration

import ffta
from ffta.load import get_utils
from ffta.pixel import Pixel
from ffta.pixel_utils import badpixels
from ffta.simulation.impulse import impulse
from ffta.simulation.utils.load import params_from_experiment as load_parm


class FFtrEFM(Process):
    """
    This class processes the deflection data into instantaneous frequency and tFP
    Implements the pixel-by-pixel processing using ffta.pixel routines
    Abstracted using the Process class for parallel processing on image dataset
    
    Example usage:

        >> from ffta.hdf_utils import process
        >> data = process.FFtrEFM(h5_main)
        >> data.test([1,2]) # tests on pixel 1,2 in row, column
        >> data.compute()
        >> data.reshape() # reshapes the tFP, shift data
        >> process.save_CSV_from_file(data.h5_main.file, data.h5_results_grp.name)
        >> process.plot_tfp(data)

    To reload old data:

        >> data = FFtrEFM()
        >> data._get_existing_datasets()
    """

    def __init__(self, h5_main, parm_dict={}, can_params={},
                 pixel_params={}, if_only=False, override=False, process_name='Fast_Free',
                 **kwargs):
        """

        :param h5_main: Dataset to process
        :type h5_main: h5py.Dataset object
            
        :param parm_dict: Additional updates to the parameters dictionary. e.g. changing the trigger.
            You can also explicitly update self.parm_dict.update({'key': value})
        :type parm_dict: dict, optional
            
        :param can_params: Cantilever parameters describing the behavior
            Can be loaded from ffta.pixel_utils.load.cantilever_params
        :type can_params: dict, optional
        
        :param pixel_params:
            Pixel class accepts the following:
                fit: bool, (default: True)
                    Whether to fit the frequency data or use derivative. 
                pycroscopy: bool (default: False)
                    Usually does not need to change. This parameter is because of
                    how Pycroscopy stores a matrix compared to original numpy ffta approach
                method: str (default: 'hilbert')
                    Method for generating instantaneous frequency, amplitude, and phase response
                    One of 
                        hilbert: Hilbert transform method (default)
                        wavelet: Morlet CWT approach
                        emd: Hilbert-Huang decomposition
                        fft: sliding FFT approach
                        fit_form: str (default: 'product')
                filter_amp : bool (default: False)
                    Whether to filter the amplitude signal around DC (to remove drive sine artifact)
        :type pixel_params: dict, optional
            
        :param if_only: If True, only calculates the instantaneous frequency and not tfp/shift
        :type if_only: bool, optional
        
        :param override: If True, forces creation of new results group. Use in _get_existing_datasets
        :type override : bool, optional
            
        :param process_name:
        :type process_name: str, optional
        
        :param kwargs: Keyword pairs to pass to Process constructor
        :type kwargs: dictionary or variable
            
        """

        self.parm_dict = parm_dict
        if not any(parm_dict):
            self.parm_dict = get_attributes(h5_main)
            self.parm_dict.update({'if_only': if_only})

        self.parm_dict['deconvolve'] = False 

        for key, val in parm_dict.items():
            self.parm_dict.update({key: val})

        if any(can_params):
            if 'Initial' in can_params:  # only care about the initial conditions
                for key, val in can_params['Initial'].items():
                    self.parm_dict.update({key: val})
            else:
                for key, val in can_params.items():
                    self.parm_dict.update({key: val})

        self.pixel_params = pixel_params
        self.override = override

        self.impulse = None

        super(FFtrEFM, self).__init__(h5_main, process_name, parms_dict=self.parm_dict, **kwargs)

        # For accidental passing ancillary datasets from Pycroscopy, this will fail
        # when pickling
        if hasattr(self, 'Position_Indices'):
            del self.Position_Indices
        if hasattr(self, 'Position_Values'):
            del self.Position_Values
        if hasattr(self, 'Spectroscopic_Indices'):
            del self.Spectroscopic_Indices
        if hasattr(self, 'Spectroscopic_Values'):
            del self.Spectroscopic_Values

        if 'Position_Indices' in self.parm_dict:
            del self.parm_dict['Position_Indices']
        if 'Position_Values' in self.parm_dict:
            del self.parm_dict['Position_Values']
        if 'Spectroscopic_Indices' in self.parm_dict:
            del self.parm_dict['Spectroscopic_Indices']
        if 'Spectroscopic_Values' in self.parm_dict:
            del self.parm_dict['Spectroscopic_Values']

        return

    def update_parm(self, **kwargs):
        """
        Update the parameters, see ffta.pixel.Pixel for details on what to update
        e.g. to switch from default Hilbert to Wavelets, for example
        
        :param kwargs:
        :type kwargs:
        
        """
        self.parm_dict.update(kwargs)

        return

    def impulse_response(self, can_path, voltage = None, plot=True):
        """
        Generates impulse response using simulation with given cantilever parameters file        

        :param can_path: Path to cantilever parameters.txt file 
        :type can_path: str

        :param voltage: Voltage to simulate the impulse response at
        :type param: float
    
        :param plot: Whether the plot the processed instantaneous frequency/Pixel response
        :type param: bool
        """
        
        can_params, force_params, sim_params, _, _ = load_parm(can_path, self.parm_dict)
        pix, cant = impulse([can_params, force_params, sim_params], voltage = voltage,
                            plot=plot, param_cfg=self.parm_dict, **self.parm_dict)
        
        self.impulse = pix.inst_freq
        self.parm_dict['impulse_window'] = [0, len(self.h5_main)-1]
        self.parm_dict['deconvolve'] = True
        
        print("Set self.parm_dict['deconvolve']=False if deconvolution is undesirable")
        
        return
    
    def test_deconv(self, window, pixel_ind=[0, 0], iterations=10):
        """
        Tests the deconvolution by bracketing the impulse around window
        A reasonable window size might be 100 us pre-trigger to 500 us post-trigger
        
        :param window: List of format [left_index, right_index] for impulse
        :type window: list
        
        :param pixel_ind: Index of the pixel in the dataset that the process needs to be tested on.
            If a list it is read as [row, column]
        :type pixel_ind: uint or list
            
        
        :param iterations: Number of Richardson-Lucy deconvolution iterations
        :type iterations: int

        """
        if len(window) != 2 or not isinstance(window, list):
            raise ValueError('window must be specified[left, right]')
        
        if isinstance(window[0], float): # passed a time index
            window = np.array(window)/self.parm_dict['sampling_rate']
            window = int(window)
        
        if type(pixel_ind) is not list:
            col = int(pixel_ind % self.parm_dict['num_rows'])
            row = int(np.floor(pixel_ind % self.parm_dict['num_rows']))
            pixel_ind = [row, col]

        # as an array, not an ffta.Pixel
        defl = get_utils.get_pixel(self.h5_main, pixel_ind, array_form=True)

        pixraw = Pixel(defl, self.parm_dict, **self.pixel_params)
        tfp, shift, inst_freq = pixraw.analyze()
        
        impulse = self.impulse[window[0]:window[1]]
        self.parm_dict['impulse_window'] = window
        self.parm_dict['conv_iterations'] = iterations
        
        pixconv = restoration.richardson_lucy(inst_freq, impulse, clip=False, num_iter=iterations)
        
        pixrl = pixraw
        tfp_raw = pixraw.tfp
        fit_raw = pixraw.best_fit
        if_raw = pixraw.inst_freq
        
        pixrl.inst_freq = pixconv
        pixrl.find_tfp()
        tfp_conv = pixrl.tfp
        fit_conv = pixrl.best_fit
        if_conv = pixrl.inst_freq
    
        print(tfp_raw, tfp_conv)
    
        # Plot the results of the original+fit compared to deconvolution+fit
        ridx = int(self.parm_dict['roi'] * self.parm_dict['sampling_rate'])
        fidx = int(pixraw.tidx)    
        ifx = np.array([fidx-1100, fidx+ridx])*1e3 / self.parm_dict['sampling_rate']
        windx = np.array(window)*1e3 / self.parm_dict['sampling_rate']
        yl0 = [if_raw[fidx:(fidx + ridx)].min(), if_raw[fidx:(fidx + ridx)].max()]
        yl1 = [if_conv[fidx:(fidx + ridx)].min(), if_conv[fidx:(fidx + ridx)].max()]
        yl2 = [self.impulse[fidx:(fidx + ridx)].min(), self.impulse[fidx:(fidx + ridx)].max()]
    
        fig, ax = plt.subplots(nrows=2, ncols=2, facecolor='white', figsize=(12, 8))
        tx = np.arange(0, pixraw.total_time, 1/pixraw.sampling_rate) * 1e3
        
        ax[0][0].plot(tx, if_raw, 'b', label='Pre-Conv')
        ax[0][0].plot(tx[fidx:(fidx + ridx)], fit_raw, 'r--', label='Pre-Conv, fit')
        ax[0][0].set_title('Raw')
        
        ax[1][0].plot(tx, if_conv, 'b', label='Conv')
        ax[1][0].plot(tx[fidx:(fidx + ridx)], fit_conv, 'r--', label='Conv, fit')
        ax[1][0].set_title('Deconvolved')
        
        ax[0][1].plot(tx, self.impulse, 'k')
        ax[0][1].axvspan(windx[0], windx[1], alpha=0.5, color='red')
        ax[0][1].set_title('Impulse response')
        
        ax[1][1].plot(tx[fidx:(fidx + ridx)], fit_raw, 'r--', label='Pre-Conv, fit')
        ax[1][1].plot(tx[fidx:(fidx + ridx)], fit_conv, 'k--', label='Conv, fit')
        ax[1][1].set_title('Comparing fits')
        ax[1][1].legend()
        
        ax[0][0].set_xlim(ifx)
        ax[0][0].set_ylim(yl0)
        ax[1][0].set_xlim(ifx)
        ax[1][0].set_ylim(yl1)
        ax[0][1].set_xlim(ifx)
        ax[0][1].set_ylim(yl2)
        ax[1][1].set_xlim(ifx)
        ax[1][0].set_xlabel('Time (ms)')
        ax[1][1].set_xlabel('Time (ms)')
        
        plt.tight_layout()

        return pixrl
    

    def test(self, pixel_ind=[0, 0]):
        """
        Test the Pixel analysis of a single pixel

        :param pixel_ind: Index of the pixel in the dataset that the process needs to be tested on.
            If a list it is read as [row, column]
        :type pixel_ind: uint or list
            

        :returns: List [inst_freq, tfp, shift]
            WHERE
            array inst_freq is the instantaneous frequency array for that pixel
            float tfp is the time to first peak
            float shift the frequency shift at time t=tfp (i.e. maximum frequency shift)

        """
        # First read the HDF5 dataset to get the deflection for this pixel

        if type(pixel_ind) is not list:
            col = int(pixel_ind % self.parm_dict['num_rows'])
            row = int(np.floor(pixel_ind % self.parm_dict['num_rows']))
            pixel_ind = [row, col]

        # as an array, not an ffta.Pixel
        defl = get_utils.get_pixel(self.h5_main, pixel_ind, array_form=True)

        pix = Pixel(defl, self.parm_dict, **self.pixel_params)

        tfp, shift, inst_freq = pix.analyze()
        pix.plot()

        return self._map_function(defl, self.parm_dict, self.pixel_params, self.impulse)

    def _create_results_datasets(self):
        '''
        Creates the datasets an Groups necessary to store the results.
        
        Parameters
        ----------
        h5_if : 'Inst_Freq' h5 Dataset
            Contains the Instantaneous Frequencies
            
        tfp : 'tfp' h5 Dataset
            Contains the time-to-first-peak data as a 1D matrix
            
        shift : 'shift' h5 Dataset
            Contains the frequency shift data as a 1D matrix
        '''

        print('Creating results datasets')

        # Get relevant parameters
        num_rows = self.parm_dict['num_rows']
        num_cols = self.parm_dict['num_cols']
        pnts_per_avg = self.parm_dict['pnts_per_avg']

        ds_shape = [num_rows * num_cols, pnts_per_avg]

        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

        copy_attributes(self.h5_main.parent, self.h5_results_grp)

        # Create dimensions
        pos_desc = [Dimension('X', 'm', np.linspace(0, self.parm_dict['FastScanSize'], num_cols)),
                    Dimension('Y', 'm', np.linspace(0, self.parm_dict['SlowScanSize'], num_rows))]

        # ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
        spec_desc = [Dimension('Time', 's', np.linspace(0, self.parm_dict['total_time'], pnts_per_avg))]
        # ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)

        # Writes main dataset
        self.h5_if = write_main_dataset(self.h5_results_grp,
                                        ds_shape,
                                        'Inst_Freq',  # Name of main dataset
                                        'Frequency',  # Physical quantity contained in Main dataset
                                        'Hz',  # Units for the physical quantity
                                        pos_desc,  # Position dimensions
                                        spec_desc,  # Spectroscopic dimensions
                                        dtype=np.float32,  # data type / precision
                                        main_dset_attrs=self.parm_dict)

        self.h5_amp = write_main_dataset(self.h5_results_grp,
                                         ds_shape,
                                         'Amplitude',  # Name of main dataset
                                         'Amplitude',  # Physical quantity contained in Main dataset
                                         'nm',  # Units for the physical quantity
                                         None,  # Position dimensions
                                         None,  # Spectroscopic dimensions
                                         h5_pos_inds=self.h5_main.h5_pos_inds,  # Copy Pos Dimensions
                                         h5_pos_vals=self.h5_main.h5_pos_vals,
                                         h5_spec_inds=self.h5_main.h5_spec_inds,
                                         # Copy Spectroscopy Dimensions
                                         h5_spec_vals=self.h5_main.h5_spec_vals,
                                         dtype=np.float32,  # data type / precision
                                         main_dset_attrs=self.parm_dict)

        self.h5_phase = write_main_dataset(self.h5_results_grp,
                                           ds_shape,
                                           'Phase',  # Name of main dataset
                                           'Phase',  # Physical quantity contained in Main dataset
                                           'degrees',  # Units for the physical quantity
                                           None,  # Position dimensions
                                           None,  # Spectroscopic dimensions
                                           h5_pos_inds=self.h5_main.h5_pos_inds,  # Copy Pos Dimensions
                                           h5_pos_vals=self.h5_main.h5_pos_vals,
                                           h5_spec_inds=self.h5_main.h5_spec_inds,
                                           # Copy Spectroscopy Dimensions
                                           h5_spec_vals=self.h5_main.h5_spec_vals,
                                           dtype=np.float32,  # data type / precision
                                           main_dset_attrs=self.parm_dict)

        self.h5_pwrdis = write_main_dataset(self.h5_results_grp,
                                            ds_shape,
                                            'PowerDissipation',  # Name of main dataset
                                            'Power',  # Physical quantity contained in Main dataset
                                            'W',  # Units for the physical quantity
                                            None,  # Position dimensions
                                            None,  # Spectroscopic dimensions
                                            h5_pos_inds=self.h5_main.h5_pos_inds,  # Copy Pos Dimensions
                                            h5_pos_vals=self.h5_main.h5_pos_vals,
                                            h5_spec_inds=self.h5_main.h5_spec_inds,
                                            # Copy Spectroscopy Dimensions
                                            h5_spec_vals=self.h5_main.h5_spec_vals,
                                            dtype=np.float32,  # data type / precision
                                            main_dset_attrs=self.parm_dict)

        _arr = np.zeros([num_rows * num_cols, 1])
        self.h5_tfp = self.h5_results_grp.create_dataset('tfp', data=_arr, dtype=np.float32)
        self.h5_shift = self.h5_results_grp.create_dataset('shift', data=_arr, dtype=np.float32)

        self.h5_if.file.flush()

        return

    def reshape(self, cal=None):
        
        '''
        Reshapes the tFP and shift data to be a matrix, then saves that dataset instead of the 1D
        
        :param cal:
        :type cal: UNivariateSpline file from ffta.simulation.cal_curve
        '''

        h5_tfp = self.h5_tfp[()]
        h5_shift = self.h5_shift[()]

        num_rows = self.parm_dict['num_rows']
        num_cols = self.parm_dict['num_cols']

        h5_tfp = np.reshape(h5_tfp, [num_rows, num_cols])
        h5_shift = np.reshape(h5_shift, [num_rows, num_cols])

        del self.h5_tfp.file[self.h5_tfp.name]
        del self.h5_tfp.file[self.h5_shift.name]

        self.h5_tfp = self.h5_results_grp.create_dataset('tfp', 
                                                         data=h5_tfp, 
                                                         dtype=np.float32)
        self.h5_shift = self.h5_results_grp.create_dataset('shift', 
                                                           data=h5_shift, 
                                                           dtype=np.float32)

        if cal:
            
            if 'tfp_cal' in self.h5_results_grp:
                del self.h5_results_grp['tfp_cal']
            tfp_cal = np.exp(cal(np.log(h5_tfp))) # cal spline is by default in log space
            self.h5_tfp_cal = self.h5_results_grp.create_dataset('tfp_cal', 
                                                                 data=tfp_cal, 
                                                                 dtype=np.float32)
            tfps = np.linspace(self.h5_tfp[()].min(), self.h5_tfp[()].max(), 50)
            if 'cal_curve' in self.h5_results_grp:
                del self.h5_results_grp['cal_curve']
            _ = self.h5_results_grp.create_dataset('cal_curve',
                                                   data = np.array([tfps, cal(np.log(tfps))]),
                                                   dtype=np.float32)

        return

    def _write_results_chunk(self):
        '''
        Write the computed results back to the H5
        In this case, there isn't any more additional post-processing required
        '''
        # Find out the positions to write to:
        pos_in_batch = self._get_pixels_in_current_batch()

        # unflatten the list of results, which are [inst_freq array, amp, phase, tfp, shift]
        _results = np.array([j for i in self._results for j in i[:1]])
        _amp = np.array([j for i in self._results for j in i[1:2]])
        _phase = np.array([j for i in self._results for j in i[2:3]])
        _tfp = np.array([j for i in self._results for j in i[3:4]])
        _shift = np.array([j for i in self._results for j in i[4:5]])
        _pwr = np.array([j for i in self._results for j in i[5:]])

        # write the results to the file
        self.h5_if[pos_in_batch, :] = _results
        self.h5_amp[pos_in_batch, :] = _amp
        self.h5_pwrdis[pos_in_batch, :] = _pwr
        self.h5_phase[pos_in_batch, :] = _phase
        self.h5_tfp[pos_in_batch, 0] = _tfp
        self.h5_shift[pos_in_batch, 0] = _shift

        return

    def _get_existing_datasets(self, index=-1):
        """
        Extracts references to the existing datasets that hold the results
        
        :param index: which existing dataset to get
        :type index:
        
        """

        if not self.override:

            self.h5_results_grp = find_dataset(self.h5_main.parent, 'Inst_Freq')[index].parent
            self.h5_new_spec_vals = self.h5_results_grp['Spectroscopic_Values']
            self.h5_tfp = self.h5_results_grp['tfp']
            self.h5_shift = self.h5_results_grp['shift']
            self.h5_if = self.h5_results_grp['Inst_Freq']
            self.h5_amp = self.h5_results_grp['Amplitude']
            self.h5_phase = self.h5_results_grp['Phase']
            try:
                self.h5_pwrdis = self.h5_results_grp['PowerDissipation']
                self.h5_tfp_cal = self.h5_results_grp['tfp_cal']
            except:
                pass
        return

    def _unit_computation(self, *args, **kwargs):
        """
        The unit computation that is performed per data chunk. This allows room for any data pre / post-processing
        as well as multiple calls to parallel_compute if necessary
        
        :param args:
        :type args:
        
        :param kwargs:
        :type kwargs:
        """

        args = [self.parm_dict, self.pixel_params, self.impulse]

        if self.verbose and self.mpi_rank == 0:
            print("Rank {} at Process class' default _unit_computation() that "
                  "will call parallel_compute()".format(self.mpi_rank))
        self._results = parallel_compute(self.data, self._map_function, cores=self._cores,
                                         lengthy_computation=False,
                                         func_args=args, func_kwargs=kwargs,
                                         verbose=self.verbose)

    @staticmethod
    def _map_function(defl, *args, **kwargs):
        """
        
        :param defl:
        :type defl:
        
        :param args:
        :type args:
        
        :param kwargs:
        :type kwargs:
        
        :returns: List [inst_freq, amplitude, phase, tfp, shift, pwr_diss]
            WHERE
            [type] inst_freq is...
            [type] amplitude is...
            [type] phase is...
            [type] tfp is...
            [type] shift is...
            [type] pwr_diss is...
        """
        parm_dict = args[0]
        pixel_params = args[1]
        impulse = args[2]
        
        pix = Pixel(defl, parm_dict, **pixel_params)

        if parm_dict['if_only']:
            inst_freq, _, _ = pix.generate_inst_freq()
            tfp = 0
            shift = 0
            amplitude = 0
            phase = 0
            pwr_diss = 0
        elif parm_dict['deconvolve']:
            iterations = parm_dict['conv_iterations']
            impulse = impulse[parm_dict['impulse_window'][0]:parm_dict['impulse_window'][1]]
            inst_freq, amplitude, phase = pix.generate_inst_freq()
            conv = restoration.richardson_lucy(inst_freq, impulse, 
                                               clip=False, num_iter=iterations)
            pix.inst_freq = conv
            pix.find_tfp()
            tfp = pix.tfp
            shift = pix.shift
            pix.calculate_power_dissipation()
            pwr_diss = pix.power_dissipated
        else:
            tfp, shift, inst_freq = pix.analyze()
            pix.calculate_power_dissipation()
            amplitude = pix.amplitude
            phase = pix.phase
            pwr_diss = pix.power_dissipated

        return [inst_freq, amplitude, phase, tfp, shift, pwr_diss]


def save_CSV_from_file(h5_file, h5_path='/', append='', mirror=False, offset=0):
    """
    Saves the tfp, shift, and fixed_tfp as CSV files
    
    :param h5_file: Reminder you can always type: h5_svd.file or h5_avg.file for this
    :type h5_file: H5Py file of FFtrEFM class
        
    :param h5_path: specific folder path to search for the tfp data. Usually not needed.
    :type h5_path: str, optional
        
    :param append: text to append to file name
    :type append: str, optional
    
    :param mirror:
    :type mirror: bool, optional
    
    :param offset: if calculating tFP with a fixed offset for fitting, this subtracts it out
    :type offset: float
        
    """

    h5_ff = h5_file

    if isinstance(h5_file, ffta.hdf_utils.process.FFtrEFM):
        print('Saving from FFtrEFM Class')
        h5_ff = h5_file.h5_main.file
        h5_path = h5_file.h5_results_grp.name

    elif not isinstance(h5_file, h5py.File):
        print('Saving from pyUSID object')
        h5_ff = h5_file.file

    tfp = find_dataset(h5_ff[h5_path], 'tfp')[0][()]
    shift = find_dataset(h5_ff[h5_path], 'shift')[0][()]
    
    try:
        tfp_cal = find_dataset(h5_ff[h5_path], 'tfp_cal')[0][()]
    except:
        tfp_cal = None
        
    tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
    tfp_fixed = np.array(tfp_fixed)
    
    if isinstance(tfp_cal, np.ndarray):
        
        tfp_cal_fixed, _ = badpixels.fix_array(tfp_cal, threshold=2)
        tfp_cal_fixed = np.array(tfp_cal_fixed)

    print(find_dataset(h5_ff[h5_path], 'shift')[0].parent.name)

    path = h5_ff.file.filename.replace('\\', '/')
    path = '/'.join(path.split('/')[:-1]) + '/'
    os.chdir(path)

    if mirror:
        np.savetxt('tfp-' + append + '.csv', np.fliplr(tfp - offset).T, delimiter=',')
        np.savetxt('shift-' + append + '.csv', np.fliplr(shift).T, delimiter=',')
        np.savetxt('tfp_fixed-' + append + '.csv', np.fliplr(tfp_fixed - offset).T, delimiter=',')
    else:
        np.savetxt('tfp-' + append + '.csv', (tfp- offset).T, delimiter=',')
        np.savetxt('shift-' + append + '.csv', shift.T, delimiter=',')
        np.savetxt('tfp_fixed-' + append + '.csv', (tfp_fixed- offset).T, delimiter=',')

    if isinstance(tfp_cal, np.ndarray):
        
        if mirror:
            np.savetxt('tfp_cal-' + append + '.csv', np.fliplr(tfp_cal- offset).T, delimiter=',')
            np.savetxt('tfp_cal_fixed-' + append + '.csv', np.fliplr(tfp_cal_fixed- offset).T, delimiter=',')
        else:
            np.savetxt('tfp_cal-' + append + '.csv', (tfp_cal- offset).T, delimiter=',')
            np.savetxt('tfp_cal_fixed-' + append + '.csv', (tfp_cal_fixed- offset).T, delimiter=',')

    return


def plot_tfp(ffprocess, scale_tfp=1e6, scale_shift=1, threshold=2, **kwargs):
    '''
    Quickly plots the tfp and shift data. If there's a height image in the h5_file associated
     with ffprocess, will plot that as well
    
    :param ffprocess:
    :type ffprocess: FFtrEFM class object (inherits Process) or the parent Group
    
    :param scale_tfp:
    :type scale_tfp:
    
    :param scale_shift:
    :type scale_shift:
    
    :param threshold:
    :type threshold:
    
    :param kwargs:
    :type kwargs:
    
    :returns: tuple (fig, a)
        WHERE
        fig is figure object
        ax is axes object
    '''
    fig, a = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))

    tfp_ax = a[0][1]
    shift_ax = a[1][1]
    tfp_cal_ax = a[1][0]

    if isinstance(ffprocess, ffta.hdf_utils.process.FFtrEFM):

        img_length = ffprocess.parm_dict['FastScanSize']
        img_height = ffprocess.parm_dict['SlowScanSize']
        
        num_cols = ffprocess.parm_dict['num_cols']
        num_rows = ffprocess.parm_dict['num_rows']
        
        tfp = ffprocess.h5_tfp[()]
        shift = ffprocess.h5_shift[()]
        
        if 'tfp_cal' in ffprocess.h5_tfp.parent:
            tfp_cal = ffprocess.h5_tfp.parent['tfp_cal'][()]
            tfp_cal_fixed, _ = badpixels.fix_array(tfp_cal, threshold=threshold)
        else:
            tfp_cal_fixed = tfp_cal = np.ones(tfp.shape)
    
    elif isinstance(ffprocess, h5py.Group):
        
        attr = get_attributes(ffprocess['Inst_Freq'])
        img_length = attr['FastScanSize']
        img_height = attr['SlowScanSize']
        
        num_cols = attr['num_cols']
        num_rows = attr['num_rows']
        
        tfp = ffprocess['tfp'][()]
        shift = ffprocess['shift'][()]
        
        if 'tfp_cal' in ffprocess:
            tfp_cal = ffprocess['tfp_cal'][()]
            tfp_cal_fixed, _ = badpixels.fix_array(tfp_cal, threshold=threshold)
        else:
            tfp_cal_fixed = tfp_cal = np.ones(tfp.shape) 
            
    kwarg = {'origin': 'lower', 'x_vec': img_length * 1e6,
             'y_vec': img_height * 1e6, 'num_ticks': 5, 'stdevs': 3, 'show_cbar': True}

    for k, v in kwarg.items():
        if k not in kwargs:
            kwargs.update({k: v})

    try:
        ht = ffprocess.h5_main.file['/height_000/Raw_Data'][:, 0]
        ht = np.reshape(ht, [num_cols, num_rows]).transpose()
        ht_ax = a[0][0]
        ht_image, cbar = plot_utils.plot_map(ht_ax, ht * 1e9, cmap='gray', **kwarg)
        cbar.set_label('Height (nm)', rotation=270, labelpad=16)
    except:
        pass

    tfp_ax.set_title('tFP Image')
    shift_ax.set_title('Shift Image')

    tfp_fixed, _ = badpixels.fix_array(tfp, threshold=threshold)


    tfp_image, cbar_tfp = plot_utils.plot_map(tfp_ax, tfp_fixed * scale_tfp,
                                              cmap='inferno', **kwargs)
    shift_image, cbar_sh = plot_utils.plot_map(shift_ax, shift * scale_shift,
                                               cmap='inferno', **kwargs)
    try:
        tfp_cal_image, cbar_tfp_cal = plot_utils.plot_map(tfp_cal_ax, tfp_cal_fixed * scale_tfp,
                                                          cmap='inferno', **kwargs)
        cbar_tfp_cal.set_label('Time Calib. (us)', rotation=90, labelpad=16)
    except:
        tfp_cal_image = tfp_cal_ax.imshow(tfp_cal_fixed * scale_tfp,
                                                          cmap='inferno', origin='lower')
    
    cbar_tfp.set_label('Time (us)', rotation=270, labelpad=16)
    cbar_sh.set_label('Frequency Shift (Hz)', rotation=270, labelpad=16)

    text = tfp_ax.text(num_cols / 2, num_rows + 3, '')

    return fig, a
