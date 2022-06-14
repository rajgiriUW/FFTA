# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:07:06 2020

@author: Raj
"""

import os
import warnings

import h5py
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from pyUSID import Dimension
from scipy.ndimage import gaussian_filter1d
from sidpy.proc.comp_utils import parallel_compute

import ffta
from ffta.gkpfm.gkpixel import GKPixel
from ffta.hdf_utils.process import FFtrEFM
from ffta.load import get_utils
from ffta.pixel_utils import badpixels

'''
To do:
    Separate the instantaneous frequency and tFP/shift calculations
'''


class GKPFM(FFtrEFM):
    """
    Implements the pixel-by-pixel processing using ffta.pixel routines
    Abstracted using the Process class for parallel processing

    Example usage:

        >> from ffta.hdf_utils import process
        >> data = process.GKPFM(h5_main)
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
                 pixel_params={}, TF_norm=[], exc_wfm=[], periods=2,
                 tip_response='', tip_excitation='', exc_wfm_file='',
                 override=False, exc_floor=10, noise_tolerance=1e-4, **kwargs):
        """
        
        :param h5_main: Dataset to process
        :type h5_main: h5py.Dataset object
        
        :param parm_dict: Additional updates to the parameters dictionary. e.g. changing the trigger.
            You can also explicitly update self.parm_dict.update({'key': value})
            The default is {}.
        :type parm_dict: dict, optional
        
        :param can_params: Cantilever parameters describing the behavior
            Can be loaded from ffta.pixel_utils.load.cantilever_params
            The default is {}.
        :type can_params: dict, optional
        
        :param pixel_params: Currently unimplemented. The default is {}.
        :type pixel_params: dict, optional
        
        :param TF_norm: Transfer function (normalized). Can use GKPixel to generate, or supply explicitly
        :type TF_norm: numpy array, optional
            
        :param exc_wfm: The excitation used to generated Transfer Function. Unneeded if supplying a TF explicitly
        :type exc_wfm: numpy array, optional
            
        :param periods: Num of cantilever cycles to process in CPD calc. The default is 2.
        :type periods: int, optional
        
        :param tip_response: File path to tip_response .ibw. Only needed if calculating TF_norm here
        :type tip_response: string, optional
            
        :param tip_excitation: File path to tip excitation .ibw. Only needed if calculating TF_norm here
        :type tip_excitation: string, optional
            
        :param exc_wfm_file: File path to .ibw for excitation during actual measurement (not TF).
        :type exc_wfm_file: string, optional
            
        :param override: If True, forces creation of new results group. Use in _get_existing_datasets
        :type override: bool, optional
            
        :param exc_floor: The noise floor for calculating Transfer Function. The default is 10.
        :type exc_floor: float, optional
            
        :param noise_tolerance: Use to determine noise_floor in reconstructing force. The default is 1e-4.
        :type noise_tolerance: float, optional
            
        """

        self.parm_dict = parm_dict

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
        self.parm_dict['tip_response'] = tip_response
        self.parm_dict['tip_excitation'] = tip_excitation
        self.exc_wfm = exc_wfm

        self.parm_dict['periods'] = periods
        self.parm_dict['noise_tolerance'] = noise_tolerance

        super().__init__(h5_main, parm_dict=self.parm_dict,
                         can_params=can_params, process_name='GKPFM',
                         **kwargs)

        # save Transfer Function
        if not any(TF_norm):
            self.process_tf(exc_floor)
        else:
            self.TF_norm = TF_norm
        # defl = get_utils.get_pixel(self.h5_main, [0, 0], array_form=True).flatten()
        # _gk = GKPixel(defl, self.parm_dict, exc_wfm=exc_wfm)
        # _gk.load_tf(self.parm_dict['tip_response'], self.parm_dict['tip_excitation'])
        # _gk.process_tf()

        # self.TF_norm = _gk.TF_norm
        # del _gk

        self.parm_dict['denoise'] = False
        self.parm_dict['filter_cpd'] = False

        return

    def update_parm(self, **kwargs):
        """
        Update the parameters, see ffta.pixel.Pixel for details on what to update
        e.g. to switch from default Hilbert to Wavelets, for example
        
        :param **kwargs:
        :type **kwargs:
        """
        self.parm_dict.update(kwargs)

        return

    def process_tf(self, exc_floor=10):
        """
        Generates a Transfer Function using GKPixel function
        
        :param exc_floor:
        :type exc_floor: float

        """

        defl = get_utils.get_pixel(self.h5_main, [0, 0], array_form=True).flatten()
        _gk = GKPixel(defl, self.parm_dict)
        _gk.load_tf(self.parm_dict['tip_response'], self.parm_dict['tip_excitation'])
        _gk.process_tf(exc_floor=exc_floor)

        self.TF_norm = _gk.TF_norm
        del _gk

        if any(np.isnan(self.TF_norm)):
            warnings.warn('TF is NaN! Lower exc_floor value to lower threshold of pass TF_norm explicitly')

        return

    def test(self, pixel_ind=[0, 0], phases_to_test=[2.0708, 2.1208, 2.1708], smooth=None):
        """
        Test the Pixel analysis of a single pixel

        :param pixel_ind: Index of the pixel in the dataset that the process needs to be tested on.
            If a list it is read as [row, column]
        :type pixel_ind: uint or list
        
        :param phases_to_test: Which phases to shift the signal with. The default is [2.0708, 2.1208, 2.1708],
            which is 0.5, 0.55, 0.5 + pi/2
        :type phases_to_test: list, optional
        
        :param smooth:
        :type smooth:
            
            
        :returns: List [inst_freq, tfp, shift]
            WHERE
            array inst_freq is the instantaneous frequency array for that pixel
            float tfp is the time to first peak
            float shift     
            shift is the frequency shift at time t=tfp (i.e. maximum frequency shift)

        """
        # First read the HDF5 dataset to get the deflection for this pixel

        if type(pixel_ind) is not list:
            col = int(pixel_ind % self.parm_dict['num_rows'])
            row = int(np.floor(pixel_ind % self.parm_dict['num_rows']))
            pixel_ind = [row, col]

        # as an array, not an ffta.Pixel
        defl = get_utils.get_pixel(self.h5_main, pixel_ind, array_form=True).flatten()

        _gk = GKPixel(defl, self.parm_dict, exc_wfm=self.exc_wfm,
                      TF_norm=self.TF_norm)

        _gk.min_phase(phases_to_test=phases_to_test, noise_tolerance=self.parm_dict['noise_tolerance'],
                      verbose=False)
        print("Set self.parm_dict['phase_shift'] to desired value")

        # If User supplies a phase shift
        if 'phase_shift' in self.parm_dict:
            _gk.force_out(plot=True, noise_tolerance=self.parm_dict['noise_tolerance'],
                          phase_shift=self.parm_dict['phase_shift'])
        else:
            _gk.force_out(plot=True, noise_tolerance=self.parm_dict['noise_tolerance'])
            self.parm_dict['phase_shift'] = phases_to_test[-1]

        if self.parm_dict['denoise']:
            print('aa')
            _gk.noise_filter()

        try:
            _gk.analyze_cpd(use_raw=False, periods=self.parm_dict['periods'])
        except LinAlgError:
            print('Make sure TF_norm is set correctly. Pass TF_norm explicitly.')
            return _gk

        if self.parm_dict['filter_cpd']:
            print('bb')
            _gk.CPD = gaussian_filter1d(_gk.CPD, 1)[:_gk.num_CPD]

        _gk.plot_cpd(smooth)

        self.cpd_dict = _gk._calc_cpd_params(return_dict=True, periods=self.parm_dict['periods'])

        _, _, _, = self._map_function(defl, self.parm_dict, self.TF_norm, self.exc_wfm)

        return _gk

    def _create_results_datasets(self):
        '''
        Creates the datasets an Groups necessary to store the results.

        '''

        print('Creating CPD results datasets')

        # Get relevant parameters
        num_rows = self.parm_dict['num_rows']
        num_cols = self.parm_dict['num_cols']
        pnts_per_avg = self.parm_dict['pnts_per_avg']

        ds_shape = [num_rows * num_cols, pnts_per_avg]
        cpd_ds_shape = [num_rows * num_cols, self.cpd_dict['num_CPD']]

        self.h5_results_grp = usid.hdf_utils.create_results_group(self.h5_main, self.process_name)
        self.h5_cpd_grp = usid.hdf_utils.create_results_group(self.h5_main, self.process_name + '_CPD')

        usid.hdf_utils.copy_attributes(self.h5_main.parent, self.h5_results_grp)
        usid.hdf_utils.copy_attributes(self.h5_main.parent, self.h5_cpd_grp)

        # Create dimensions
        pos_desc = [Dimension('X', 'm', np.linspace(0, self.parm_dict['FastScanSize'], num_cols)),
                    Dimension('Y', 'm', np.linspace(0, self.parm_dict['SlowScanSize'], num_rows))]

        # ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
        spec_desc = [Dimension('Time', 's', np.linspace(0, self.parm_dict['total_time'], pnts_per_avg))]
        cpd_spec_desc = [Dimension('Time', 's', np.linspace(0, self.parm_dict['total_time'], self.cpd_dict['num_CPD']))]
        # ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)

        # Writes main dataset
        self.h5_force = usid.hdf_utils.write_main_dataset(self.h5_results_grp,
                                                          ds_shape,
                                                          'force',  # Name of main dataset
                                                          'Force',  # Physical quantity contained in Main dataset
                                                          'N',  # Units for the physical quantity
                                                          pos_desc,  # Position dimensions
                                                          spec_desc,  # Spectroscopic dimensions
                                                          dtype=np.float32,  # data type / precision
                                                          main_dset_attrs=self.parm_dict)

        self.h5_cpd = usid.hdf_utils.write_main_dataset(self.h5_cpd_grp,
                                                        cpd_ds_shape,
                                                        'CPD',  # Name of main dataset
                                                        'Potential',  # Physical quantity contained in Main dataset
                                                        'V',  # Units for the physical quantity
                                                        None,  # Position dimensions
                                                        cpd_spec_desc,  # Spectroscopic dimensions
                                                        h5_pos_inds=self.h5_main.h5_pos_inds,  # Copy Pos Dimensions
                                                        h5_pos_vals=self.h5_main.h5_pos_vals,
                                                        dtype=np.float32,  # data type / precision
                                                        main_dset_attrs=self.parm_dict)

        self.h5_cap = usid.hdf_utils.write_main_dataset(self.h5_cpd_grp,
                                                        cpd_ds_shape,
                                                        'capacitance',  # Name of main dataset
                                                        'Capacitance',  # Physical quantity contained in Main dataset
                                                        'F',  # Units for the physical quantity
                                                        None,  # Position dimensions
                                                        None,
                                                        h5_pos_inds=self.h5_main.h5_pos_inds,  # Copy Pos Dimensions
                                                        h5_pos_vals=self.h5_main.h5_pos_vals,
                                                        h5_spec_inds=self.h5_cpd.h5_spec_inds,
                                                        # Copy Spectroscopy Dimensions
                                                        h5_spec_vals=self.h5_cpd.h5_spec_vals,
                                                        dtype=np.float32,  # data type / precision
                                                        main_dset_attrs=self.parm_dict)

        self.h5_cpd.file.flush()

        return

    def reshape(self):
        '''
        Currently unimplemented. See ffta.hdf_utils.process.FFtrEFM for an example
        '''

        return

    def _write_results_chunk(self):
        '''
        Write the computed results back to the H5
        In this case, there isn't any more additional post-processing required
        '''
        # Find out the positions to write to:
        pos_in_batch = self._get_pixels_in_current_batch()

        # unflatten the list of results, which are [inst_freq array, amp, phase, tfp, shift]
        _force = np.array([j for i in self._results for j in i[:1]])
        _cpd = np.array([j for i in self._results for j in i[1:2]])
        _capacitance = np.array([j for i in self._results for j in i[2:3]])

        # write the results to the file
        self.h5_force[pos_in_batch, :] = _force
        self.h5_cpd[pos_in_batch, :] = _cpd[:, :self.h5_cpd.shape[1]]
        self.h5_cap[pos_in_batch, :] = _capacitance[:, :self.h5_cap.shape[1]]

        return

    def _get_existing_datasets(self, index=-1):
        """
        Extracts references to the existing datasets that hold the results
        
        :param index: which existing dataset to get
        :type index:
        
        """

        if not self.override:
            self.h5_results_grp = usid.hdf_utils.find_dataset(self.h5_main.parent, 'CPD')[index].parent
            self.h5_new_spec_vals = self.h5_results_grp['Spectroscopic_Values']
            self.h5_cpd = self.h5_results_grp['CPD']
            self.h5_capacitance = self.h5_results_grp['capacitance']
            self.h5_force = usid.hdf_utils.find_dataset(self.h5_main.parent, 'CPD')[index].parent['force']

        return

    def _unit_computation(self, *args, **kwargs):
        """
        The unit computation that is performed per data chunk. This allows room for any data pre / post-processing
        as well as multiple calls to parallel_compute if necessary
        
        :param *args:
        :type *args:
        
        :param **kwargs:
        :type **kwargs:
        """

        args = [self.parm_dict, self.TF_norm, self.exc_wfm]

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
        
        :param *args:
        :type *args:
        
        :returns: List [force, cpd, capacitance]
            WHERE
            [type] force is...
            [type] cpd is...
            [type capacitance is...
        """

        parm_dict = args[0]
        TF_norm = args[1]
        exc_wfm = args[2]

        gk = GKPixel(defl, parm_dict, exc_wfm=exc_wfm, TF_norm=TF_norm)
        gk.force_out(noise_tolerance=parm_dict['noise_tolerance'], phase_shift=parm_dict['phase_shift'])

        if parm_dict['denoise']:
            gk.noise_filter()

        gk.analyze_cpd(use_raw=False, periods=parm_dict['periods'])

        cpd = gk.CPD
        if parm_dict['filter_cpd']:
            cpd = gaussian_filter1d(gk.CPD, 1)[:gk.num_CPD]

        capacitance = gk.capacitance
        force = gk.force

        return [force, cpd, capacitance]


def save_CSV_from_file(h5_file, h5_path='/', append='', mirror=False):
    """
    Saves the tfp, shift, and fixed_tfp as CSV files
    
    :param h5_file: Reminder you can always type: h5_svd.file or h5_avg.file for this
    :type h5_file: H5Py file of FFtrEFM class
        
    :param h5_path: specific folder path to search for the tfp data. Usually not needed.
    :type h5_path: str, optional
        
    :param append: text to append to file name
    :type append: str, optional
    
    :param mirror:
    :type mirror: bool
        
    """

    h5_ff = h5_file

    if isinstance(h5_file, ffta.hdf_utils.process.FFtrEFM):
        print('Saving from FFtrEFM Class')
        h5_ff = h5_file.h5_main.file
        h5_path = h5_file.h5_results_grp.name

    elif not isinstance(h5_file, h5py.File):
        print('Saving from pyUSID object')
        h5_ff = h5_file.file

    cpd = usid.hdf_utils.find_dataset(h5_ff[h5_path], 'CPD')[0][()]
    # tfp_fixed = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp_fixed')[0][()]
    capacitance = usid.hdf_utils.find_dataset(h5_ff[h5_path], 'capacitance')[0][()]

    print(usid.hdf_utils.find_dataset(h5_ff[h5_path], 'CPD')[0].parent.name)

    path = h5_ff.file.filename.replace('\\', '/')
    path = '/'.join(path.split('/')[:-1]) + '/'
    os.chdir(path)

    if mirror:
        np.savetxt('cpd-' + append + '.csv', np.fliplr(cpd).T, delimiter=',')
        np.savetxt('capacitance-' + append + '.csv', np.fliplr(capacitance).T, delimiter=',')
    else:
        np.savetxt('cpd-' + append + '.csv', cpd.T, delimiter=',')
        np.savetxt('capacitance-' + append + '.csv', capacitance.T, delimiter=',')

    return


def plot_tfp(ffprocess, scale_tfp=1e6, scale_shift=1, threshold=2, **kwargs):
    '''
    Quickly plots the tfp and shift data. If there's a height image in the h5_file associated
     with ffprocess, will plot that as well
    
    :param ffprocess:
    :type ffprocess: FFtrEFM class object (inherits Process)
    
    :param scale_tfp:
    :type scale_tfp:
    
    :param scale_shift:
    :type scale_shift:
    
    :param threshold:
    :type threshold:
    
    :param **kwargs:
    :type **kwargs:
    
    :returns: tuple (fig, a)
        WHERE
        figure object fig is...
        axes object a is...
    
    '''
    fig, a = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))

    tfp_ax = a[0][1]
    shift_ax = a[1][1]

    img_length = ffprocess.parm_dict['FastScanSize']
    img_height = ffprocess.parm_dict['SlowScanSize']
    kwarg = {'origin': 'lower', 'x_vec': img_length * 1e6,
             'y_vec': img_height * 1e6, 'num_ticks': 5, 'stdevs': 3, 'show_cbar': True}

    for k, v in kwarg.items():
        if k not in kwargs:
            kwargs.update({k: v})

    num_cols = ffprocess.parm_dict['num_cols']
    num_rows = ffprocess.parm_dict['num_rows']
    try:
        ht = ffprocess.h5_main.file['/height/Raw_Data'][:, 0]
        ht = np.reshape(ht, [num_cols, num_rows]).transpose()
        ht_ax = a[0][0]
        ht_image, cbar = usid.viz.plot_utils.plot_map(ht_ax, ht * 1e9, cmap='gray', **kwarg)
        cbar.set_label('Height (nm)', rotation=270, labelpad=16)
    except:
        pass

    tfp_ax.set_title('tFP Image')
    shift_ax.set_title('Shift Image')

    tfp_fixed, _ = badpixels.fix_array(ffprocess.h5_tfp[()], threshold=threshold)

    tfp_image, cbar_tfp = usid.viz.plot_utils.plot_map(tfp_ax, tfp_fixed * scale_tfp,
                                                       cmap='inferno', **kwargs)
    shift_image, cbar_sh = usid.viz.plot_utils.plot_map(shift_ax, ffprocess.h5_shift[()] * scale_shift,
                                                        cmap='inferno', **kwargs)

    cbar_tfp.set_label('Time (us)', rotation=270, labelpad=16)
    cbar_sh.set_label('Frequency Shift (Hz)', rotation=270, labelpad=16)
    text = tfp_ax.text(num_cols / 2, num_rows + 3, '')

    return fig, a
