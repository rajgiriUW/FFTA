# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:07:06 2020

@author: Raj
"""

import pyUSID as usid
import ffta
from ffta.pixel import Pixel
from ffta.pixel_utils import badpixels
import os
import numpy as np
from ffta.load import get_utils
from pyUSID.processing.comp_utils import parallel_compute
from pyUSID.io.write_utils import Dimension
import h5py

from matplotlib import pyplot as plt


class FFtrEFM(usid.Process):
	"""
	Implements the pixel-by-pixel processing using ffta.pixel routines
	Abstracted using the Process class for parallel processing

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
		Parameters
		----------
		h5_main : h5py.Dataset object
			Dataset to process

		if_only : bool, optional
			If True, only calculates the instantaneous frequency and not tfp/shift

		parm_dict : dict, optional
			Additional updates to the parameters dictionary. e.g. changing the trigger.
			You can also explicitly update self.parm_dict.update({'key': value})
		
		can_params : dict, optional
			Cantilever parameters describing the behavior
			Can be loaded from ffta.pixel_utils.load.cantilever_params
		
		pixel_params : dict, optional
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
	
		override : bool, optional
			If True, forces creation of new results group. Use in _get_existing_datasets
	
		kwargs : dictionary or variable
			Keyword pairs to pass to Process constructor
		"""

		self.parm_dict = parm_dict
		self.parm_dict['if_only'] = if_only
		if not any(parm_dict):
			self.parm_dict = usid.hdf_utils.get_attributes(h5_main)
			self.parm_dict.update({'if_only': if_only})

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
		"""
		self.parm_dict.update(kwargs)

		return

	def test(self, pixel_ind=[0, 0]):
		"""
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

		return self._map_function(defl, self.parm_dict, self.pixel_params)

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

		self.h5_results_grp = usid.hdf_utils.create_results_group(self.h5_main, self.process_name)

		# h5_meas_group = usid.hdf_utils.create_indexed_group(self.h5_main.parent, self.process_name)
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

		self.h5_amp = usid.hdf_utils.write_main_dataset(self.h5_results_grp,
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

		self.h5_phase = usid.hdf_utils.write_main_dataset(self.h5_results_grp,
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

		self.h5_pwrdis = usid.hdf_utils.write_main_dataset(self.h5_results_grp,
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
		
		index = which existing dataset to get
		
		"""

		if not self.override:

			self.h5_results_grp = usid.hdf_utils.find_dataset(self.h5_main.parent, 'Inst_Freq')[index].parent
			self.h5_new_spec_vals = self.h5_results_grp['Spectroscopic_Values']
			self.h5_tfp = self.h5_results_grp['tfp']
			self.h5_shift = self.h5_results_grp['shift']
			self.h5_if = self.h5_results_grp['Inst_Freq']
			self.h5_amp = self.h5_results_grp['Amplitude']
			self.h5_phase = self.h5_results_grp['Phase']
			try:
				self.h5_pwrdis = self.h5_results_grp['PowerDissipation']
			except:
				pass
		return

	def _unit_computation(self, *args, **kwargs):
		"""
		The unit computation that is performed per data chunk. This allows room for any data pre / post-processing
		as well as multiple calls to parallel_compute if necessary
		"""
		# TODO: Try to use the functools.partials to preconfigure the map function
		# cores = number of processes / rank here

		args = [self.parm_dict, self.pixel_params]

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
		pixel_params = args[1]

		pix = Pixel(defl, parm_dict, **pixel_params)

		if parm_dict['if_only']:
			inst_freq, _, _ = pix.generate_inst_freq()
			tfp = 0
			shift = 0
			amplitude = 0
			phase = 0
			pwr_diss = 0
		else:
			tfp, shift, inst_freq = pix.analyze()
			pix.calculate_power_dissipation()
			amplitude = pix.amplitude
			phase = pix.phase
			pwr_diss = pix.power_dissipated

		return [inst_freq, amplitude, phase, tfp, shift, pwr_diss]


def save_CSV_from_file(h5_file, h5_path='/', append='', mirror=False):
	"""
	Saves the tfp, shift, and fixed_tfp as CSV files
	
	Parameters
	----------
	h5_file : H5Py file of FFtrEFM class
		Reminder you can always type: h5_svd.file or h5_avg.file for this
	
	h5_path : str, optional
		specific folder path to search for the tfp data. Usually not needed.
	
	append : str, optional
		text to append to file name
	"""

	h5_ff = h5_file

	if isinstance(h5_file, ffta.hdf_utils.process.FFtrEFM):
		print('Saving from FFtrEFM Class')
		h5_ff = h5_file.h5_main.file
		h5_path = h5_file.h5_results_grp.name

	elif not isinstance(h5_file, h5py.File):
		print('Saving from pyUSID object')
		h5_ff = h5_file.file

	tfp = usid.hdf_utils.find_dataset(h5_ff[h5_path], 'tfp')[0][()]
	# tfp_fixed = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp_fixed')[0][()]
	shift = usid.hdf_utils.find_dataset(h5_ff[h5_path], 'shift')[0][()]

	tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
	tfp_fixed = np.array(tfp_fixed)

	print(usid.hdf_utils.find_dataset(h5_ff[h5_path], 'shift')[0].parent.name)

	path = h5_ff.file.filename.replace('\\', '/')
	path = '/'.join(path.split('/')[:-1]) + '/'
	os.chdir(path)

	if mirror:
		np.savetxt('tfp-' + append + '.csv', np.fliplr(tfp).T, delimiter=',')
		np.savetxt('shift-' + append + '.csv', np.fliplr(shift).T, delimiter=',')
		np.savetxt('tfp_fixed-' + append + '.csv', np.fliplr(tfp_fixed).T, delimiter=',')
	else:
		np.savetxt('tfp-' + append + '.csv', tfp.T, delimiter=',')
		np.savetxt('shift-' + append + '.csv', shift.T, delimiter=',')
		np.savetxt('tfp_fixed-' + append + '.csv', tfp_fixed.T, delimiter=',')

	return


def plot_tfp(ffprocess, scale_tfp=1e6, scale_shift=1, threshold=2, **kwargs):
	'''
	Quickly plots the tfp and shift data. If there's a height image in the h5_file associated
	 with ffprocess, will plot that as well
	
	Parameters
	----------
	ffprocess : FFtrEFM class object (inherits Process)
	
	Returns
	-------
	fig, a : figure and axes objects
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
