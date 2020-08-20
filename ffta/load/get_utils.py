# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 13:40:54 2018

@author: Raj
"""
import pycroscopy as px
import pyUSID as usid

from ffta.line import Line
from ffta.pixel import Pixel

import numpy as np

'''
Functions for extracting certain segments from an HDF FFtrEFM file
'''


def get_params(h5_path, key='', verbose=False, del_indices=True):
	"""
	Gets dict of parameters from the FF-file
	Returns a specific key-value if requested
	
	Parameters
	----------
	h5_path : str or h5py
		Can pass either an h5_path to a file or a file already in use
	
	key : str, optional
		Returns specific key in the parameters dictionary
		
	verbose : bool, optional
		Prints all parameters to console
		
	del_indices : bool, optional
		Deletes relative links within the H5Py and any quantity/units
	"""

	if isinstance(h5_path, str):
		h5_path = px.io.HDFwriter(h5_path).file

	parameters = usid.hdf_utils.get_attributes(h5_path)

	# if this dataset does not have complete FFtrEFM parameters
	if 'trigger' not in parameters:
		parameters = usid.hdf_utils.get_attributes(h5_path.parent)

	# still not there? hard-select the main dataset
	if 'trigger' not in parameters:

		try:
			h5_file = px.io.HDFwriter(h5_path).file
			parameters = usid.hdf_utils.get_attributes(h5_file['FF_Group'])

		except:
			raise TypeError('No proper parameters file found.')

	# STILL not there? Find the FF AVg
	if 'trigger' not in parameters:

		try:
			h5_file = px.io.HDFwriter(h5_path).file
			parameters = usid.hdf_utils.get_attributes(h5_file['FF_Group/FF_Avg'])

		except:
			raise TypeError('No proper parameters file found.')

	if any(key):
		return parameters[key]

	if verbose:
		print(parameters)

	del_keys = ['Position_Indices', 'Position_Values', 'Spectroscopic_Indices',
				'Spectroscopic_Values', 'quantity', 'units']

	for key in del_keys:
		if key in parameters:
			del parameters[key]

	return parameters


def change_params(h5_path, new_vals={}, verbose=False):
	"""
	Changes a parameter to a new value
	
	This is equivalent to h5_main.parent.attrs[key] = new_value
	
	Parameters
	----------
	h5_path : str or h5py
		Can pass either an h5_path to a file or a file already in use
	
	key : dict, optional
		Returns specific keys in the parameters dictionary
	
	value : str, int, float
		The new value for the key. There is no error checking for this
	
	verbose : bool
		Prints all parameters to console
	
	"""
	parameters = usid.hdf_utils.get_attributes(h5_path)

	if verbose:
		print('Old parameters:')
		for key in new_vals:
			print(key, ':', parameters[key])

	for key in new_vals:
		h5_path.attrs[key] = new_vals[key]

	if verbose:
		print('\nNew parameters:')
		for key in new_vals:
			print(key, ':', parameters[key])

	return parameters


def get_line(h5_path, line_num, params={},
			 array_form=False, avg=False, transpose=False):
	"""
	Gets a line of data.
	
	If h5_path is a dataset it processes based on user-defined pnts_avg
	If h5_path is a group/file/string_path then it can be a Line class or array
	
	Parameters
	----------
	h5_path : str or h5py or Dataset
		Can pass either an h5_path to a file or a file already in use or a specific Dataset
	
	line_num : int
		Returns specific line in the dataset
	
	params: dict
		If explicitly changing parameters (to test a feature), you can pass any subset and this will overwrite it
		e.g. parameters  = {'drive_freq': 10} will extract the Line, then change Line.drive_freq = 10
	
	array_form : bool, optional
		Returns the raw array contents rather than Line class
		
	avg : bool, optional
		Averages the pixels of the entire line 
		This will force output to be an array
		
	transpose : bool, optional
		For legacy FFtrEFM code, pixel is required in (n_points, n_signals) format
		
	Returns
	-------
	signal_line : numpy 2D array, iff array_form == True
		ndarray containing the line 
		
	line_inst : Line, iff array_form == False
		Line class containing the signal_array object and parameters
	"""

	# If not a dataset, then find the associated Group
	if 'Dataset' not in str(type(h5_path)):

		parameters = get_params(h5_path)
		h5_file = px.io.HDFwriter(h5_path).file

		d = usid.hdf_utils.find_dataset(h5_file, 'FF_Raw')[0]
		c = parameters['num_cols']
		pnts = parameters['pnts_per_line']

	else:  # if a Dataset, extract parameters from the shape.

		d = h5_path[()]
		parameters = get_params(h5_path)

		c = parameters['num_cols']
		pnts = parameters['pnts_per_line']

	signal_line = d[line_num * pnts:(line_num + 1) * pnts, :]

	if avg == True:
		signal_line = (signal_line.transpose() - signal_line.mean(axis=1)).transpose()
		signal_line = signal_line.mean(axis=0)

	if transpose == True:
		signal_line = signal_line.transpose()

	if array_form == True or avg == True:
		return signal_line

	if any(params):
		for key, val in params.items():
			parameters[key] = val

	line_inst = Line(signal_line, parameters, c, pycroscopy=True)

	return line_inst


def get_pixel(h5_path, rc, params={}, pixel_params={},
			  array_form=False, avg=False, transpose=False):
	"""
	Gets a pixel of data, returns all the averages within that pixel
	Returns a specific key if requested
	Supplying a direct link to a Dataset is MUCH faster than just the file
	Note that you should supply the deflection data, not instantaneous_frequency
	
	Parameters
	----------
	h5_path : str or h5py or Dataset
		Can pass either an h5_path to a file or a file already in use or specific Dataset
		Again, should pass the deflection data (Rebuilt_Data, or FF_Avg)
	
	rc : list [r, c]
		Pixel location in terms of ROW, COLUMN

	params: dict
		If explicitly changing parameters (to test a feature), you can pass any subset and this will overwrite it
		e.g. parameters  = {'drive_freq': 10} will extract the Pixel, then change Pixel.drive_freq = 10

	pixel_params : dict, optional
		Parameters 'fit', 'pycroscopy', 'method', 'fit_form'
		See ffta.pixel for details. 'pycroscopy' is set to True in this function

	array_form : bool, optional
		Returns the raw array contents rather than Pixel class
	
	avg : bool, optional
		Averages the pixels of n_pnts_per_pixel and then creates Pixel of that
		
	transpose : bool, optional
		For legacy FFtrEFM code, pixel is required in (n_points, n_signals) format
		
	Returns
	-------
	signal_pixel : 1D numpy array, iff array_form == True
		Ndarray of the (n_points) in specific pixel    
	
	pixel_inst : Pixel, iff array_form == False
		Line class containing the signal_array object and parameters
	"""

	# If not a dataset, then find the associated Group
	if 'Dataset' not in str(type(h5_path)):
		p = get_params(h5_path)
		h5_file = px.io.HDFwriter(h5_path).file

		d = usid.hdf_utils.find_dataset(h5_file, 'FF_Raw')[0]
		c = p['num_cols']
		pnts = int(p['pnts_per_pixel'])
		parameters = usid.hdf_utils.get_attributes(d.parent)

	else:

		d = h5_path[()]
		parameters = get_params(h5_path)

		c = parameters['num_cols']
		pnts = parameters['pnts_per_pixel']

	signal_pixel = d[rc[0] * c + rc[1]:rc[0] * c + rc[1] + pnts, :]

	if avg == True:
		signal_pixel = signal_pixel.mean(axis=0)

	if transpose == True:  # this does nothing is avg==True
		signal_pixel = signal_pixel.transpose()

	if array_form == True:
		return signal_pixel

	if signal_pixel.shape[0] == 1:
		signal_pixel = np.reshape(signal_pixel, [signal_pixel.shape[1]])

	if any(params):
		for key, val in params.items():
			parameters[key] = val

	pixel_params.update({'pycroscopy': True})  # must be True in this specific case

	# pixel_inst = Pixel(signal_pixel, parameters, pycroscopy=True)
	pixel_inst = Pixel(signal_pixel, parameters, **pixel_params)

	return pixel_inst
