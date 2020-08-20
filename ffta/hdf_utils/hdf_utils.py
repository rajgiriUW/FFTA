# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:28 2018

@author: Raj
"""
import pycroscopy as px
import pyUSID as usid

import warnings
import numpy as np

from ffta.load import get_utils

from pycroscopy.io.write_utils import build_ind_val_dsets, build_ind_val_matrices, Dimension

"""
Common HDF interfacing functions

_which_h5_group : Returns a group corresponding to main FFtrEFM file
get_params : Returns the common parameters list from config file
change_params : Changes specific parameters (replicates normal command line function)
get_line : Gets a specific line, returns as either array or Line class
get_pixel : Gets a specific pixel, returns as either array or Pixel class
h5_list : Gets list of files corresponding to a key, for creating unique folders/Datasets
hdf_commands : Creates workspace-compatible commands for common HDF variable standards
add_standard_sets : Adds the standard data sets needed for much processing

"""


def _which_h5_group(h5_path):
	"""
	Used internally in get_ functions to indentify type of H5_path parameter.
	H5_path can be passed as string (to h5 location), or as an existing
	variable in the workspace.
	
	This tries 
	
	If this is a Dataset, it will try and return the parent as that is 
		by default where all relevant attributs are
	
	Parameters
	----------
	h5_path : str, HDF group, HDF file
	
	Returns
	-------
	h5Py Group
		h5Py group corresponding to "FF_Group" typically at hdf.file['/FF_Group']
	"""
	ftype = str(type(h5_path))

	# h5_path is a file path
	if 'str' in ftype:
		hdf = px.io.HDFwriter(h5_path)
		p = usid.hdf_utils.find_dataset(hdf.file, 'FF_Raw')[0]

		return p.parent

	# h5_path is an HDF Group
	if 'Group' in ftype:
		p = h5_path

	# h5_path is an HDF File
	elif 'File' in ftype:
		p = usid.hdf_utils.find_dataset(h5_path, 'FF_Raw')[0]

		p = p.parent

	elif 'Dataset' in ftype:
		p = h5_path.parent

	return p


def h5_list(h5_file, key):
	'''
	Returns list of names matching a key in the h5 group passed.
	This is useful for creating unique keys in datasets
	This ONLY works on the specific group and is not for searching the HDF5 file folder
	
	e.g. this checks for -processed folder, increments the suffix
	>>    names = hdf_utils.h5_list(hdf.file['/FF_Group'], 'processed')
	>>    try:
	>>        suffix = names[-1][-4:]
	>>        suffix = str(int(suffix)+1).zfill(4)
	
	Parameters
	----------
	h5_file : h5py File
		hdf.file['/Measurement_000/Channel_000'] or similar
		
	key : str
		string to search for, e.g. 'processing'
	'''
	names = []

	for i in h5_file:
		if key in i:
			names.append(i)

	return names


def add_standard_sets(h5_path, group, fast_x=32e-6, slow_y=8e-6,
					  parm_dict={}, ds='FF_Raw', verbose=False):
	"""
	Adds Position_Indices and Position_Value datasets to a folder within the h5_file
	
	Uses the values of fast_x and fast_y to determine the values
	
	Parameters
	----------
	h5_path : h5 File or str 
		Points to a path to process
	
	group : str or H5PY group 
		Location to process data to, either as str or H5PY
		
	parms_dict : dict, optional
		Parameters to be passed. By default this should be at the command line for FFtrEFM data
		
	ds : str, optional
		Dataset name to search for within this group and set as h5_main
		
	verbose : bool, optional
		Whether to write to the command line
	"""

	hdf = px.io.HDFwriter(h5_path)

	if not any(parm_dict):
		parm_dict = get_utils.get_params(h5_path)

	if 'FastScanSize' in parm_dict:
		fast_x = parm_dict['FastScanSize']

	if 'SlowScanSize' in parm_dict:
		slow_y = parm_dict['SlowScanSize']

	try:
		num_rows = parm_dict['num_rows']
		num_cols = parm_dict['num_cols']
		pnts_per_avg = parm_dict['pnts_per_avg']
		dt = 1 / parm_dict['sampling_rate']
	except:  # some defaults
		warnings.warn('Improper parameters specified.')
		num_rows = 64
		num_cols = 128
		pnts_per_avg = 1
		dt = 1

	try:
		grp = px.io.VirtualGroup(group)
	except:
		grp = px.io.VirtualGroup(group.name)

	pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
				Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]
	ds_pos_ind, ds_pos_val = build_ind_val_dsets(pos_desc, is_spectral=False, verbose=verbose)

	spec_desc = [Dimension('Time', 's', np.linspace(0, pnts_per_avg, dt))]
	ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True, verbose=verbose)

	aux_ds_names = ['Position_Indices', 'Position_Values',
					'Spectroscopic_Indices', 'Spectroscopic_Values']

	grp.add_children([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals])

	h5_refs = hdf.write(grp, print_log=verbose)

	h5_main = hdf.file[grp.name]

	if any(ds):
		h5_main = usid.hdf_utils.find_dataset(hdf.file[grp.name], ds)[0]

	try:
		usid.hdf_utils.link_h5_objects_as_attrs(h5_main, usid.hdf_utils.get_h5_obj_refs(aux_ds_names, h5_refs))
	except:
		usid.hdf_utils.link_h5_objects_as_attrs(h5_main, usid.hdf_utils.get_h5_obj_refs(aux_ds_names, h5_refs))

	hdf.flush()

	return h5_main


def add_single_dataset(h5_path, group, dset, dset_name, verbose=False):
	'''
	Adds a single dataset (dset) to group h5_grp in h5_main
	
	Parameters
	----------
	h5_path : h5 File or str 
		Points to a path to process
	
	group : str or H5PY group 
		Location to process data to, either as str or H5PY
		
	dset : ndarray
		Dataset name to search for within this group and set as h5_main
		
	dset_name : str
		Dataset name for the h5 folder
	
	'''

	hdf = px.io.HDFwriter(h5_path)
	h5_file = hdf.file

	if isinstance(group, str):
		grp_tr = px.io.VirtualGroup(group)
		grp_name = group
	else:
		grp_tr = px.io.VirtualGroup(group.name)
		grp_name = group.name

	grp_ds = px.io.VirtualDataset(dset_name, dset, parent=h5_file[grp_name])
	grp_tr.add_children([grp_ds])

	if verbose:
		hdf.write(grp_tr, print_log=True)
		usid.hdf_utils.print_tree(h5_file, rel_paths=True)

	else:
		hdf.write(grp_tr, print_log=False)

	hdf.flush()

	return hdf.file
