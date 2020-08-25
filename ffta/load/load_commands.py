# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:53:47 2020

@author: Raj
"""
from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import h5py
import pyUSID as usid
from ffta.load import get_utils


def hdf_commands(h5_path, ds='FF_Raw'):
	"""
	Creates a bunch of typical workspace HDF5 variables for scripting use

	Parameters
	----------
	h5_path : str
		String path to H5PY file

	ds : str, optional
		The dataset to search for and set as h5_main.

	This prints the valid commands to the workspace. Then just highlight and
		copy-paste to execute

	h5_path : str
		Path to hdf5 file on disk
	"""

	commands = ['***Copy-paste all commands below this line, then hit ENTER***',
				'import h5py']

	if not isinstance(h5_path, str):
		raise TypeError('Pass a file path (string), not an H5 file')

	try:
		hdf = h5py.File(h5_path, 'r+')
		commands.append("hdf = h5py.File(h5_path, 'r+')")
	except:
		pass

	try:
		h5_file = hdf.file
		commands.append("h5_file = hdf.file")
	except:
		pass

	try:
		h5_main = usid.hdf_utils.find_dataset(hdf.file, ds)[0]
		commands.append("h5_main = usid.hdf_utils.find_dataset(hdf.file, '" + ds + "')[0]")
	except:
		pass

	try:
		h5_if = usid.hdf_utils.find_dataset(hdf.file, 'inst_freq')[-1]
		commands.append("h5_if = usid.hdf_utils.find_dataset(hdf.file, 'inst_freq')[-1]")
	except:
		pass

	try:
		h5_if = usid.hdf_utils.find_dataset(hdf.file, 'Inst_Freq')[-1]
		commands.append("h5_if = usid.hdf_utils.find_dataset(hdf.file, 'Inst_Freq')[-1]")
	except:
		pass

	try:
		h5_tfp = usid.hdf_utils.find_dataset(hdf.file, 'tfp')[-1]
		commands.append("h5_tfp= usid.hdf_utils.find_dataset(hdf.file, 'tfp')[-1]")
	except:
		pass

	try:
		h5_shift = usid.hdf_utils.find_dataset(hdf.file, 'shift')[-1]
		commands.append("h5_shift= usid.hdf_utils.find_dataset(hdf.file, 'shift')[-1]")
	except:
		pass

	try:
		h5_avg = usid.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[-1]
		commands.append("h5_avg = usid.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[-1]")
	except:
		pass

	try:
		h5_filt = usid.hdf_utils.find_dataset(hdf.file, 'Filtered_Data')[-1]
		commands.append("h5_filt = usid.hdf_utils.find_dataset(hdf.file, 'Filtered_Data')[-1]")
	except:
		pass

	try:
		h5_rb = usid.hdf_utils.find_dataset(hdf.file, 'Rebuilt_Data')[-1]
		commands.append("h5_rb = usid.hdf_utils.find_dataset(hdf.file, 'Rebuilt_Data')[-1]")
	except:
		pass

	try:
		parameters = usid.hdf_utils.get_attributes(h5_avg)
		commands.append("parameters = usid.hdf_utils.get_attributes(h5_avg)")
	except:
		pass

	try:
		h5_ll = get_utils.get_line(h5_if, line_num=0)
		commands.append("h5_ll = ffta.load.get_utils.get_line(h5_if, line_num=0)")
	except:
		pass

	try:
		h5_px = get_utils.get_pixel(h5_if, rc=[0, 0])
		commands.append("h5_px = ffta.load.get_utils.get_pixel(h5_if, rc=[0,0])")
	except:
		pass

	try:
		h5_svd = usid.hdf_utils.find_dataset(hdf.file, 'U')[-1]
		commands.append("h5_svd = usid.hdf_utils.find_dataset(hdf.file, 'U')[-1].parent")
	except:
		pass

	try:
		h5_cpd = usid.hdf_utils.find_dataset(hdf.file, 'cpd')[-1]
		commands.append("h5_cpd = usid.hdf_utils.find_dataset(hdf.file, 'cpd')[-1]")
	except:
		pass

	try:
		h5_ytime = usid.hdf_utils.find_dataset(hdf.file, 'y_time')[-1]
		commands.append("h5_ytime = usid.hdf_utils.find_dataset(hdf.file, 'y_time')[-1]")
	except:
		pass

	try:
		h5_Y = usid.hdf_utils.find_dataset(hdf.file, 'Y')[-1]
		commands.append("h5_Y = usid.hdf_utils.find_dataset(hdf.file, 'Y')[-1]")
	except:
		pass

	for i in commands:
		print(i)

	return
