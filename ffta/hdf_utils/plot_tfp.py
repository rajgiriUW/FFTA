# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:05:42 2020

@author: Raj
"""

from matplotlib import pyplot as plt
# from ffta.hdf_utils import get_utils
from ffta.load import get_utils
from ffta.pixel_utils import badpixels
import pyUSID as usid
import numpy as np
import os


def plot_tfps(h5_file, h5_path='/', append='', savefig=True, stdevs=2):
	"""
	Plots the relevant tfp, inst_freq, and shift values as separate image files
	
	Parameters
	----------
	h5_file : h5Py File
	
	h5_path : str, optional
		Location of the relevant datasets to be saved/plotted. e.g. h5_rb.name
	
	append : str, optional
		A string to include in the saved figure filename
		
	savefig : bool, optional
		Whether or not to save the image
		
	stdevs : int, optional
		Number of standard deviations to display
	"""

	try:
		h5_if = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Inst_Freq')[0]
	except:
		h5_if = usid.hdf_utils.find_dataset(h5_file[h5_path], 'inst_freq')[0]

	parm_dict = get_utils.get_params(h5_if)
	#  parm_dict = usid.hdf_utils.get_attributes(h5_file[h5_path])

	if 'Dataset' in str(type(h5_file[h5_path])):
		h5_path = h5_file[h5_path].parent.name

	tfp = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp')[0][()]
	shift = usid.hdf_utils.find_dataset(h5_file[h5_path], 'shift')[0][()]

	if tfp.shape[1] == 1:
		# Forgot to reconvert and/or needs to be converted
		[ysz, xsz] = h5_if.pos_dim_sizes
		tfp = np.reshape(tfp, [ysz, xsz])
		shift = np.reshape(shift, [ysz, xsz])

	try:
		tfp_fixed = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp_fixed')[0][()]
	except:
		tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
		tfp_fixed = np.array(tfp_fixed)

	xs = parm_dict['FastScanSize']
	ys = parm_dict['SlowScanSize']
	asp = ys / xs
	if asp != 1:
		asp = asp * 2

	fig, a = plt.subplots(nrows=3, figsize=(8, 9))

	[vmint, vmaxt] = np.mean(tfp) - 2 * np.std(tfp), np.mean(tfp) - 2 * np.std(tfp)
	[vmins, vmaxs] = np.mean(shift) - 2 * np.std(shift), np.mean(shift) - 2 * np.std(shift)

	_, cbar_t = usid.viz.plot_utils.plot_map(a[0], tfp_fixed * 1e6, x_vec=xs * 1e6, y_vec=ys * 1e6,
											 aspect=asp, cmap='inferno', stdevs=stdevs)
	_, cbar_r = usid.viz.plot_utils.plot_map(a[1], 1 / (1e3 * tfp_fixed), x_vec=xs * 1e6, y_vec=ys * 1e6,
											 aspect=asp, cmap='inferno', stdevs=stdevs)
	_, cbar_s = usid.viz.plot_utils.plot_map(a[2], shift, x_vec=xs * 1e6, y_vec=ys * 1e6,
											 aspect=asp, cmap='inferno', stdevs=stdevs)

	cbar_t.set_label('tfp (us)', rotation=270, labelpad=16)
	a[0].set_title('tfp', fontsize=12)

	cbar_r.set_label('Rate (kHz)', rotation=270, labelpad=16)
	a[1].set_title('1/tfp', fontsize=12)

	cbar_s.set_label('shift (Hz)', rotation=270, labelpad=16)
	a[2].set_title('shift', fontsize=12)

	fig.tight_layout()

	if savefig:
		path = h5_file.file.filename.replace('\\', '/')
		path = '/'.join(path.split('/')[:-1]) + '/'
		os.chdir(path)
		fig.savefig('tfp_shift_' + append + '_.tif', format='tiff')

	return fig, a
