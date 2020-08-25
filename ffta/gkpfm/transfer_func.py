# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:10:38 2019

@author: Raj
"""

import pyUSID as usid
from pyUSID.io.write_utils import Dimension

from igor import binarywave as bw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal as sg

import ffta
import time

from pycroscopy.processing.fft import get_noise_floor


def transfer_function(h5_file, tf_file='', params_file='',
					  psd_freq=1e6, offset=0.0016, sample_freq=10e6,
					  plot=False):
	'''
	Reads in the transfer function .ibw, then creates two datasets within
	a parent folder 'Transfer_Function'
	
	This will destructively overwrite an existing Transfer Function in there
	
	1) TF (transfer function)
	2) Freq (frequency axis for computing Fourier Transforms)
	
	Parameters
	----------
	tf_file : ibw
		Transfer Function .ibw File
		
	params_file : string
		The filepath in string format for the parameters file containing
			Q, AMPINVOLS, etc.
	
	psd_freq : float
		The maximum range of the Power Spectral Density.
		For Asylum Thermal Tunes, this is often 1 MHz on MFPs and 2 MHz on Cyphers
		
	offset : float
		To avoid divide-by-zero effects since we will divide by the transfer function
			when generating GKPFM data
			
	sample_freq : float
		The desired output sampling. This should match your data.   
			
	Returns
	-------
	h5_file['Transfer_Function'] : the Transfer Function group
	'''
	if not any(tf_file):
		tf_file = usid.io_utils.file_dialog(caption='Select Transfer Function file ',
											file_filter='IBW Files (*.ibw)')
	data = bw.load(tf_file)
	tf = data.get('wave').get('wData')

	if 'Transfer_Function' in h5_file:
		del h5_file['/Transfer_Function']
	h5_file.create_group('Transfer_Function')
	h5_file['Transfer_Function'].create_dataset('TF', data=tf)

	freq = np.linspace(0, psd_freq, len(tf))
	h5_file['Transfer_Function'].create_dataset('Freq', data=freq)

	parms = params_list(params_file, psd_freq=psd_freq)

	for k in parms:
		h5_file['Transfer_Function'].attrs[k] = float(parms[k])

	tfnorm = float(parms['Q']) * (tf - np.min(tf)) / (np.max(tf) - np.min(tf))
	tfnorm += offset
	h5_file['Transfer_Function'].create_dataset('TFnorm', data=tfnorm)

	TFN_RS, FQ_RS = resample_tf(h5_file, psd_freq=psd_freq, sample_freq=sample_freq)
	TFN_RS = float(parms['Q']) * (TFN_RS - np.min(TFN_RS)) / (np.max(TFN_RS) - np.min(TFN_RS))
	TFN_RS += offset

	h5_file['Transfer_Function'].create_dataset('TFnorm_resampled', data=TFN_RS)
	h5_file['Transfer_Function'].create_dataset('Freq_resampled', data=FQ_RS)

	if plot:
		plt.figure()
		plt.plot(freq, tfnorm, 'b')
		plt.plot(FQ_RS, TFN_RS, 'r')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Amplitude (m)')
		plt.yscale('log')
		plt.title('Transfer Function')

	return h5_file['Transfer_Function']


def resample_tf(h5_file, psd_freq=1e6, sample_freq=10e6):
	'''
	Resamples the Transfer Function based on the desired target frequency
	
	This is important for dividing the transfer function elements together
	
	Parameters
	----------
	psd_freq : float
		The maximum range of the Power Spectral Density.
		For Asylum Thermal Tunes, this is often 1 MHz on MFPs and 2 MHz on Cyphers
		
	sample_freq : float
		The desired output sampling. This should match your data.   
	
	'''
	TFN = h5_file['Transfer_Function/TFnorm'][()]
	# FQ = h5_file['Transfer_Function/Freq'][()]

	# Generate the iFFT from the thermal tune data
	tfn = np.fft.ifft(TFN)
	# tq = np.linspace(0, 1/np.abs(FQ[1] - FQ[0]), len(tfn))

	# Resample
	scale = int(sample_freq / psd_freq)
	print('Rescaling by', scale, 'X')
	tfn_rs = sg.resample(tfn, len(tfn) * scale)  # from 1 MHz to 10 MHz
	TFN_RS = np.fft.fft(tfn_rs)
	FQ_RS = np.linspace(0, sample_freq, len(tfn_rs))

	return TFN_RS, FQ_RS


def params_list(path='', psd_freq=1e6, lift=50):
	'''
	Reads in a Parameters file as saved in Igor as a dictionary
	
	For use in creating attributes of transfer Function
	
	'''
	if not any(path):
		path = usid.io.io_utils.file_dialog(caption='Select Parameters Files ',
											file_filter='Text (*.txt)')

	df = pd.read_csv(path, sep='\t', header=1)
	df = df.set_index(df['Unnamed: 0'])
	df = df.drop(columns='Unnamed: 0')

	parm_dict = df.to_dict()['Initial']
	parm_dict['PSDFreq'] = psd_freq
	parm_dict['Lift'] = lift

	return parm_dict


def test_Ycalc(h5_main, pixel_ind=[0, 0], transfer_func=None, resampled=True, ratios=None,
			   verbose=True, noise_floor=1e-3, phase=-np.pi, plot=False, scaling=1):
	'''
	Divides the response by the transfer function
	
	Parameters
	----------
	h5_main : h5py dataset of USIDataset
	
	tf : transfer function, optional
		This can be the resampled or normal transfer function
		For best results, use the "normalized" transfer function
		"None" will default to /Transfer_Function folder
		
	resampled : bool, optional
		Whether to use the upsampled Transfer Function or the original
		
	verbose: bool, optional
		Gives user feedback during processing
		
	noise_floor : float, optional
		For calculating what values to filter as the noise floor of the data
		0 or None circumvents this
		
	phase : float, optional
		Practically any value between -pi and +pi works
		
	scaling : int, optional
		scales the transfer function by this number if, for example, the TF was
		acquired on a line and you're dividing by a point (or vice versa)'
	
		'''
	t0 = time.time()

	parm_dict = usid.hdf_utils.get_attributes(h5_main)
	drive_freq = parm_dict['drive_freq']
	response = ffta.hdf_utils.get_utils.get_pixel(h5_main, pixel_ind, array_form=True, transpose=False).flatten()

	response -= np.mean(response)
	RESP = np.fft.fft(response)
	Yout = np.zeros(len(RESP), dtype=complex)

	# Create frequency axis for the pixel
	samp = parm_dict['sampling_rate']
	fq_y = np.linspace(0, samp, len(Yout))

	noise_limit = np.ceil(get_noise_floor(RESP, noise_floor)[0])

	# Get the transfer function and transfer function frequency values
	fq_tf = h5_main.file['Transfer_Function/Freq'][()]
	if not transfer_func:
		if resampled:

			transfer_func = h5_main.file['Transfer_Function/TFnorm_resampled'][()]
			fq_tf = h5_main.file['Transfer_Function/Freq_resampled'][()]

		else:

			transfer_func = h5_main.file['Transfer_Function/TFnorm'][()]

	if verbose:
		t1 = time.time()
		print('Time for pixels:', t1 - t0)

	Yout_divided = np.zeros(len(RESP), dtype=bool)

	TFratios = np.ones(len(RESP))

	# Calculate the TF scaled to the sample size of response function
	for x, f in zip(transfer_func, fq_tf):

		if np.abs(x) > noise_floor:

			xx = np.searchsorted(fq_y, f)
			if not Yout_divided[xx]:
				TFratios[xx] = x
				TFratios[-xx] = x

				Yout_divided[xx] = True

	signal_bins = np.arange(len(TFratios))
	signal_kill = np.where(np.abs(RESP) < noise_limit)
	pass_frequencies = np.delete(signal_bins, signal_kill)

	drive_bin = (np.abs(fq_y - drive_freq)).argmin()
	RESP_ph = (RESP) * np.exp(-1j * fq_y / (fq_y[drive_bin]) * phase)

	# Step 3C)  iFFT the response above a user defined noise floor to recover Force in time domain.
	Yout[pass_frequencies] = RESP_ph[pass_frequencies]
	Yout = Yout / (TFratios * scaling)
	yout = np.real(np.fft.ifft(np.fft.ifftshift(Yout)))

	if verbose:
		t2 = time.time()
		print('Time for pixels:', t2 - t1)

	if plot:
		fig, ax = plt.subplots(figsize=(12, 7))
		ax.semilogy(fq_y, np.abs(Yout), 'b', label='F3R')
		ax.semilogy(fq_y[signal_bins], np.abs(Yout[signal_bins]), 'og', label='F3R')
		ax.semilogy(fq_y[signal_bins], np.abs(RESP[signal_bins]), '.r', label='Response')
		ax.set_xlabel('Frequency (kHz)', fontsize=16)
		ax.set_ylabel('Amplitude (a.u.)', fontsize=16)
		ax.legend(fontsize=14)
		ax.set_yscale('log')
		ax.set_xlim(0, 3 * drive_freq)
		ax.set_title('Noise Spectrum', fontsize=16)

	return TFratios, Yout, yout


def Y_calc(h5_main, transfer_func=None, resampled=True, ratios=None, verbose=False,
		   noise_floor=1e-3, phase=-np.pi, plot=False, scaling=1):
	'''
	Divides the response by the transfer function
	
	Parameters
	----------
	h5_main : h5py dataset of USIDataset
	
	tf : transfer function, optional
		This can be supplied or use the calculated version
		For best results, use the "normalized" transfer function
		"None" will default to /Transfer_Function folder
		
	resampled : bool, optional
		Whether to use the upsampled Transfer Function or the original
		
	verbose: bool, optional
		Gives user feedback during processing
	
	ratios : array, optional
		Array of the size of h5_main (1-D) with the transfer function data
		If not given, it's found via the test_Y_calc function
	
	noise_floor : float, optional
		For calculating what values to filter as the noise floor of the data
		0 or None circumvents this
		
	phase : float, optional
		Practically any value between -pi and +pi works
		
	scaling : int, optional
		scales the transfer function by this number if, for example, the TF was
		acquired on a line and you're dividing by a point (or vice versa)'
	
	'''
	parm_dict = usid.hdf_utils.get_attributes(h5_main)
	drive_freq = parm_dict['drive_freq']

	ds = h5_main[()]
	Yout = np.zeros(ds.shape, dtype=complex)
	yout = np.zeros(ds.shape)

	# Create frequency axis for the pixel
	samp = parm_dict['sampling_rate']
	fq_y = np.linspace(0, samp, Yout.shape[1])

	response = ds[0, :]
	response -= np.mean(response)
	RESP = np.fft.fft(response)
	noise_limit = np.ceil(get_noise_floor(RESP, noise_floor)[0])

	# Get the transfer function and transfer function frequency values
	# Use test calc to scale the transfer function to the correct size
	if not transfer_func:
		if resampled:

			transfer_func, _, _ = test_Ycalc(h5_main, resampled=True,
											 verbose=verbose, noise_floor=noise_floor)
		else:

			transfer_func, _, _ = test_Ycalc(h5_main, resampled=False,
											 verbose=verbose, noise_floor=noise_floor)

	import time
	t0 = time.time()

	signal_bins = np.arange(len(transfer_func))
	for c in np.arange(h5_main.shape[0]):

		if verbose:
			if c % 100 == 0:
				print('Pixel:', c)

		response = ds[c, :]
		response -= np.mean(response)
		RESP = np.fft.fft(response)

		signal_kill = np.where(np.abs(RESP) < noise_limit)
		pass_frequencies = np.delete(signal_bins, signal_kill)

		drive_bin = (np.abs(fq_y - drive_freq)).argmin()
		RESP_ph = (RESP) * np.exp(-1j * fq_y / (fq_y[drive_bin]) * phase)

		Yout[c, pass_frequencies] = RESP_ph[pass_frequencies]
		Yout[c, :] = Yout[c, :] / (transfer_func * scaling)
		yout[c, :] = np.real(np.fft.ifft(Yout[c, :]))

	t1 = time.time()

	print('Time for pixels:', t1 - t0)

	return Yout, yout


def check_phase(h5_main, transfer_func, phase_list=[-np.pi, -np.pi / 2, 0],
				plot=True, noise_tolerance=1e-6, samp_rate=10e6):
	'''
	Uses the list of phases in phase_list to plot the various phase offsets
	relative to the driving excitation
	'''
	ph = -3.492  # phase from cable delays between excitation and response
	row_ind = 0

	test_row = np.fft.fftshift(np.fft.fft(h5_main[row_ind]))
	noise_floor = get_noise_floor(test_row, noise_tolerance)[0]
	print('Noise floor = ', noise_floor)
	Noiselimit = np.ceil(noise_floor)

	parm_dict = usid.hdf_utils.get_attributes(h5_main)
	drive_freq = parm_dict['drive_freq']

	freq = np.arange(-samp_rate / 2, samp_rate / 2, samp_rate / len(test_row))
	tx = np.arange(0, parm_dict['total_time'], parm_dict['total_time'] / len(freq))

	exc_params = {'ac': 1, 'dc': 0, 'phase': 0, 'frequency': drive_freq}
	exc_params['ac']
	excitation = (exc_params['ac'] * np.sin(tx * 2 * np.pi * exc_params['frequency'] \
											+ exc_params['phase']) + exc_params['dc'])

	for ph in phase_list:
		# Try Force Conversion on Filtered data of single line (row_ind above)
		G_line = np.zeros(freq.size, dtype=complex)  # G = raw
		G_wPhase_line = np.zeros(freq.size, dtype=complex)  # G_wphase = phase-shifted

		signal_ind_vec = np.arange(freq.size)
		ind_drive = (np.abs(freq - drive_freq)).argmin()

		# filt_line is from filtered data above
		test_line = test_row - np.mean(test_row)
		test_line = np.fft.fftshift(np.fft.fft(test_line))
		signal_kill = np.where(np.abs(test_line) < Noiselimit)
		signal_ind_vec = np.delete(signal_ind_vec, signal_kill)

		# Original/raw data; TF_norm is from the Tune file transfer function
		G_line[signal_ind_vec] = test_line[signal_ind_vec]
		G_line = (G_line / transfer_func)
		G_time_line = np.real(np.fft.ifft(np.fft.ifftshift(G_line)))  # time-domain

		# Phase-shifted data
		test_shifted = (test_line) * np.exp(-1j * freq / (freq[ind_drive]) * ph)
		G_wPhase_line[signal_ind_vec] = test_shifted[signal_ind_vec]
		G_wPhase_line = (G_wPhase_line / transfer_func)
		G_wPhase_time_line = np.real(np.fft.ifft(np.fft.ifftshift(G_wPhase_line)))

		phaseshifted = np.reshape(G_wPhase_time_line, (parm_dict['num_cols'], parm_dict['num_rows']))
		fig, axes = usid.plot_utils.plot_curves(excitation, phaseshifted, use_rainbow_plots=True,
												x_label='Voltage (Vac)', title='Phase Shifted',
												num_plots=4, y_label='Deflection (a.u.)')
		axes[0][0].set_title('Phase ' + str(ph))

	return


def save_Yout(h5_main, Yout, yout):
	'''
	Writes the results to teh HDF5 file
	'''
	parm_dict = usid.hdf_utils.get_attributes(h5_main)

	# Get relevant parameters
	num_rows = parm_dict['num_rows']
	num_cols = parm_dict['num_cols']
	pnts_per_avg = parm_dict['pnts_per_avg']

	h5_gp = h5_main.parent
	h5_meas_group = usid.hdf_utils.create_indexed_group(h5_gp, 'GKPFM_Frequency')

	# Create dimensions
	pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
				Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

	# ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
	spec_desc = [Dimension('Frequency', 'Hz', np.linspace(0, parm_dict['sampling_rate'], pnts_per_avg))]
	# ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)

	# Writes main dataset
	h5_y = usid.hdf_utils.write_main_dataset(h5_meas_group,
											 Yout,
											 'Y',  # Name of main dataset
											 'Deflection',  # Physical quantity contained in Main dataset
											 'V',  # Units for the physical quantity
											 pos_desc,  # Position dimensions
											 spec_desc,  # Spectroscopic dimensions
											 dtype=np.cdouble,  # data type / precision
											 main_dset_attrs=parm_dict)

	usid.hdf_utils.copy_attributes(h5_y, h5_gp)

	h5_meas_group = usid.hdf_utils.create_indexed_group(h5_gp, 'GKPFM_Time')
	spec_desc = [Dimension('Time', 's', np.linspace(0, parm_dict['total_time'], pnts_per_avg))]
	h5_y = usid.hdf_utils.write_main_dataset(h5_meas_group,
											 yout,
											 'y_time',  # Name of main dataset
											 'Deflection',  # Physical quantity contained in Main dataset
											 'V',  # Units for the physical quantity
											 pos_desc,  # Position dimensions
											 spec_desc,  # Spectroscopic dimensions
											 dtype=np.float32,  # data type / precision
											 main_dset_attrs=parm_dict)

	usid.hdf_utils.copy_attributes(h5_y, h5_gp)

	h5_y.file.flush()

	return


def check_response(h5_main, pixel=0, ph=0):
	parm_dict = usid.hdf_utils.get_attributes(h5_main)
	freq = parm_dict['drive_freq']
	txl = np.linspace(0, parm_dict['total_time'], h5_main[pixel, :].shape[0])

	resp_wfm = np.sin(txl * 2 * np.pi * freq + ph)

	plt.figure()
	plt.plot(resp_wfm, h5_main[()][pixel, :])

	return
