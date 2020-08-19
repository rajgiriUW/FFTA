# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:15:36 2018

@author: Raj
"""

import pycroscopy as px
from pycroscopy.processing.fft import FrequencyFilter
import pyUSID as usid
import numpy as np
from scipy import signal as sps

from ffta.load import get_utils

from ffta import pixel
from matplotlib import pyplot as plt

import warnings

'''
For filtering data using the pycroscopy filter command

To set up a filter, you can choose any of the following:
	Harmonic Filter: pick a frequency and bandpass filters that + 2w + 3e etc
	Bandpass Filter: pick a specific frequency and pass that
	Lowpass Filter: pick a frequency and pass all below that
	Noise Filter: pick frequencies to selectively remove (like electrical noise, etc)

# a harmonic filter center of 2000 points long at 100kHz and 2*100 kHz, with a 5000 Hz wide window, at 1 MHz sampling
>>> hbf = px.processing.fft.HarmonicPassFilter(2000, 10e6, 100e3, 5000, 2)
>>> ffta.hdf_utils.filtering.test_filter(h5_main, hbf) #will display the result before applying to the whole dataset
>>> ffta.hdf_utils.filtering.fft_filter(h5_main, hbf)


'''


def test_filter(hdf_file, freq_filts, parameters={}, pixelnum=[0, 0], noise_tolerance=5e-7,
				show_plots=True, check_filter=True):
	"""
	Applies FFT Filter to the file at a specific line and displays the result

	Parameters
	----------
	hdf_file : h5Py file or Nx1 NumPy array (preferred is NumPy array)
		hdf_file to work on, e.g. hdf.file['/FF-raw'] if that's a Dataset
		if ndarray, uses passed or default parameters
		Use ndarray.flatten() to ensure correct dimensions

	freq_filts : list of FrequencyFilter class objects
		Contains the filters to apply to the test signal

	parameters : dict, optional
		Contains parameters in FF-raw file for constructing filters. Automatic if a Dataset/File
		Must contain num_pts and samp_rate to be functional
	
	pixelnum : int, optional
		For extracting a specific pixel to do FFT Filtering on
		
	show_plots : bool, optional
		Turns on FFT plots from Pycroscopy

	noise_tolerance : float 0 to 1
		Amount of noise below which signal is set to 0

	Returns
	-------
	filt_line : numpy.ndarray
		Filtered signal of hdf_file
	
	freq_filts : list
		The filter parameters to be passed to SignalFilter
	
	fig_filt, axes_filt: matplotlib controls
		Only functional if show_plots is on
	"""

	reshape = False
	ftype = str(type(hdf_file))
	if ('h5py' in ftype) or ('Dataset' in ftype):  # hdf file

		parameters = get_utils.get_params(hdf_file)
		hdf_file = get_utils.get_pixel(hdf_file, [pixelnum[0], pixelnum[1]], array_form=True, transpose=False)
		hdf_file = hdf_file.flatten()

	if len(hdf_file.shape) == 2:
		reshape = True
		hdf_file = hdf_file.flatten()

	sh = hdf_file.shape

	# Test filter on a single line:
	filt_line, fig_filt, axes_filt = px.processing.gmode_utils.test_filter(hdf_file,
																		   frequency_filters=freq_filts,
																		   noise_threshold=noise_tolerance,
																		   show_plots=show_plots)

	# If need to reshape
	if reshape:
		filt_line = np.reshape(filt_line, sh)

	# Test filter out in Pixel
	if check_filter:
		plt.figure()
		plt.plot(hdf_file, 'b')
		plt.plot(filt_line, 'k')

		h5_px_filt = pixel.Pixel(filt_line, parameters)
		h5_px_filt.clear_filter_flags()
		h5_px_filt.analyze()
		h5_px_filt.plot(newplot=True)

		h5_px_raw = pixel.Pixel(hdf_file, parameters)
		h5_px_raw.analyze()
		h5_px_raw.plot(newplot=True)

	#    h5_px_raw_unfilt = pixel.Pixel(hdf_file, parameters)
	#    h5_px_raw_unfilt.clear_filter_flags()
	#    h5_px_raw_unfilt.analyze()
	#    h5_px_raw_unfilt.plot(newplot=False,c1='y', c2='c')

	return filt_line, freq_filts, fig_filt, axes_filt


def fft_filter(h5_main, freq_filts, noise_tolerance=5e-7, make_new=False, verbose=False):
	"""
	Stub for applying filter above to the entire FF image set
	
	Parameters
	----------
	h5_main : h5py.Dataset object
		Dataset to work on, e.g. h5_main = px.hdf_utils.getDataSet(hdf.file, 'FF_raw')[0]
	
	freq_filts : list
		List of frequency filters usually generated in test_line above
		
	noise_tolerance : float, optional
		Level below which data are set to 0. Higher values = more noise (more tolerant)
	
	make_new : bool, optional
		Allows for re-filtering the data by creating a new folder
	
	Returns
	-------
	
	h5_filt : Dataset
		Filtered dataset within latest -FFT_Filtering Group
		
	"""

	h5_filt_grp = usid.hdf_utils.check_for_old(h5_main, 'FFT_Filtering')

	if make_new == True or not any(h5_filt_grp):

		sig_filt = px.processing.SignalFilter(h5_main, frequency_filters=freq_filts,
											  noise_threshold=noise_tolerance,
											  write_filtered=True, write_condensed=False,
											  num_pix=1, verbose=verbose, cores=2, max_mem_mb=512)

		h5_filt_grp = sig_filt.compute()

	else:
		print('Taking previously computed results')
		h5_filt = h5_filt_grp[0]['Filtered_Data']

	h5_filt = h5_filt_grp['Filtered_Data']
	usid.hdf_utils.copy_attributes(h5_main.parent, h5_filt)
	usid.hdf_utils.copy_attributes(h5_main.parent, h5_filt.parent)

	return h5_filt


def lowpass(hdf_file, parameters={}, pixelnum=[0, 0], f_cutoff=None):
	'''
	Interfaces to px.pycroscopy.fft.LowPassFilter
	
	:param hdf_file:

	:param parameters:
	
	:param pixelnum:
		See test_filter below

	:param f_cutoff: int
		frequency to cut off. Defaults to 2*drive frequency rounded to nearest 100 kHz

	'''
	hdf_file, num_pts, drive, samp_rate = _get_pixel_for_filtering(hdf_file, parameters, pixelnum)

	if not f_cutoff:
		lpf_cutoff = np.round(drive / 1e5, decimals=0) * 2 * 1e5  # 2times the drive frequency, round up

	lpf = px.processing.fft.LowPassFilter(num_pts, samp_rate, lpf_cutoff)

	return lpf


def bandpass(hdf_file, parameters={}, pixelnum=[0, 0], f_center=None, f_width=10e3, harmonic=None, fir=False):
	'''
	Interfaces to pycroscopy.processing.fft.BandPassFilter
	Note that this is effectively a Harmonic Filter of number_harmonics 1, but with finite impulse response option

	:param hdf_file:
	:param parameters:
	:param pixelnum:
		See test_filter below
	:param f_center: int
		center frequency for the specific band to pass
	:param f_width: int
		width of frequency to pass
	:param harmonic: int
		if specified, sets the band to this specific multiple of the drive frequency
	:param fir: bool
		uses an Finite Impulse Response filter instead of a normal boxcar
	'''

	hdf_file, num_pts, drive, samp_rate = _get_pixel_for_filtering(hdf_file, parameters, pixelnum)

	# default is the 2*w signal (second harmonic for KPFM)
	if not f_center:
		if not harmonic:
			f_center = drive * 2
		else:
			f_center = int(drive * harmonic)

	bpf = px.processing.fft.BandPassFilter(num_pts, samp_rate, f_center, f_width, fir=fir)

	return bpf


def harmonic(hdf_file, parameters={}, pixelnum=[0, 0], first_harm=1, bandwidth=None, num_harmonics=5):
	'''
	Interfaces with px.processing.fft.HarmonicFilter

	Parameters
	----------
	hdf_file, parameters, pixelnum : see comments in test_filter below

	first_harm : int
		The first harmonic based on the drive frequency to use
		For G-KPFM this should be explicitly set to 2

	bandwidth : int
		bandwidth for filtering. For computational purposes this is hard-set to 2500 (2.5 kHz)

	num_harmonics : int
		The number of harmonics to use (omega, 2*omega, 3*omega, etc)
	'''

	hdf_file, num_pts, drive, samp_rate = _get_pixel_for_filtering(hdf_file, parameters, pixelnum)

	if not bandwidth:
		bandwidth = 2500
	elif bandwidth > 2500:
		warnings.warn('Bandwidth of that level might cause errors')
		bandwidth = 2500

	first_harm = drive * first_harm

	hbf = px.processing.fft.HarmonicPassFilter(num_pts, samp_rate, first_harm, bandwidth, num_harmonics)

	return hbf


def noise_filter(hdf_file, parameters={}, pixelnum=[0, 0],
				 centers=[10E3, 50E3, 100E3, 150E3, 200E3],
				 widths=[20E3, 1E3, 1E3, 1E3, 1E3]):
	'''
	Interfaces with pycroscopy.processing.fft.NoiseBandFilter

	:param hdf_file:
	:param parameters:
	:param pixelnum:
		See test_filter
	:param centers: list
		List of Frequencies to filter out
	:param widths:
		List of frequency widths for each filter. e,g. in default case (10 kHz center, 20 kHz width) is from 0 to 20 kHz
	'''

	hdf_file, num_pts, drive, samp_rate = _get_pixel_for_filtering(hdf_file, parameters, pixelnum)

	nf = px.processing.fft.NoiseBandFilter(num_pts, samp_rate, centers, widths)

	return nf


def _get_pixel_for_filtering(hdf_file, parameters={}, pixelnum=[0, 0]):
	ftype = str(type(hdf_file))
	if ('h5py' in ftype) or ('Dataset' in ftype):  # hdf file

		parameters = usid.hdf_utils.get_attributes(hdf_file)
		hdf_file = get_utils.get_pixel(hdf_file, [pixelnum[0], pixelnum[1]], array_form=True, transpose=False)
		hdf_file = hdf_file.flatten()

	if len(hdf_file.shape) == 2:
		hdf_file = hdf_file.flatten()

	num_pts = hdf_file.shape[0]
	drive = parameters['drive_freq']
	samp_rate = parameters['sampling_rate']

	return hdf_file, num_pts, drive, samp_rate


# placeholder until accepted in pull request
class BandPassFilter(FrequencyFilter):
	def __init__(self, signal_length, samp_rate, f_center, f_width,
				 fir=False, fir_taps=1999):
		"""
		Builds a bandpass filter

		Parameters
		----------
		signal_length : unsigned int
			Points in the FFT. Assuming Signal in frequency space (ie - after FFT shifting)
		samp_rate : unsigned integer
			Sampling rate
		f_center : unsigned integer
			Center frequency for filter
		f_width : unsigned integer
			Frequency width of the pass band
		fir : bool, optional
			True uses a finite impulse response (FIR) response instead of a standard boxcar. FIR is causal
		fir_taps : int
			Number of taps (length of filter) for finite impulse response filter

		Returns
		-------
		bpf : 1D numpy array describing the bandpass filter

		"""

		if f_center >= 0.5 * samp_rate:
			raise ValueError('Filter cutoff exceeds Nyquist rate')

		self.f_center = f_center
		self.f_width = f_width

		super(BandPassFilter, self).__init__(signal_length, samp_rate)

		cent = int(round(0.5 * signal_length))

		# very simple boxcar
		ind = int(round(signal_length * (f_center / samp_rate)))
		sz = int(round(cent * f_width / samp_rate))

		bpf = np.zeros(signal_length, dtype=np.float32)

		# Finite Impulse Response or Boxcar
		if not fir:

			bpf[cent - ind - sz:cent - ind + sz + 1] = 1
			bpf[cent + ind - sz:cent + ind + sz + 1] = 1

		else:

			freq_low = (f_center - f_width) / (0.5 * samp_rate)
			freq_high = (f_center + f_width) / (0.5 * samp_rate)

			band = [freq_low, freq_high]

			taps = sps.firwin(int(fir_taps), band, pass_zero=False,
							  window='blackman')
			bpf = np.abs(np.fft.fftshift(np.fft.fft(taps, n=signal_length)))

		self.value = bpf

	def get_parms(self):
		basic_parms = super(BandPassFilter, self).get_parms()
		prefix = 'band_pass_'
		this_parms = {prefix + 'start_freq': self.f_center, prefix + 'band_width': self.f_width}
		this_parms.update(basic_parms)
		return this_parms
