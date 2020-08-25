"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2020"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import logging
import numpy as np
from scipy import signal as sps
from scipy import optimize as spo
from scipy import interpolate as spi
from scipy import integrate as spg

from ffta.pixel_utils import noise
from ffta.pixel_utils import parab
from ffta.pixel_utils import fitting
from ffta.pixel_utils import dwavelet
from ffta.pixel_utils import tfp_calc

from matplotlib import pyplot as plt

import pywt

import time


class Pixel:
	"""
	Signal Processing to Extract Time-to-First-Peak.

	Extracts Time-to-First-Peak (tFP) from digitized Fast-Free Time-Resolved
	Electrostatic Force Microscopy (FF-trEFM) signals [1-2]. It includes a few
	types of frequency analysis:

	a) Hilbert Transform
	b) Wavelet Transform
	c) Hilbert-Huang Transform (EMD)

	signal_array : (n_points, n_signals) array_like
		2D real-valued signal array, corresponds to a pixel.
	params : dict
		Includes parameters for processing. The list of parameters is:

		trigger = float (in seconds)
		total_time = float (in seconds)
		sampling_rate = int (in Hz)
		drive_freq = float (in Hz)

		roi = float (in seconds)
		window = string (see documentation of scipy.signal.get_window)
		bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
		filter_bandwidth = float (default: 5kHz)
		n_taps = integer (default: 1799)
		wavelet_analysis = bool (0: Hilbert method, 1: Wavelet Method)
		wavelet_parameter = int (default: 5)
		recombination = bool (0: Data are for Charging up, 1: Recombination)
		fit_phase = bool (0: fit to frequency, 1: fit to phase)
	can_params : dict, optional
		Contains the cantilever parameters (e.g. AMPINVOLS).
		see ffta.pixel_utils.load.cantilever_params
	fit : bool, optional
		Find tFP by just raw minimum (False) or fitting product of 2 exponentials (True)
	pycroscopy : bool, optional
		Pycroscopy requires different orientation, so this corrects for this effect.
	fit_form : str, optional
		Functional form used when fitting. 
		
		One of 
			product: product of two exponentials (default)
			sum: sum of two exponentials
			exp: single expential decay
			ringdown: single exponential decay of amplitude, not frequency, scaled to return Q
	
	method : str, optional
		Method for generating instantaneous frequency, amplitude, and phase response
		
		One of

			hilbert: Hilbert transform method (default)
			wavelet: Morlet CWT approach
			stft: short time Fourier transform (sliding FFT)
	
	filter_amplitude : bool, optional
		The Hilbert Transform amplitude can sometimes have drive frequency artifact.
	filter_frequency : bool, optional
		Filters the instantaneous frequency to remove noise peaks

	Attributes
	----------
	n_points : int
		Number of points in a signal.
	n_signals : int
		Number of signals to be averaged in a pixel.
	signal_array : (n_signals, n_points) array_like
		Array that contains original signals.
	signal : (n_points,) array_like
		Signal after phase-locking and averaging.
	tidx : int
		Index of trigger in time-domain.
	phase : (n_points,) array_like
		Phase of the signal, only calculated with Hilbert Transform method.
	cwt_matrix : (n_widths, n_points) array_like
		Wavelet matrix for continuous wavelet transform.
	inst_freq : (n_points,) array_like
		Instantenous frequency of the signal.
	tfp : float
		Time from trigger to first-peak, in seconds.
	shift : float
		Frequency shift from trigger to first-peak, in Hz.

	Methods
	-------
	analyze()
		Analyzes signals and returns tfp, shift and inst_freq.

	Notes
	-----
	Frequency shift from wavelet analysis is not in Hertz. It should be used
	with caution.

	analyze() does not raise an exception if there is one, however it logs the
	exception if logging is turned on. This is implemented this way to avoid
	crashing in the case of exception when this method is called from C API.

	References
	----------
	.. \[1\] Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time
	   resolution atomic force microscopy for probing nanoscale dynamics.
	   Nano Lett. 2012;12(2):893-8.
	   \[2\] Karatay D, Harrison JA, et al. Fast time-resolved electrostatic
	   force microscopy: Achieving sub-cycle time resolution. Rev Sci Inst.
	   2016;87(5):053702

	Examples
	--------
	>>> from ffta import pixel, pixel_utils
	>>>
	>>> signal_file = '../data/SW_0000.ibw'
	>>> params_file = '../data/parameters.cfg'
	>>>
	>>> signal_array = pixel_utils.load.signal(signal_file)
	>>> n_pixels, params = pixel_utils.load.configuration(params_file)
	>>>
	>>> p = pixel.Pixel(signal_array, params)
	>>> tfp, shift, inst_freq = p.analyze()
	>>>
	>>> p.plot()

	"""

	def __init__(self, signal_array, params, can_params=None,
				 fit=True, pycroscopy=False,
				 method='hilbert', fit_form='product', filter_amplitude=False,
				 filter_frequency=False):

		# Create parameter attributes for optional parameters.
		# These defaults are overwritten by values in 'params'

		# FIR (Hilbert) filtering parameters
		if can_params is None:
			can_params = {}
		self.n_taps = 1499
		self.filter_bandwidth = 5000
		self.filter_frequency = filter_frequency

		# Wavelet parameters
		self.wavelet_analysis = False
		self.wavelet = 'cmor1-1'  # default complex Morlet wavelet
		self.scales = np.arange(100, 2, -1)
		self.wavelet_params = {}  # currently just optimize flag is supported

		# Short Time Fourier Transform
		self.fft_analysis = False
		self.fft_cycles = 2
		self.fft_params = {}  # for STFT

		self.recombination = False
		self.phase_fitting = False
		self.check_drive = True

		# Assign the fit parameter.
		self.fit = fit
		self.fit_form = fit_form
		self.method = method
		self.filter_amplitude = filter_amplitude

		# Default Cantilever parameters, plugging in some reasonable defaults
		self.AMPINVOLS = 122e-9
		self.SpringConstant = 23.2
		self.k = self.SpringConstant
		self.DriveAmplitude = 1.7e-9
		self.Mass = 4.55e-12
		self.Beta = 3114
		self.Q = 360

		# Read parameter attributes from parameters dictionary.
		for key, value in params.items():
			setattr(self, key, value)

		for key, value in can_params.items():
			setattr(self, key, float(value))

		if self.filter_frequency:
			self.bandpass_filter = 0  # turns off FIR

		# Assign values from inputs.
		self.signal_array = signal_array
		self.signal_orig = None  # used in amplitude calc to undo any Windowing beforehand
		if pycroscopy:
			self.signal_array = signal_array.T
		self.tidx = int(self.trigger * self.sampling_rate)

		# Set dimensions correctly
		# Three cases: 1) 2D (has many averages) 2) 1D (but set as 1xN) and 3) True 1D
		if len(signal_array.shape) == 2 and 1 not in signal_array.shape:

			self.n_points, self.n_signals = self.signal_array.shape
			self._n_points_orig = self.signal_array.shape[0]

		else:

			self.n_signals = 1
			self.signal_array = self.signal_array.flatten()
			self.n_points = self.signal_array.shape[0]
			self._n_points_orig = self.signal_array.shape[0]

		# Keep the original values for restoring the signal properties.
		self._tidx_orig = self.tidx
		self.tidx_orig = self.tidx

		# Initialize attributes that are going to be assigned later.
		self.signal = None
		self.phase = None
		self.inst_freq = None
		self.tfp = None
		self.shift = None
		self.cwt_matrix = None

		self.verbose = False  # for console feedback

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

		return

	def clear_filter_flags(self):
		"""Removes flags from parameters for setting filters"""

		self.bandpass_filter = 0

		return

	def remove_dc(self):
		"""Removes DC components from signals."""

		if self.n_signals != 1:

			for i in range(self.n_signals):
				self.signal_array[:, i] -= np.mean(self.signal_array[:, i])

		return

	def phase_lock(self):
		"""Phase-locks signals in the signal array. This also cuts signals."""

		# Phase-lock signals.
		self.signal_array, self.tidx = noise.phase_lock(self.signal_array, self.tidx,
														np.ceil(self.sampling_rate / self.drive_freq))

		# Update number of points after phase-locking.
		self.n_points = self.signal_array.shape[0]

		return

	def average(self):
		"""Averages signals."""

		if self.n_signals != 1:  # if not multi-signal, don't average
			self.signal = self.signal_array.mean(axis=1)

		else:
			self.signal = np.copy(self.signal_array)

		return

	def check_drive_freq(self):
		"""Calculates drive frequency of averaged signals, and check against
		   the given drive frequency."""

		n_fft = 2 ** int(np.log2(self.tidx))  # For FFT, power of 2.
		dfreq = self.sampling_rate / n_fft  # Frequency separation.

		# Calculate drive frequency from maximum power of the FFT spectrum.
		signal = self.signal[:n_fft]
		fft_amplitude = np.abs(np.fft.rfft(signal))
		drive_freq = fft_amplitude.argmax() * dfreq

		# Difference between given and calculated drive frequencies.
		difference = np.abs(drive_freq - self.drive_freq)

		# If difference is too big, reassign. Otherwise, continue. != 0 for accidental DC errors
		if difference >= dfreq and drive_freq != 0:
			self.drive_freq = drive_freq

		return

	def apply_window(self):
		"""Applies the window given in parameters."""

		self.signal *= sps.get_window(self.window, self.n_points)

		return

	def dwt_denoise(self):
		"""Uses DWT to denoise the signal prior to processing."""

		rate = self.sampling_rate
		lpf = self.drive_freq * 0.1
		self.signal, _, _ = dwavelet.dwt_denoise(self.signal, lpf, rate / 2, rate)

		return

	def fir_filter(self):
		"""Filters signal with a FIR bandpass filter."""

		# Calculate bandpass region from given parameters.
		nyq_rate = 0.5 * self.sampling_rate
		bw_half = self.filter_bandwidth / 2

		freq_low = (self.drive_freq - bw_half) / nyq_rate
		freq_high = (self.drive_freq + bw_half) / nyq_rate

		band = [freq_low, freq_high]

		# Create taps using window method.
		try:
			taps = sps.firwin(int(self.n_taps), band, pass_zero=False,
							  window='blackman')
		except:
			print('band=', band)
			print('nyq=', nyq_rate)
			print('drive=', self.drive_freq)

		self.signal = sps.fftconvolve(self.signal, taps, mode='same')

		# Shifts trigger due to causal nature of FIR filter
		self.tidx -= (self.n_taps - 1) / 2

		return

	def iir_filter(self):
		"""Filters signal with two Butterworth filters (one lowpass,
		one highpass) using filtfilt. This method has linear phase and no
		time delay."""

		# Calculate bandpass region from given parameters.
		nyq_rate = 0.5 * self.sampling_rate
		bw_half = self.filter_bandwidth / 2

		freq_low = (self.drive_freq - bw_half) / nyq_rate
		freq_high = (self.drive_freq + bw_half) / nyq_rate

		# Do a high-pass filtfilt operation.
		b, a = sps.butter(9, freq_low, btype='high')
		self.signal = sps.filtfilt(b, a, self.signal)

		# Do a low-pass filtfilt operation.
		b, a = sps.butter(9, freq_high, btype='low')
		self.signal = sps.filtfilt(b, a, self.signal)

		return

	def amplitude_filter(self):
		'''
		Filters the drive signal out of the amplitude response
		'''
		AMP = np.fft.fftshift(np.fft.fft(self.amplitude))

		DRIVE = self.drive_freq / (self.sampling_rate / self.n_points)  # drive location in frequency space
		center = int(len(AMP) / 2)

		# crude boxcar
		AMP[:center - int(DRIVE / 2) + 1] = 0
		AMP[center + int(DRIVE / 2) - 1:] = 0

		self.amplitude = np.abs(np.fft.ifft(np.fft.ifftshift(AMP)))

		return

	def frequency_filter(self):
		'''
		Filters the instantaneous frequency around DC peak to remove noise
		Uses self.filter_bandwidth for the frequency filter
		'''
		FREQ = np.fft.fftshift(np.fft.fft(self.inst_freq))

		center = int(len(FREQ) / 2)

		df = self.sampling_rate / self.n_points
		drive_bin = int(np.ceil(self.drive_freq / df))
		bin_width = int(self.filter_bandwidth / df)

		if bin_width > drive_bin:
			print('width exceeds first resonance')
			bin_width = drive_bin - 1

		FREQ[:center - bin_width] = 0
		FREQ[center + bin_width:] = 0

		self.inst_freq = np.real(np.fft.ifft(np.fft.ifftshift(FREQ)))

		return

	def frequency_harmonic_filter(self, width=5):
		'''
		Filters the instantaneous frequency to remove noise
		Defaults to DC and then every multiple harmonic up to sampling
		
		Parameters
		----------
		width : int, optional
			Size of the boxcar around the various peaks
		'''
		FREQ = np.fft.fftshift(np.fft.fft(self.inst_freq))

		center = int(len(FREQ) / 2)

		# Find drive_bin
		df = self.sampling_rate / self.n_points
		drive_bin = int(np.ceil(self.drive_freq / df))
		bins = np.arange(len(FREQ) / 2)[::drive_bin]
		bins = np.append(center - bins, center + bins)

		FREQ_filt = np.zeros(len(FREQ), dtype='complex128')
		for b in bins:
			FREQ_filt[int(b) - width:int(b) + width] = FREQ[int(b) - width:int(b) + width]

		self.inst_freq = np.real(np.fft.ifft(np.fft.ifftshift(FREQ)))

		return

	def hilbert(self):
		"""Analytical signal and calculate phase/frequency via Hilbert transform"""

		self.hilbert_transform()
		self.calculate_amplitude()
		self.calculate_phase()
		self.calculate_inst_freq()

		return

	def hilbert_transform(self):
		"""Gets the analytical signal doing a Hilbert transform."""

		self.signal = sps.hilbert(self.signal)

		return

	def calculate_amplitude(self):
		"""Calculates the amplitude of the analytic signal. Uses pre-filter
		signal to do this."""
		#
		if self.n_signals != 1:
			signal_orig = self.signal_array.mean(axis=1)
		else:
			signal_orig = self.signal_array

		self.amplitude = np.abs(sps.hilbert(signal_orig))

		if not np.isnan(self.AMPINVOLS):
			self.amplitude *= self.AMPINVOLS

		return

	def calculate_power_dissipation(self):
		"""Calculates the power dissipation using amplitude, phase, and frequency
		and the Cleveland eqn (see DOI:10.1063/1.121434)"""

		phase = self.phase  # + np.pi/2 #offsets the phase to be pi/2 at resonance

		# check for incorrect values (some off by 1e9 in our code)

		A = self.k / self.Q * self.amplitude ** 2 * (self.inst_freq + self.drive_freq)
		B = self.Q * self.DriveAmplitude * np.sin(phase) / self.amplitude
		C = self.inst_freq / self.drive_freq

		self.power_dissipated = A * (B - C)

		return

	def calculate_phase(self, correct_slope=True):
		"""Gets the phase of the signal and correct the slope by removing
		the drive phase."""

		# Unwrap the phase.
		self.phase = np.unwrap(np.angle(self.signal))

		if correct_slope:
			# Remove the drive from phase.
			# self.phase -= (2 * np.pi * self.drive_freq *
			#               np.arange(self.n_points) / self.sampling_rate)

			# A curve fit on the initial part to make sure that it worked.
			start = int(0.3 * self.tidx)
			end = int(0.7 * self.tidx)
			fit = self.phase[start:end]

			xfit = np.polyfit(np.arange(start, end), fit, 1)

			# Remove the fit from phase.
			self.phase -= (xfit[0] * np.arange(self.n_points)) + xfit[1]

		self.phase = -self.phase  # need to correct for negative in DDHO solution

		self.phase += np.pi / 2  # corrects to be at resonance pre-trigger

		return

	def calculate_inst_freq(self):
		"""Calculates the first derivative of the phase using Savitzky-Golay
		filter."""

		dtime = 1 / self.sampling_rate  # Time step.

		# Do a Savitzky-Golay smoothing derivative
		# using 5 point 1st order polynomial.

		# -self.phase to correct for sign in DDHO solution
		self.inst_freq_raw = sps.savgol_filter(-self.phase, 5, 1, deriv=1,
											   delta=dtime)

		# Bring trigger to zero.
		self.tidx = int(self.tidx)
		self.inst_freq = self.inst_freq_raw - self.inst_freq_raw[self.tidx]

		return

	def find_tfp(self):
		"""Calculate tfp and shift based self.fit_form and self.fit selection"""
		ridx = int(self.roi * self.sampling_rate)
		cut = self.inst_freq[self.tidx:(self.tidx + ridx)]
		cut -= self.inst_freq[self.tidx]
		self.cut = cut
		t = np.arange(cut.shape[0]) / self.sampling_rate

		if not self.fit:

			tfp_calc.find_minimum(self, cut)

		elif self.fit_form == 'sum':

			tfp_calc.fit_freq_sum(self, cut, t)

		elif self.fit_form == 'exp':

			tfp_calc.fit_freq_exp(self, cut, t)

		elif self.fit_form == 'ringdown':

			cut = self.amplitude[self.tidx:(self.tidx + ridx)]
			tfp_calc.fit_ringdown(self, ridx, cut, t)

		elif self.fit_form == 'product':

			tfp_calc.fit_freq_product(self, cut, t)

		elif self.fit_form == 'phase':

			cut = -1 * (self.phase[self.tidx:(self.tidx + ridx)] - self.phase[self.tidx])
			tfp_calc.fit_phase(self, cut, t)

		return

	def restore_signal(self):
		"""Restores the signal length and position of trigger to original
		values."""

		# Difference between current and original values.
		d_trig = int(self._tidx_orig - self.tidx)
		d_points = int(self._n_points_orig - self.n_points)

		# Check if the signal length can accomodate the shift or not.
		if d_trig >= d_points:

			# Pad from left and set the original length.
			self.inst_freq = np.pad(self.inst_freq, (d_trig, 0), 'edge')
			self.inst_freq = self.inst_freq[:self._n_points_orig]

			self.phase = np.pad(self.phase, (d_trig, 0), 'edge')
			self.phase = self.phase[:self._n_points_orig]

			self.amplitude = np.pad(self.amplitude, (d_trig, 0), 'edge')
			self.amplitude = self.amplitude[:self._n_points_orig]

		else:

			# Calculate how many points is needed for padding from right.
			pad_right = d_points - d_trig
			self.inst_freq = np.pad(self.inst_freq, (d_trig, pad_right),
									'edge')
			self.phase = np.pad(self.phase, (d_trig, pad_right),
								'edge')
			self.amplitude = np.pad(self.amplitude, (d_trig, pad_right),
									'edge')

		# Set the public variables back to original values.
		self.tidx = self._tidx_orig
		self.n_points = self._n_points_orig

		return

	def calculate_cwt(self, f_center=None, verbose=False, optimize=False, fit=False):
		'''
		Calculate instantaneous frequency using continuous wavelet transfer
		
		wavelet specified in self.wavelet. See PyWavelets CWT documentation
		
		Parameters
		----------
		Optimize : bool, optionals
			Currently placeholder for iteratively determining wavelet scales
		'''

		# wavlist = pywt.wavelist(kind='continuous')
		# w0, wavelet_increment, cwt_scale = self.__get_cwt__()

		# determine if scales will capture the relevant frequency
		if not f_center:
			f_center = self.drive_freq

		dt = 1 / self.sampling_rate
		sc = pywt.scale2frequency(self.wavelet, self.scales) / dt
		if self.verbose:
			print('Wavelet scale from', np.min(sc), 'to', np.max(sc))

		if f_center < np.min(sc) or f_center > np.max(sc):
			raise ValueError('Choose a scale that captures frequency of interest')

		if optimize:
			print('!')
			drive_bin = self.scales[np.searchsorted(sc, f_center)]
			hi = int(1.2 * drive_bin)
			lo = int(0.8 * drive_bin)
			self.scales = np.arange(hi, lo, -0.1)

		spectrogram, freq = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=dt)

		if not fit:

			inst_freq, amplitude, _ = parab.ridge_finder(np.abs(spectrogram), np.arange(len(freq)))

		# slow serial curve fitting
		else:

			inst_freq = np.zeros(self.n_points)
			amplitude = np.zeros(self.n_points)

			for c in range(spectrogram.shape[1]):

				SIG = spectrogram[:, c]
				if fit:
					pk = np.argmax(np.abs(SIG))
					popt = np.polyfit(np.arange(20),
									  np.abs(SIG[pk - 10:pk + 10]), 2)
					inst_freq[c] = -0.5 * popt[1] / popt[0]
					amplitude[c] = np.abs(SIG)[pk]

		# rescale to correct frequency 
		inst_freq = pywt.scale2frequency(self.wavelet, inst_freq + self.scales[0]) / dt

		phase = spg.cumtrapz(inst_freq)
		phase = np.append(phase, phase[-1])
		tidx = int(self.tidx * len(inst_freq) / self.n_points)

		self.amplitude = amplitude
		self.inst_freq_raw = inst_freq
		self.inst_freq = -1 * (inst_freq - inst_freq[tidx])  # -1 due to way scales are ordered
		self.spectrogram = np.abs(spectrogram)
		self.wavelet_freq = freq  # the wavelet frequencies

		# subtract the w*t line (drive frequency line) from phase
		start = int(0.3 * tidx)
		end = int(0.7 * tidx)
		xfit = np.polyfit(np.arange(start, end), phase[start:end], 1)
		phase -= (xfit[0] * np.arange(len(inst_freq))) + xfit[1]

		self.phase = phase

		return

	def calculate_stft(self, time_res=20e-6, nfft=200):
		'''
		Sliding FFT approach
		
		Parameters
		----------
		time_res : float, optional
			What timescale to evaluate each FFT over
				  
		fit : bool, optional
			Fits a parabola to the frequency peak to get the actual frequency
			Otherwise defaults to parabolic interpolation (see parab.fit)
		
		nfft : int
			Length of FFT calculated in the spectrogram. More points gets much slower
			but the longer the FFT the finer the frequency bin spacing            
		'''

		pts_per_ncycle = int(time_res * self.sampling_rate)

		if nfft < pts_per_ncycle:
			print('Error with nfft setting')
			nfft = pts_per_ncycle

		# drivebin = int(self.drive_freq / (self.sampling_rate / nfft ))
		freq, times, spectrogram = sps.spectrogram(self.signal,
												   self.sampling_rate,
												   nperseg=pts_per_ncycle,
												   noverlap=pts_per_ncycle - 1,
												   nfft=nfft,
												   window=self.window,
												   mode='magnitude')

		# Parabolic ridge finder
		inst_freq, amplitude, _ = parab.ridge_finder(spectrogram, freq)

		# Correctly pad the signals
		_pts = self.n_points - len(inst_freq)
		_pre = int(np.floor(_pts / 2))
		_post = int(np.ceil(_pts / 2))

		inst_freq = np.pad(inst_freq, (_pre, _post))
		amplitude = np.pad(amplitude, (_pre, _post))

		phase = spg.cumtrapz(inst_freq)
		phase = np.append(phase, phase[-1])
		tidx = int(self.tidx * len(inst_freq) / self.n_points)

		self.amplitude = amplitude
		self.inst_freq_raw = inst_freq
		self.inst_freq = inst_freq - inst_freq[tidx]
		self.spectrogram = spectrogram
		self.stft_freq = freq
		self.stft_times = times

		# subtract the w*t line (drive frequency line) from phase
		start = int(0.3 * tidx)
		end = int(0.7 * tidx)
		xfit = np.polyfit(np.arange(start, end), phase[start:end], 1)
		phase -= (xfit[0] * np.arange(len(inst_freq))) + xfit[1]

		self.phase = phase

		return

	def plot(self, newplot=True, raw=False):
		""" 
		Quick visualization of best_fit and cut.
		
		Parameters
		----------
		newplot : bool, optional
			generates a new plot (True) or plots on existing plot figure (False)
		
		raw : bool, optional
		"""

		if newplot:
			fig, a = plt.subplots(nrows=3, figsize=(6, 9), facecolor='white')

		dt = 1 / self.sampling_rate
		ridx = int(self.roi * self.sampling_rate)
		fidx = int(self.tidx)

		cut = self.amplitude[fidx:(fidx + ridx)]

		cut = [fidx, (fidx + ridx)]
		tx = np.arange(cut[0], cut[1]) * dt

		a[0].plot(tx * 1e3, self.inst_freq[cut[0]:cut[1]], 'r-')

		if self.fit_form == 'ringdown':
			a[1].plot(tx * 1e3, self.best_fit, 'g--')
		else:
			a[0].plot(tx * 1e3, self.best_fit, 'g--')
		a[1].plot(tx * 1e3, self.amplitude[cut[0]:cut[1]], 'b')
		a[2].plot(tx * 1e3, self.phase[cut[0]:cut[1]] * 180 / np.pi, 'm')

		a[0].set_title('Instantaneous Frequency')
		a[0].set_ylabel('Frequency Shift (Hz)')
		a[1].set_ylabel('Amplitude (nm)')
		a[2].set_ylabel('Phase (deg)')
		a[2].set_xlabel('Time (ms)')

		plt.tight_layout()

		return

	def generate_inst_freq(self, timing=False):
		"""
		Generates the instantaneous frequency
		
		Parameters
		----------
		timing : bool, optional
			prints the time to execute (for debugging)

		Returns
		-------
		inst_freq : (n_points,) array_like
			Instantaneous frequency of the signal.
		"""

		if timing:
			t1 = time.time()

		# Remove DC component, first.
		# self.remove_dc()

		# Phase-lock signals.
		# self.phase_lock()

		# Average signals.
		self.average()

		# Remove DC component again, introduced by phase-locking.
		# self.remove_dc()

		# Check the drive frequency.
		if self.check_drive:
			self.check_drive_freq()

		# DWT Denoise
		# self.dwt_denoise()

		if self.method == 'wavelet':

			# Calculate instantenous frequency using wavelet transform.
			self.calculate_cwt(**self.wavelet_params)

		elif self.method == 'stft':

			# Calculate instantenous frequency using sliding FFT
			self.calculate_stft(**self.fft_params)

		elif self.method == 'hilbert':
			# Hilbert transform method

			# Apply window.
			if self.window != 0:
				self.apply_window()

			# Filter the signal with a filter, if wanted.
			if self.bandpass_filter == 1:

				self.fir_filter()

			elif self.bandpass_filter == 2:

				self.iir_filter()

			# Get the analytical signal doing a Hilbert transform.
			self.hilbert()

			# Filter out oscillatory noise from amplitude
			if self.filter_amplitude:
				self.amplitude_filter()

		else:
			raise ValueError('Invalid analysis method! Valid options: hilbert, wavelet, fft')

		if timing:
			print('Time:', time.time() - t1, 's')

		# Filter out oscillatory noise from instantaneous frequency
		if self.filter_frequency:
			self.frequency_filter()

		return self.inst_freq, self.amplitude, self.phase

	def analyze(self):
		"""
		Analyzes the pixel with the given method.

		Returns
		-------
		tfp : float
			Time from trigger to first-peak, in seconds.
		shift : float
			Frequency shift from trigger to first-peak, in Hz.
		inst_freq : (n_points,) array_like
			Instantenous frequency of the signal.
		"""

		self.inst_freq, self.amplitude, self.phase = self.generate_inst_freq()

		# If it's a recombination image invert it to find minimum.
		if self.recombination:
			self.inst_freq = self.inst_freq * -1

		# Find where the minimum is.
		self.find_tfp()

		# Restore the length due to FIR filter being causal
		if self.method == 'hilbert':
			self.restore_signal()

		# If it's a recombination image invert it to find minimum.
		if self.recombination:
			self.inst_freq = self.inst_freq * -1
			self.best_fit = self.best_fit * -1
			self.cut = self.cut * -1

		if self.phase_fitting:

			return self.tfp, self.shift, self.phase

		else:

			return self.tfp, self.shift, self.inst_freq
