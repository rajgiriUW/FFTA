# -*- coding: utf-8 -*-

"""tfp.py: Routines for fitting the frequency/phase/amplitude to extract tFP/shift """

from . import fitting
import numpy as np
from scipy import interpolate as spi
from scipy import optimize as spo


def find_minimum(pix, cut):
	"""
	Finds when the minimum of instantaneous frequency happens using spline fitting
	
	Parameters
	----------
	pix : ffta.pixel.Pixel object
		pixel object to analyze
	cut : ndarray
		The slice of frequency data to fit against

	Returns
	-------
	pix.tfp : float
		tFP value
	pix.shift : float
		frequency shift value at time t=tfp
	pix.best_fit : ndarray
		Best-fit line calculated from spline function

	"""

	# Cut the signal into region of interest.
	# ridx = int(pix.roi * pix.sampling_rate)
	# cut = pix.inst_freq[pix.tidx:(pix.tidx + ridx)]

	# Define a spline to be used in finding minimum.
	ridx = len(cut)

	x = np.arange(ridx)
	y = cut

	_spline_sz = 2 * pix.sampling_rate / pix.drive_freq
	func = spi.UnivariateSpline(x, y, k=4, ext=3, s=_spline_sz)

	# Find the minimum of the spline using TNC method.
	res = spo.minimize(func, cut.argmin(),
					   method='TNC', bounds=((0, ridx),))
	idx = res.x[0]

	pix.best_fit = func(np.arange(ridx))

	# Do index to time conversion and find shift.
	pix.tfp = idx / pix.sampling_rate
	pix.shift = func(0) - func(idx)

	return


def fit_freq_product(pix, cut, t):
	'''
	Fits the frequency shift to an approximate functional form using
	an analytical fit with bounded values.

	Parameters
	----------
	pix : ffta.pixel.Pixel object
		pixel object to analyze
	cut : ndarray
		The slice of frequency data to fit against
	t : ndarray
		The time-array (x-axis) for fitting

	Returns
	-------
	pix.tfp : float
		tFP value
	pix.shift : float
		frequency shift value at time t=tfp
	pix.rms : float
		fitting error
	pix.popt : ndarray
		The fit parameters for the function fitting.fit_product
	pix.best_fit : ndarray
		Best-fit line calculated from popt and fit function

	'''

	# Fit the cut to the model.
	popt = fitting.fit_product(pix.Q, pix.drive_freq, t, cut)

	A, tau1, tau2 = popt

	# Analytical minimum of the fit.
	# self.tfp = tau2 * np.log((tau1 + tau2) / tau2)
	# self.shift = -A * np.exp(-self.tfp / tau1) * np.expm1(-self.tfp / tau2)

	# For diagnostic purposes.
	pix.popt = popt
	pix.best_fit = -A * (np.exp(-t / tau1) - 1) * np.exp(-t / tau2)

	pix.tfp = np.argmin(pix.best_fit) / pix.sampling_rate
	pix.shift = np.min(pix.best_fit)

	pix.rms = np.sqrt(np.mean(np.square(pix.best_fit - cut)))

	return


def fit_freq_sum(pix, ridx, cut, t):
	'''
	Fits the frequency shift to an approximate functional form using
	an analytical fit with bounded values.
	Parameters
	----------
	pix : ffta.pixel.Pixel object
		pixel object to analyze
	cut : ndarray
		The slice of frequency data to fit against
	t : ndarray
		The time-array (x-axis) for fitting

	Returns
	-------
	pix.tfp : float
		tFP value
	pix.shift : float
		frequency shift value at time t=tfp
	pix.rms : float
		fitting error
	pix.popt : ndarray
		The fit parameters for the function fitting.fit_sum
	pix.best_fit : ndarray
		Best-fit line calculated from popt and fit function
	'''
	# Fit the cut to the model.
	popt = fitting.fit_sum(pix.Q, pix.drive_freq, t, cut)
	A1, A2, tau1, tau2 = popt

	# For diagnostic purposes.
	pix.popt = popt
	pix.best_fit = A1 * (np.exp(-t / tau1) - 1) - A2 * np.exp(-t / tau2)

	pix.tfp = np.argmin(pix.best_fit) / pix.sampling_rate
	pix.shift = np.min(pix.best_fit)

	return


def fit_freq_exp(pix, ridx, cut, t):
	'''
	Fits the frequency shift to a single exponential in the case where
	there is no return to 0 Hz offset (if drive is cut).
	
	Parameters
	----------
	pix : ffta.pixel.Pixel object
		pixel object to analyze
	cut : ndarray
		The slice of frequency data to fit against
	t : ndarray
		The time-array (x-axis) for fitting

	Returns
	-------
	pix.tfp : float
		tFP value
	pix.shift : float
		frequency shift value at time t=tfp
	pix.popt : ndarray
		The fit parameters for the function fitting.fit_exp
	pix.best_fit : ndarray
		Best-fit line calculated from popt and fit function
	'''
	# Fit the cut to the model.
	popt = fitting.fit_exp(t, cut)

	# For diagnostics
	A, y0, tau = popt
	pix.popt = popt
	pix.best_fit = A * (np.exp(-t / tau)) + y0

	pix.shift = A
	pix.tfp = tau

	return


def fit_ringdown(pix, ridx, cut, t):
	'''
	Fits the amplitude to determine Q from single exponential fit.
	
	Parameters
	----------
	pix : ffta.pixel.Pixel object
		pixel object to analyze
	cut : ndarray
		The slice of amplitude data to fit against
	t : ndarray
		The time-array (x-axis) for fitting

	Returns
	-------
	pix.tfp : float
		Q calculated from ringdown equation
	pix.ringdown_Q : float
		Same as tFP. This is the actual variable, tFP is there for code simplicity
	pix.shift : float
		amplitude of the single exponential decay
	pix.popt : ndarray
		The fit parameters for the function fitting.fit_ringdown
	pix.best_fit : ndarray
		Best-fit line calculated from popt and fit function
	'''
	# Fit the cut to the model.
	popt = fitting.fit_ringdown(t, cut * 1e9)
	popt[0] *= 1e-9
	popt[1] *= 1e-9

	# For diagnostics
	A, y0, tau = popt
	pix.popt = popt
	pix.best_fit = A * (np.exp(-t / tau)) + y0

	pix.shift = A
	pix.tfp = np.pi * pix.drive_freq * tau  # same as ringdown_Q to help with pycroscopy bugs that call tfp
	pix.ringdown_Q = np.pi * pix.drive_freq * tau

	return


def fit_phase(pix, ridx, cut, t):
	'''
	Fits the phase to an approximate functional form using an
	analytical fit with bounded values.
	
	Parameters
	----------
	pix : ffta.pixel.Pixel object
		pixel object to analyze
	cut : ndarray
		The slice of frequency data to fit against
	t : ndarray
		The time-array (x-axis) for fitting

	Returns
	-------
	pix.tfp : float
		tFP value
	pix.shift : float
		frequency shift value at time t=tfp
	pix.popt : ndarray
		The fit parameters for the function fitting.fit_phase
	pix.best_fit : ndarray
		Best-fit line calculated from popt and fit function for the frequency data
	pix.best_phase : ndarray
		Best-fit line calculated from popt and fit function for the phase data
	'''
	# Fit the cut to the model.
	popt = fitting.fit_phase(pix.Q, pix.drive_freq, t, cut)

	A, tau1, tau2 = popt

	# Analytical minimum of the fit.
	pix.tfp = tau2 * np.log((tau1 + tau2) / tau2)
	pix.shift = A * np.exp(-pix.tfp / tau1) * np.expm1(-pix.tfp / tau2)

	# For diagnostic purposes.
	postfactor = (tau2 / (tau1 + tau2)) * np.exp(-t / tau2) - 1

	pix.popt = popt
	pix.best_fit = -A * np.exp(-t / tau1) * np.expm1(-t / tau2)
	pix.best_phase = A * tau1 * np.exp(-t / tau1) * postfactor + A * tau1 * (1 - tau2 / (tau1 + tau2))

	return
