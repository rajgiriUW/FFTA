"""dwavelet.py: contains functions used in DWT calculations."""

import pywt
import numpy as np

def dwt_denoise(signal, fLow, fHigh, sampling_rate):
	""" Uses Discrete Wavelet Transform to denoise signal
		around a desired frequency band.

	   Parameters
	   ----------
	   fLow: float
		   frequency below which DWT coefficients zeroed

	   fHigh: float
		   frequency above which DWT coefficients zeroed

	   sampling_rate: float
		   Sample rate of signal in Hz

	"""

	coeffs = pywt.wavedec(signal, 'db1')

	frequencies = np.zeros(len(coeffs))

	# The upper frequency at each DWT level
	for i in xrange(len(coeffs)):

		frequencies[-1-i] = sampling_rate / (2 ** (i+1))

	# Levels corresponding to fLow and fHigh
	fLow_idx = np.searchsorted(frequencies, fLow)
	fHigh_idx = np.searchsorted(frequencies, fHigh)

	# Correct issues with the index searching
	if frequencies[fLow_idx] > fLow:

		fLow_idx -= 1

	if frequencies[fHigh_idx] < fHigh:

		fHigh_idx = np.min([fHigh_idx+1, xrange(len(coeffs))])

	# Set coefficients to 0
	for i in xrange(fLow_idx):

		coeffs[i][:] = coeffs[i][:] * 0

#    for i in xrange(len(coeffs) - fHigh_idx):
#        coeffs[i+fHigh_idx][:] = coeffs[i+fHigh_idx][:] * 0

	denoised = pywt.waverec(coeffs, 'db1')

	return denoised, coeffs, frequencies

def dwt_scalogram(coeffs):

	maxsize = len(coeffs[-1])

	scalogram = np.zeros(maxsize)
	xpts = np.arange(maxsize)

	# interpolates each row over [0...maxsize]
	for i in xrange(len(coeffs)):

	   rptx = np.linspace(0, maxsize, len(coeffs[i]))
	   samplerow = np.interp(xpts, rptx, coeffs[i])
	   scalogram = np.vstack((scalogram, samplerow) )

	# delete dummy first row that's needed for vstack to work
	scalogram = np.delete(scalogram, (0), axis=0)

	return scalogram
