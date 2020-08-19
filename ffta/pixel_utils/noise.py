"""noise.py: Includes functions for reducing noise in a pixel."""
# pylint: disable=E1101,R0902
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"

import numpy as np
import scipy.linalg as spl
import scipy.spatial.distance as spsd


def phase_lock(signal_array, tidx, cidx):
	"""
	Aligns signals of a pixel on the rising edge of first zeros, if they are
	not aligned due to phase jitter of analog-to-digital converter.

	Parameters
	----------
	signal_array : (n_points, n_signals), array_like
		2D real-valued signal array.
	tidx: int
		Time to trigger from the start of signals as index.
	cidx: int
		Period of the signal as number of points,
		i.e. drive_freq/sampling_rate

	Returns
	-------
	signal_array : (n_points, n_signals), array_like
		Phase-locked signal array
	tidx : int
		Index of trigger after phase-locking.

	"""

	# Initialize variables.
	n_points, n_signals = signal_array.shape

	above = signal_array[:cidx, :] > 0
	below = np.logical_not(above)
	left_shifted_above = above[1:, :]

	idxs = (left_shifted_above[:, :] & below[0:-1, :]).argmax(axis=0)

	# Have all signals same length by cutting them from both ends.
	total_cut = int(idxs.max() + idxs.min() + 1)
	new_signal_array = np.empty((n_points - total_cut, n_signals))
	tidx -= int(idxs.mean())  # Shift trigger index by average amount.

	# Cut signals.
	for i, idx in enumerate(idxs):

		new_signal_array[:, i] = signal_array[idx:-(total_cut - idx), i]

	return new_signal_array, tidx


def pca_discard(signal_array, k):
	"""
	Discards noisy signals using Principal Component Analysis and Mahalonobis
	distance.

	Parameters
	----------
	signal_array : (n_points, n_signals), array_like
		2D real-valued signal array.
	k : int
		Number of eigenvectors to use, can't be bigger than n_signals.

	Returns
	-------
	idx : array
		Return the indices of noisy signals.

	"""

	# Get the size of signals.
	n_points, n_signals = signal_array.shape

	# Remove the mean from all signals.
	mean_signal = signal_array.mean(axis=1).reshape(n_points, 1)
	signal_array = signal_array - mean_signal

	# Calculate the biggest eigenvector of covariance matrix.
	_, eigvec = spl.eigh(np.dot(signal_array.T, signal_array),
						 eigvals=(n_signals - k, n_signals - 1))

	eigsig = np.dot(signal_array, eigvec)  # Convert it to eigensignal.

	weights = np.dot(eigsig.T, signal_array) / spl.norm(eigsig)
	mean = weights.mean(axis=1).reshape(1, k)
	idx = np.where(spsd.cdist(weights.T, mean, 'mahalanobis') > 2)

	return idx
