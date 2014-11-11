"""noise.py: Includes functions for reducing noise in a pixel."""

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
    signal_array : array, [n_points, n_signals]
        2D real-valued signal array.

    tidx: int
        Time to trigger from the start of signals as index.

    cidx: int
        Period of the signal as number of points, i.e. drive_freq/sampling_rate

    Returns
    -------
    signal_array : array, [n_points, n_signals]
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

    signal_array = new_signal_array
    # Find the new trigger index.
    tidx += fzero(signal_array.mean(axis=1)[
        (tidx - total_cut):(tidx + total_cut)]) - total_cut

    return signal_array, tidx


def pca_discard(signals, k):
    """
    Discards noisy signals using Principal Component Analysis.

    Parameters
    ----------
    signals : array_like

            Real signals of a pixel, should be in a format of dxN.

    Returns
    -------
    idx : array_like

            Return the indices of noisy signals.

    """

    d, N = signals.shape  # Get the size of signals.

    meansignal = signals.mean(axis=1).reshape(d, 1)  # Calculate mean signal.
    signals -= meansignal  # Subtract mean signal from every signal.

    # Calculate the biggest eigenvector of covariance matrix.
    _, eigvec = spl.eigh(np.dot(signals.T, signals), eigvals=(N-k, N-1))
    eigsig = np.dot(signals, eigvec)  # Convert it to eigensignal.

    weights = np.dot(eigsig.T, signals) / spl.norm(eigsig)
    mu = weights.mean(axis=1).reshape(1, k)
    idx = np.where(spsd.cdist(weights.T, mu, 'mahalanobis') > 2)

    return idx


def fzero(x):
    """Fast zero-finding function in lambda form."""
    return np.where(x[:-1] * x[1:] <= 0)[0][0]
