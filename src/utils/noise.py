"""noise.py: Includes functions for reducing noise in a pixel."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
import scipy.linalg as spl
import scipy.spatial.distance as spsd


def phase_lock(signal_array, tidx):
    """
    Align signals of a pixel in phase, if they are not aligned due to phase
    jitter of analog-to-digital converter.

    Parameters
    ----------
    signal_array : array, [n_points, n_signals]
        2D real-valued signal array.

    tidx: int
        Time to trigger from the start of signals as index.

    Returns
    -------
    signal_array : array, [n_points, n_signals]
        Phase-locked signal array

    tidx : int
        Index of trigger after phase-locking.

    """

    # Initialize variables.
    n_points, n_signals = signal_array.shape
    index_shift = np.empty(n_signals)  # Keep shifts from the beginning.
    idx = int(0.01 * n_points)  # Look at only the first 1% points of signals.

    # Fast zero-finding function in lambda form.
    fzero = lambda x: np.where(x[:-1] * x[1:] <= 0)[0][0]

    # Find positions of first zeros from beginning to 1% of a signal.
    for j in range(n_signals):

        index_shift[j] = fzero(signal_array[:idx, j])

    # Have all signals same length by cutting them from both ends.
    total_cut = index_shift.max() + index_shift.min()
    new_signal_array = np.empty((n_points - total_cut, n_signals))
    tidx -= int(index_shift.mean())  # Shift trigger index by average amount.

    # Cut signals.
    for j in range(n_signals):

        new_signal_array[:, j] = \
            signal_array[index_shift[j]:-(total_cut - index_shift[j]), j]

    signal_array = new_signal_array
    # Find the new trigger index.
    tidx += fzero(signal_array.mean(axis=1)[
        (tidx - total_cut):(tidx + total_cut)]) - total_cut

    return signal_array, tidx


def discard(signals, k):
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

    (d, N) = signals.shape  # Get the size of signals.

    meansignal = signals.mean(axis=1).reshape(d, 1)  # Calculate mean signal.
    signals -= meansignal  # Subtract mean signal from every signal.

    # Calculate the biggest eigenvector of covariance matrix.
    eigval, eigvec = spl.eigh(np.dot(signals.T, signals), eigvals=(N-k, N-1))
    eigsig = np.dot(signals, eigvec)  # Convert it to eigensignal.

    weights = np.dot(eigsig.T, signals) / spl.norm(eigsig)
    mu = weights.mean(axis=1).reshape(1, k)
    idx = np.where(spsd.cdist(weights.T, mu, 'mahalanobis') > 2)

    return idx
