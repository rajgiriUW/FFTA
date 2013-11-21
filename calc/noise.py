"""noise.py: Includes functions for reducing noise in a pixel."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
import scipy.linalg as spl
import scipy.spatial.distance as spsd


def phaselock(signals, tidx):
    """
    Aligns signals of a pixel in phase, if they are not aligned due to phase
    jitter of analog-to-digital converter.

    Parameters
    ----------
    signals : array_like

            Real signals of a pixel, should be in a format of dxN.

    tidx: int

            Time to trigger from the start of signals as index.

    Returns
    -------
    nsignals : array_like

            Phase-locked signals of a pixel.

    tidx : int

            Index of trigger after phase-locking.

    """

    # Initialize variables.
    (d, N) = signals.shape
    shift = np.empty(N)  # Shift from the beginning.
    sidx = int(0.01 * d)  # Look at only the first .003 points of signals.

    fzero = lambda signal: np.where(signal[:-1] * signal[1:] <= 0)[0][0]

    # Find positions of first zeros from beginning.
    for n in range(N):

        shift[n] = fzero(signals[:sidx, n])

    # Have all signals same length by cutting them from both ends.
    totalcut = shift.max() + shift.min()  # Number of points that is discarded.
    nsignals = np.empty((d - totalcut, N))  # Initialize new signals.
    tidx -= int(shift.mean())  # Shift trigger index by average amount.

    # Cut signals.
    for n in range(N):

        nsignals[:, n] = signals[shift[n]:-(totalcut - shift[n]), n]

    # Find the new triggrer index
    tidx += fzero(nsignals.mean(axis=1)[(tidx - 10):(tidx + 10)]) - 10

    return nsignals, tidx


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
