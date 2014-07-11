"""noise.py: Includes functions for reducing noise in a pixel."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"

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

    # Find positions of first zeros from beginning to 1% of a signal.
    for i in range(n_signals):

        index_shift[i] = fzero(signal_array[:idx, i])

    # Have all signals same length by cutting them from both ends.
    total_cut = int(index_shift.max() + index_shift.min() + 1)
    new_signal_array = np.empty((n_points - total_cut, n_signals))
    tidx -= int(index_shift.mean())  # Shift trigger index by average amount.

    # Cut signals.
    for i in range(n_signals):

        new_signal_array[:, i] = \
            signal_array[index_shift[i]:-(total_cut - index_shift[i]), i]

    signal_array = new_signal_array
    # Find the new trigger index.
    tidx += fzero(signal_array.mean(axis=1)[
        (tidx - total_cut):(tidx + total_cut)]) - total_cut

    return signal_array, tidx

def fzero(x):
    """Fast zero-finding function in lambda form."""
    return (np.where(x[:-1] * x[1:] <= 0)[0][0])
