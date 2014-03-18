"""utils.py: Includes utility functions."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
from scipy import linalg


def lowess(x, y, f=2./3, it=3):

    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3

    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(it):

        for i in range(n):

            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                         [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


def diff_lowess(x, y, window_length=100):

    weights = np.zeros(window_length)
    wl = window_length / 2
    weights[wl:] = (1 - (np.arange(wl) / float(wl)) ** 3) ** 3
    weights[:wl] = weights[wl:][::-1]

    bin_count = int(np.floor(len(x) / window_length))
    split_array = [y[(i * window_length):((i + 1) * window_length)]
                   for i in range(bin_count)]
    diff_y = np.empty(bin_count)

    xfit = np.arange(window_length) / (x[1] - x[0])

    for i, yfit in enumerate(split_array):

        fit = np.polyfit(xfit, yfit, 1, w=weights)
        diff_y[i] = fit[0]

    return diff_y


def parabola_fit(y, max_idx):
    """Parabola Fit around hree points to find a "true vertex".

    Uses parabola equation to fit to the peak and two surrounding points.

    Credit: Raj Giridharagopal, raj@rajgiri.net

    Parameters
    ----------
    y : array, shape =  [n]
        Array to fit maximum.
    max_idx : int
        Index of maximum in the array.

    Returns
    -------
    max_interp : float
        Interpolated maximum position.

    """

    x1 = max_idx - 1
    x2 = max_idx
    x3 = max_idx + 1

    y1 = y[x1]
    y2 = y[x2]
    y3 = y[x3]

    d = (x1 - x2) * (x1 - x3) * (x2 - x3)

    A = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) / d
    B = (x1 ** 2 * (y2 - y3) + x2 ** 2 * (y3 - y1) + x3 ** 2 * (y1 - y2)) / d

    max_interp = -B / (2 * A)

    return max_interp


def max_2d_fit(matrix):

    max_indices = np.empty(matrix.shape[1])

    for i, col in enumerate(matrix.T):

        max_indices[i] = parabola_fit(col, col.argmax())

    return max_indices
