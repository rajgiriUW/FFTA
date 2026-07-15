import numpy as np
from scipy.signal import argrelextrema


def get_peaks(x):
    """
    :param x:
    :type x:

    :returns: tuple (maxpeaks[0], minpeaks[0])
        WHERE
        [type] maxpeaks[0] is...
        [type] minpeaks[0] is...
    """
    maxpeaks = argrelextrema(x, np.greater, order=2)
    minpeaks = argrelextrema(x, np.less, order=2)

    return maxpeaks[0], minpeaks[0]
