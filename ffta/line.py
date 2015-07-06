"""line.py: Contains line class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np
from ffta import pixel


class Line(object):
    """
    Signal Processing to Extract Time-to-First-Peak.

    This class is a container for pixels in a line. Since the AFM scans are in
    lines, tha data taken is grouped in lines. This class takes the line data
    and passes it along to pixels.

    Parameters
    ----------
    signal_array : (n_points, n_signals) array_like
        2D real-valued signal array, corresponds to a line
    params : dict
        Includes parameters for processing. The list of parameters is:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)
        drive_freq = float (in Hz)

        roi = float (in seconds)
        window = string (see documentation of scipy.signal.get_window)
        bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
        filter_bandwidth = float (in Hz)
        n_taps = integer (default: 999)
        wavelet_analysis = bool (0: Hilbert method, 1: Wavelet Method)
        wavelet_parameter = int (default: 5)
        recombination = bool (0: FF-trEFMm, 1: Recombination)
    n_pixels : int
        Number of pixels in a line.

    Attributes
    ----------
    n_points : int
        Number of points in a signal.
    n_signals : int
        Number of signals in a line.
    inst_freq : (n_points, n_pixels) array_like
        Instantenous frequencies of the line.
    tfp : (n_pixels,) array_like
        Time from trigger to first-peak, in seconds.
    shift : (n_pixels,) array_like
        Frequency shift from trigger to first-peak, in Hz.

    See Also
    --------
    pixel: Pixel processing for FF-trEFM data.
    simulate: Simulation for synthetic FF-trEFM data.
    scipy.signal.get_window: Windows for signal processing.

    Examples
    --------
    >>> from ffta import line, utils
    >>>
    >>> signal_file = '../data/SW_0000.ibw'
    >>> params_file = '../data/parameters.cfg'
    >>>
    >>> signal_array = utils.load.signal(signal_file)
    >>> n_pixels, params = utils.load.configuration(params_file)
    >>>
    >>> l = line.Line(signal_array, params, n_pixels)
    >>> tfp, shift, inst_freq = l.analyze()

    """

    def __init__(self, signal_array, params, n_pixels):

        # Pass inputs to the object.
        self.signal_array = signal_array
        self.n_pixels = n_pixels
        self.params = params

        # Initialize tFP and shift arrays.
        self.tfp = np.empty(n_pixels)
        self.shift = np.empty(n_pixels)
        self.inst_freq = np.empty((signal_array.shape[0], n_pixels))

        return

    def analyze(self):
        """
        Analyzes the line with the given method.

        Returns
        -------
        tfp : (n_pixels,) array_like
            Time from trigger to first-peak, in seconds.
        shift : (n_pixels,) array_like
            Frequency shift from trigger to first-peak, in Hz.
        inst_freq : (n_points, n_pixels) array_like
            Instantenous frequencies of the line.

        """

        # Split the signal array into pixels.
        pixel_signals = np.split(self.signal_array, self.n_pixels, axis=1)

        # Iterate over pixels and return tFP and shift arrays.
        for i, pixel_signal in enumerate(pixel_signals):

            p = pixel.Pixel(pixel_signal, self.params)
            (self.tfp[i], self.shift[i], self.inst_freq[:, i]) = p.analyze()

        return (self.tfp, self.shift, self.inst_freq)
