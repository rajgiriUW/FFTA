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
    signal_array : (n_signals, n_points) array_like
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
        self.n_pixels = int(n_pixels)
        self.params = params

        # Initialize tFP and shift arrays.
        self.tfp = np.empty(self.n_pixels)
        self.shift = np.empty(self.n_pixels)
        self.inst_freq = np.empty((signal_array.shape[1], self.n_pixels))
        
        self.avgs_per_pixel = int(signal_array.shape[0]/self.n_pixels)

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

        # for pycroscopy the array is arranged differently (n_pixels, n_points). 
        # This code preserves existing functionality but the format should be deprecated
        try:
      
            pixel_signals = np.split(self.signal_array, self.n_pixels, axis=0)

        # exception = non-pycroscopy format
        except:

            self.inst_freq = np.empty((self.signal_array.shape[0], self.n_pixels))
            self.avgs_per_pixel = int(self.signal_array.shape[1]/self.n_pixels)
            
            pixel_signals = np.split(self.signal_array.transpose(), self.n_pixels, axis=0)

        # Iterate over pixels and return tFP and shift arrays.
        for i, pixel_signal in enumerate(pixel_signals):

            p = pixel.Pixel(pixel_signal, self.params)
            
            (self.tfp[i], self.shift[i], self.inst_freq[:, i]) = p.analyze()

        return (self.tfp, self.shift, self.inst_freq)

    def pixel_wise_avg(self, avgs_per_pixel=1):
        """
        Averages the line per pixel and saves the result as signal_avg_array
        
        Returns
        -------
        signal_avg_array : (n_points, n_pixels) numpy array
            Returns signal_averaged time-domain signal at each pixel
        """
        
        #avgs_per_pixel 
        
        return #signal_avg_array