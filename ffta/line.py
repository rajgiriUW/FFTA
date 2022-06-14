"""line.py: Contains line class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import numpy as np

from ffta import pixel


class Line:

    def __init__(self, signal_array, params, n_pixels, pycroscopy=False):
        """
        Signal Processing to Extract Time-to-First-Peak.

        This class is a container for pixels in a line. Since the AFM scans are in
        lines, tha data taken is grouped in lines. This class takes the line data
        and passes it along to pixels.
            
        Attributes
        ----------
        n_pixels : int
            Number of points in a signal.
        inst_freq : (n_points, n_pixels) array_like
            Instantenous frequencies of the line.
        tfp : (n_pixels,) array_like
            Time from trigger to first-peak, in seconds.
        shift : (n_pixels,) array_like
            Frequency shift from trigger to first-peak, in Hz.

        See Also
        --------
        ffta.pixel: Pixel processing for FF-trEFM data.
        ffta.simulation: Simulation for synthetic FF-trEFM data.

        Examples
        --------
        >>> from ffta import line, pixel_utils
        >>>
        >>> signal_file = '../data/SW_0000.ibw'
        >>> params_file = '../data/parameters.cfg'
        >>>
        >>> signal_array = utils.load.signal(signal_file)
        >>> n_pixels, params = utils.load.configuration(params_file)
        >>>
        >>> l = line.Line(signal_array, params, n_pixels)
        >>> tfp, shift, inst_freq = l.analyze()

        :param signal_array: 2D real-valued signal array, corresponds to a line
        :type signal_array: (n_signals, n_points) array_like
        
        :param params: Includes parameters for processing. The list of parameters is:
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
            recombination = bool (0: FF-trEFM, 1: Recombination)
        :type params: dict
            
        :param n_pixels: Number of pixels in a line.
        :type n_pixels: int
        
        :param pycroscopy: Pycroscopy requires different orientation, so this corrects for this effect.
        :type pycroscopy: bool, optional

        """
        # Pass inputs to the object.
        self.signal_array = signal_array
        if pycroscopy:
            self.signal_array = signal_array.T
        self.n_pixels = int(n_pixels)
        self.params = params

        # Initialize tFP and shift arrays.
        self.tfp = np.empty(self.n_pixels)
        self.shift = np.empty(self.n_pixels)
        self.inst_freq = np.empty((self.signal_array.shape[0], self.n_pixels))

        self.avgs_per_pixel = int(self.signal_array.shape[1] / self.n_pixels)
        self.n_signals = self.signal_array.shape[0]

        return

    def analyze(self):
        """
        Analyzes the line with the given method.

        :returns: tuple (tfp, shift, inst_freq)
            WHERE
            array_like tfp is time from trigger to first-peak, in seconds in format (n_pixels,)
            array_like shift is frequency shift from trigger to first-peak, in Hz in format (n_pixels,)
            array_like inst_freq is instantaneous frequencies of the line in format (n_pixels,)
        
        """

        # Split the signal array into pixels.
        pixel_signals = np.split(self.signal_array, self.n_pixels, axis=1)

        # Iterate over pixels and return tFP and shift arrays.
        for i, pixel_signal in enumerate(pixel_signals):
            p = pixel.Pixel(pixel_signal, self.params)

            (self.tfp[i], self.shift[i], self.inst_freq[:, i]) = p.analyze()

        return (self.tfp, self.shift, self.inst_freq)

    def pixel_wise_avg(self):
        """
        Averages the line per pixel and saves the result as signal_avg_array
        This functionality is primarily used in Pycroscopy-loading functions
        
        :returns: signal_averaged time-domain signal at each pixel
        :rtype: (n_points, n_pixels) numpy array
          
        """

        self.signal_avg_array = np.empty((self.signal_array.shape[0], self.n_pixels))

        for i in range(self.n_pixels):
            avg = self.signal_array[:, i * self.avgs_per_pixel:(i + 1) * self.avgs_per_pixel]
            self.signal_avg_array[:, i] = avg.mean(axis=1)

        return self.signal_avg_array

    def clear_filter_flags(self):
        """
        Removes flags from parameters for setting filters
        """

        # self.params['window'] = 0
        self.params['bandpass_filter'] = 0

        return
