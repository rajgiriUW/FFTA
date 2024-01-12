"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2024"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

from ffta.FFTA import FFTA


class Pixel(FFTA):
    '''
    Legacy version of the FFTA class.
    '''

    def __init__(self,
                 signal_array,
                 params={},
                 can_params={},
                 fit=True,
                 pycroscopy=False,
                 method='hilbert',
                 fit_form='product',
                 filter_amplitude=False,
                 filter_frequency=False,
                 recombination=False,
                 trigger=None,
                 total_time=None,
                 sampling_rate=None,
                 roi=None):
        super(FFTA, self).__init__(self,
                                   signal_array,
                                   params={},
                                   can_params={},
                                   fit=True,
                                   pycroscopy=False,
                                   method='hilbert',
                                   fit_form='product',
                                   filter_amplitude=False,
                                   filter_frequency=False,
                                   recombination=False,
                                   trigger=None,
                                   total_time=None,
                                   sampling_rate=None,
                                   roi=None)
