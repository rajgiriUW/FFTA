"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2024"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

from .FFTA import FFTA
import warnings

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

        warnings.warn('This class will be rename "FFTA" in next major release.',
                      category=DeprecationWarning, stacklevel=2)
        super().__init__(signal_array,
                         params=params,
                         can_params=can_params,
                         fit=fit,
                         pycroscopy=pycroscopy,
                         method='hilbert',
                         fit_form='product',
                         filter_amplitude=filter_amplitude,
                         filter_frequency=filter_frequency,
                         recombination=recombination,
                         trigger=trigger,
                         total_time=total_time,
                         sampling_rate=sampling_rate,
                         roi=roi)

