"""pixel_utils: Backwards-compatibility shim. Use ffsignal_utils instead."""
# This package is retained so that existing code using
# `from ffta.pixel_utils import ...` or `from ffta import pixel_utils`
# continues to work without modification.
# Will be removed in the next major release.

import warnings

warnings.warn(
    "ffta.pixel_utils is deprecated and will be removed in the next major "
    "release; use ffta.ffsignal_utils instead.",
    category=DeprecationWarning,
    stacklevel=2,
)

__all__ = ['badpixels', 'dwavelet', 'fitting', 'load', 'noise', 'parab',
           'peakdetect', 'plot', 'tfp_calc']

from . import badpixels  # noqa: F401
from . import dwavelet  # noqa: F401
from . import fitting  # noqa: F401
from . import load  # noqa: F401
from . import noise  # noqa: F401
from . import parab  # noqa: F401
from . import peakdetect  # noqa: F401
from . import plot  # noqa: F401
from . import tfp_calc  # noqa: F401
