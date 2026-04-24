"""pixel.py: Backwards-compatibility shim. Use ffsignal.py instead."""
# This module is retained so that existing code using
# `from ffta.pixel import Pixel` or `from ffta import pixel`
# continues to work without modification.
# Will be removed in the next major release.

from ffta.ffsignal import FFSignal, Pixel

__all__ = ['FFSignal', 'Pixel']
