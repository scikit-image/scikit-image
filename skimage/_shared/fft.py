"""Prefer FFTs via the new scipy.fft module when available (SciPy 1.4+)

Otherwise fall back to numpy.fft.
"""
try:
    import scipy.fft
    from scipy.fft import next_fast_len
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft

    # next_fast_len was implemented in scipy 0.18
    # In case it cannot be imported, we use the id function
    def next_fast_len(size):
        """Dummy next_fast_len that returns size unmodified."""
        return size


__all__ = ['fftmodule', 'next_fast_len']
