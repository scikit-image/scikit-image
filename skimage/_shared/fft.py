"""Prefer FFTs via the new scipy.fft module when available (SciPy 1.4+)

Otherwise fall back to numpy.fft.

Like numpy 1.15+ scipy 1.3+ is also using pocketfft, but a newer
C++/pybind11 version called pypocketfft
"""
try:
    import scipy.fft
    from scipy.fft import next_fast_len
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft
    from scipy.fftpack import next_fast_len

__all__ = ['fftmodule', 'next_fast_len']
