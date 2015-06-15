"""cwavelet.py: contains functions used in CWT calculations."""

import numpy as np
from scipy import fftpack as spf
from numba import jit

PI2 = 2 * np.pi


@jit
def normalization(s, dt):

    return np.sqrt(2 * np.pi * s / dt)


@jit
def morletft(s, w, w0, dt):
    """Fourier tranformed Morlet Function.

    Parameters
    ----------
    s : array
        Wavelet scales

    w : array
        Angular frequencies

    w0 : float
        Omega0

    dt : float
        Time step

    Returns
    -------
    wavelet : array
        Normalized Fourier-transformed Morlet Function

    """

    p = np.pi ** -0.25
    wavelet = np.zeros((s.shape[0], w.shape[0]))
    pos = (w > 0)

    for i in xrange(s.shape[0]):

        n = normalization(s[i], dt)
        wavelet[i, pos] = n * p * np.exp(-(s[i] * w[pos] - w0) ** 2 / 2.0)

    return wavelet


@jit
def angularfreq(N, dt):
    """Compute angular frequencies.

    Parameters
    ----------
    N : integer
      Number of data samples

    dt : float
      Time step

    Returns
    -------
    w : array
        Angular frequencies

    """

    N2 = N / 2.0
    w = np.empty(N)

    for i in xrange(w.shape[0]):

        if i <= N2:

            w[i] = (2 * np.pi * i) / (N * dt)

        else:

            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w


@jit
def cwt(x, dt, s, w0=2):
    """Continuous Wavelet Transform.

    Parameters
    ----------
    x : array
        Data samples to transform

    dt : float
        Time step

    s : array
      Wavelet scales

    w0 : float
       Omega0

    Returns
    -------
    X : array
        Transformed data

    """

    x -= np.mean(x)
    w = angularfreq(x.shape[0], dt)
    wft = morletft(s, w, w0, dt)

    X = np.zeros(wft.shape, dtype=np.complex128)

    x_ft = spf.fft(x)

    for i in xrange(wft.shape[0]):

        X[i, :] = spf.ifft(x_ft * wft[i])

    return X
