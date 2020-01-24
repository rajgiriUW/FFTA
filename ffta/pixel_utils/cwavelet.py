"""cwavelet.py: contains functions used in CWT calculations."""

import numpy as np
from scipy import fftpack as spf

PI2 = 2 * np.pi


def normalization(s, dt):

    return np.sqrt(2 * np.pi * s / dt)


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

    for i in range(s.shape[0]):

        n = normalization(s[i], dt)
        wavelet[i, pos] = n * p * np.exp(-(s[i] * w[pos] - w0) ** 2 / 2.0)

    return wavelet


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

    for i in range(w.shape[0]):

        if i <= N2:

            w[i] = (2 * np.pi * i) / (N * dt)

        else:

            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w


def cwt(x, dt, scales, p=2):
    """Continuous Wavelet Tranform.

    :Parameters:
       x : 1d array_like object
          data
       dt : float
          time step
       scales : 1d array_like object
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter ('omega0' for morlet, 'm' for paul
          and dog)

    :Returns:
       X : 2d numpy array
          transformed data
    """

    x -= np.mean(x)
    w = angularfreq(N=x.shape[0], dt=dt)

    wft = morletft(s=scales, w=w, w0=p, dt=dt)

    X = np.empty((wft.shape[0], wft.shape[1]), dtype=np.complex128)

    x_ft = spf.fft(x)

    for i in range(X.shape[0]):

        X[i] = spf.ifft(x_ft * wft[i])

    return X
