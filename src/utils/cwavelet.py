import numpy as np


def normalization(s, dt):

    return np.sqrt((2 * np.pi * s) / dt)


def morletft(s, w, w0, dt):
    """
    Fourier transformed Morlet function.

    Parameters
    ----------
    s: array, shape = [n]
        scales

    w: array, shape = [m]
        angular frequencies

    w0: float
        omega0, frequency

    dt: float
        time step

    Returns
    -------
    wavelet: array, shape = [n, m]
        (normalized) fourier transformed morlet function

    """

    p = np.pi ** (-1.0/4.0)
    wavelet = np.zeros((s.shape[0], w.shape[0]))
    pos = w > 0

    for i in range(s.shape[0]):

        n = normalization(s[i], dt)
        wavelet[i][pos] = n * p * np.exp(-(s[i] * w[pos] - w0) ** 2 / 2.0)

    return wavelet


def angularfreq(N, dt):
    """
    Compute angular frequencies.

    Parameters
    ----------
    N: int
        number of data samples

    dt: float
        time step

    Returns
    -------
    w: array, shape = [N]
        angular frequencies

    """
    w = np.empty(N)

    for i in range(w.shape[0]):

        if i <= (N / 2.0):

            w[i] = (2 * np.pi * i) / (N * dt)

        else:

            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w


def autoscales(N, dt, dj, w0):
    """
    Compute scales as fractional power of two.

    Parameters
    ----------
    N: int
        number of data samples

    dt: float
        time step

    dj: float
        scale resolution (smaller values of dj give finer resolution)

    w0: float
        omega0 for Morlet wavelet

    Returns
    -------
    scales: array, shape = [N]
        scales

    """

    s0 = (dt * (w0 + np.sqrt(2 + w0 ** 2))) / (2 * np.pi)

    J = np.floor(dj ** -1 * np.log2((N * dt) / s0))
    scales = np.empty(J + 1)

    for i in range(scales.shape[0]):

        scales[i] = s0 * 2 ** (i * dj)

    return scales


def fourier_from_scales(scales, w0):
    """
    Compute the equivalent Fourier period from scales for Morlet wavelet.

    Parameters
    ----------
    scales: array, shape = [N]
        scales

    w0: float
        omega0

    Returns
    -------
    fourier_lambdas: array, shape = [N]
        Fourier wavelengths

    """

    fourier_lambdas = (4 * np.pi * scales) / (w0 + np.sqrt(2 + w0 ** 2))

    return fourier_lambdas


def scales_from_fourier(fourier_lambdas, w0):
    """
    Compute scales from Fourier period for Morlet wavelet.

    Parameters
    ----------
    fourier_lambdas: array, shape = [N]
        Fourier wavelengths

    w0: float
        omega0

    Returns
    -------
    scales: array, shape = [N]
        scales

    """

    scales = (fourier_lambdas * (w0 + np.sqrt(2 + w0 ** 2))) / (4 * np.pi)

    return scales


def cwt(x, dt, scales, w0=2.):
    """
    Continuous wavelet transform with Morlet wavelet.

    Parameters
    ----------
    x: array, shape = [m]
        data

    dt: float
        time step

    scales : array, shape = [n]
        scales

    w0: float
        omega0 for Morlet wavelet.

    Returns
    -------
    X : array, shape = [m, n]
        transformed data

    """

    x -= x.mean()

    w = angularfreq(x.shape[0], dt)
    wft = morletft(scales, w, w0, dt)

    X = np.empty((wft.shape[0], wft.shape[1]), dtype=np.complex128)

    x_ft = np.fft.fft(x)

    for i in range(X.shape[0]):

        X[i] = np.fft.ifft(x_ft * wft[i])

    return X
