#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cwavelet.py: contains functions used in CWT calculations

This is the mlpy continuous.py without the extraneous functions (so it compiles)
"""

import numpy as np
from scipy import signal as sig

PI2 = 2 * np.pi


def normalization(s, dt):
    return np.sqrt((2 * np.pi * s) / dt)


def morletft(s, w, w0, dt):
    """Fourier tranformed morlet function.
    
    Input
      * *s*    - scales
      * *w*    - angular frequencies
      * *w0*   - omega0 (frequency)
      * *dt*   - time step
    Output
      * (normalized) fourier transformed morlet function
    """
    
    p = np.pi**(-1.0/4.0)  
    wavelet = np.zeros((s.shape[0], w.shape[0]))
    pos = w > 0

    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        wavelet[i][pos] = n * p * np.exp(-(s[i] * w[pos] - w0)**2 / 2.0)
        
    return wavelet

    
def paulft(s, w, order, dt):
    """Fourier tranformed paul function.
    
    Input
      * *s*     - scales
      * *w*     - angular frequencies
      * *order* - wavelet order
      * *dt*    - time step
    Output
      * (normalized) fourier transformed paul function
    """
    
    p = 2.0**order / np.sqrt(order * gsl.sf_fact((2 * order) - 1))
    wavelet = np.zeros((s.shape[0], w.shape[0]))
    pos = w > 0

    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        tmp = s[i] * w[pos]
        wavelet[i][pos] = n * p * tmp**order * np.exp(-tmp)

    return wavelet


def dogft(s, w, order, dt):
    """Fourier tranformed DOG function.
    
    Input
      * *s*     - scales
      * *w*     - angular frequencies
      * *order* - wavelet order
      * *dt*    - time step
    Output
      * (normalized) fourier transformed DOG function
    """

    p =  - (0.0 + 1.0j)**order  / np.sqrt(gsl.sf_gamma(order + 0.5))    
    wavelet = np.zeros((s.shape[0], w.shape[0]), dtype = np.complex128)
    
    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        h = s[i] * w
        wavelet[i] = n * p * h**order * np.exp(-h**2 / 2.0)

    return wavelet


def angularfreq(N, dt):
    """Compute angular frequencies.

    :Parameters:   
       N : integer
          number of data samples
       dt : float
          time step
    
    :Returns:
        angular frequencies : 1d numpy array
    """

    # See (5) at page 64.
    
    N2 = N / 2.0
    w = np.empty(N)

    for i in range(w.shape[0]):
        if i <= N2:
            w[i] = (2 * np.pi * i) / (N * dt)
        else:
            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w


def autoscales(N, dt, dj, wf, p):
     """Compute scales as fractional power of two.

     :Parameters:
        N : integer
           number of data samples
        dt : float
           time step
        dj : float
           scale resolution (smaller values of dj give finer resolution)
        wf : string
           wavelet function ('morlet', 'paul', 'dog')
        p : float
           omega0 ('morlet') or order ('paul', 'dog')
     
     :Returns:
        scales : 1d numpy array
           scales
     """
     
     if wf == 'dog':
         s0 = (dt * np.sqrt(p + 0.5)) / np.pi
     elif wf == 'paul':
         s0 = (dt * ((2 * p) + 1)) / (2 * np.pi)
     elif wf == 'morlet':
         s0 = (dt * (p + np.sqrt(2 + p**2))) / (2 * np.pi)
     else:
         raise ValueError('wavelet function not available')
     
     #  See (9) and (10) at page 67.

     J = np.floor(dj**-1 * np.log2((N * dt) / s0))
     s = np.empty(J + 1)
    
     for i in range(s.shape[0]):
         s[i] = s0 * 2**(i * dj)
    
     return s


def fourier_from_scales(scales, wf, p):
    """Compute the equivalent fourier period
    from scales.
    
    :Parameters:
       scales : list or 1d numpy array
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter ('omega0' for morlet, 'm' for paul
          and dog)
    
    :Returns:
       fourier wavelengths
    """

    scales_arr = np.asarray(scales)

    if wf == 'dog':
        return  (2 * np.pi * scales_arr) / np.sqrt(p + 0.5)
    elif wf == 'paul':
        return  (4 * np.pi * scales_arr) / float((2 * p) + 1)
    elif wf == 'morlet':
        return  (4 * np.pi * scales_arr) / (p + np.sqrt(2 + p**2))
    else:
        raise ValueError('wavelet function not available')


def scales_from_fourier(f, wf, p):
    """Compute scales from fourier period.

    :Parameters:
       f : list or 1d numpy array
          fourier wavelengths
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter ('omega0' for morlet, 'm' for paul
          and dog)
    
    :Returns:
       scales
    """

    f_arr = np.asarray(f)

    if wf == 'dog':
        return  (f_arr * np.sqrt(p + 0.5)) / (2 * np.pi)
    elif wf == 'paul':
        return (f_arr * ((2 * p) + 1)) / (4 * np.pi)
    elif wf == 'morlet':
        return (f_arr * (p + np.sqrt(2 + p**2))) / (4 * np.pi)
    else:
        raise ValueError('wavelet function not available')


def cwt(x, dt, scales, wf='dog', p=2):
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
   

    x_arr = np.asarray(x) - np.mean(x)
    scales_arr = np.asarray(scales)

    if x_arr.ndim is not 1:
        raise ValueError('x must be an 1d numpy array of list')

    if scales_arr.ndim is not 1:
        raise ValueError('scales must be an 1d numpy array of list')

    w = angularfreq(N=x_arr.shape[0], dt=dt)
        
    if wf == 'dog':
        wft = dogft(s=scales_arr, w=w, order=p, dt=dt)
    elif wf == 'paul':
        wft = paulft(s=scales_arr, w=w, order=p, dt=dt)
    elif wf == 'morlet':
        wft = morletft(s=scales_arr, w=w, w0=p, dt=dt)
    else:
        raise ValueError('wavelet function is not available')
    
    X_ARR = np.empty((wft.shape[0], wft.shape[1]), dtype=np.complex128)
    
    x_arr_ft = np.fft.fft(x_arr)

    for i in range(X_ARR.shape[0]):
        X_ARR[i] = np.fft.ifft(x_arr_ft * wft[i])
    
    return X_ARR


def icwt(X, dt, scales, wf='dog', p=2):
    """Inverse Continuous Wavelet Tranform.
    The reconstruction factor is not applied.

    :Parameters:
       X : 2d array_like object
          transformed data
       dt : float
          time step
       scales : 1d array_like object
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter

    :Returns:
       x : 1d numpy array
          data
    """  
    
    X_arr = asarray(X)
    scales_arr = asarray(scales)

    if X_arr.shape[0] != scales_arr.shape[0]:
        raise ValueError('X, scales: shape mismatch')

    # See (11), (13) at page 68
    X_ARR = empty_like(X_arr)
    for i in range(scales_arr.shape[0]):
        X_ARR[i] = X_arr[i] / sqrt(scales_arr[i])
    
    x = sum(real(X_ARR), axis=0)
   
    return x

    
