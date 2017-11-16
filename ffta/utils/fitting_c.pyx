import numpy as np
cimport numpy as np
from libc.math cimport exp
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt

def ddho_freq(double t, double A, double tau1, double tau2):

    cdef double decay = exp(-t / tau1)
    cdef double relaxation = (1 - exp(-t / tau2))
    
    return (A * decay * relaxation)

def cost(double[:] inst_freq, double[:] p):

    cdef int N = inst_freq.shape[0]
    cdef double result = 0
    cdef double[:] t = range(N) / 1.0e7
    cdef size_t i = 0

    for i in range(N):

        result += (ddho_freq(t[i], p[0], p[1], p[2]) - inst_freq[i]) ** 2

    return result

def fit_bounded(np.float64_t Q, np.float64_t drive_freq, np.ndarray t, np.ndarray inst_freq):

        # Initial guess for relaxation constant.
        cdef np.float64_t inv_beta = Q / (np.pi * drive_freq)

        # bounded optimization using scipy.minimize
        pinit = [inst_freq.min(), 1e-4, inv_beta]

        func = lambda p: cost(inst_freq, *p)

        popt, n_eval, rcode = fmin_tnc(cost, pinit, approx_grad=True, maxfun=40,
                                       bounds=[(None, -1.0),
                                               (5e-7, 0.1),
                                               (1e-5, 0.1)])

        return popt
