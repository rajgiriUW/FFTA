import numpy as np
cimport numpy as np
from scipy.optimize import fmin_tnc


def ddho_freq(np.ndarray t, np.float64_t A, np.float64_t tau1, np.float64_t tau2):

    cdef np.ndarray decay = np.exp(-t / tau1)
    cdef np.ndarray relaxation = np.expm1(-t / tau2)
    cdef np.ndarray result = -A * decay * relaxation

    return result

def fit_bounded(np.float64_t Q, np.float64_t drive_freq, np.ndarray t, np.ndarray inst_freq):

        # Initial guess for relaxation constant.
        cdef np.float64_t inv_beta = Q / (np.pi * drive_freq)

        # Cost function to minimize.
        cost = lambda p: np.sum((ddho_freq(t, *p) - inst_freq) ** 2)

        # bounded optimization using scipy.minimize
        pinit = [inst_freq.min(), 1e-4, inv_beta]

        popt, n_eval, rcode = fmin_tnc(cost, pinit, approx_grad=True,
                                       bounds=[(None, -1.0),
                                               (5e-7, 0.1),
                                               (1e-5, 0.1)])

        return popt
