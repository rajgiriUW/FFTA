import numpy as np
from scipy.optimize import fmin_tnc


def ddho_freq(t, A, tau1, tau2):

    decay = np.exp(-t / tau1)
    relaxation = np.expm1(-t / tau2)

    return -A * decay * relaxation


def fit_bounded(Q, drive_freq, t, inst_freq):

        # Initial guess for relaxation constant.
        inv_beta = Q / (np.pi * drive_freq)

        # Cost function to minimize.
        cost = lambda p: np.sum((ddho_freq(t, *p) - inst_freq) ** 2)

        # bounded optimization using scipy.minimize
        pinit = [inst_freq.min(), 1e-4, inv_beta]

        popt, n_eval, rcode = fmin_tnc(cost, pinit, approx_grad=True,disp=0,
                                       bounds=[(-10000, -1.0),
                                               (5e-7, 0.1),
                                               (1e-5, 0.1)])

        return popt
