"""fitting.py: Routines for fitting cantilever data to extract tFP/shift"""
import numpy as np
from scipy.optimize import minimize

'''
Fit Equations
'''


def ddho_freq_product(t, A, tau1, tau2):
    '''
    Uses a product of exponentials as the functional form
    
    :param t:
    :type t:
    
    :param A:
    :type A:
    
    :param tau1:
    :type tau1:
    
    :param tau2:
    :type tau2:
    
    :returns:
    :rtype:
    '''
    decay = np.exp(-t / tau1) - 1
    relaxation = -1 * np.exp(-t / tau2)

    return A * decay * relaxation


def ddho_freq_sum(t, A1, A2, tau1, tau2):
    '''
    Uses a sum of exponentials as the functional form
    
    :param t:
    :type t:
    
    :param A1:
    :type A1:
    
    :param A2:
    :type A2:
    
    :param tau1:
    :type tau1:
    
    :param tau2:
    :type tau2:
    
    :returns:
    :rtype:
    '''
    decay = np.exp(-t / tau1) - 1
    relaxation = -1 * np.exp(-t / tau2)

    return A1 * decay + A2 * relaxation


def cut_exp(t, A, y0, tau):
    '''
    Uses a single exponential for the case of no drive
    
    :param t:
    :type t:
    
    :param A:
    :type A:
    
    :param y0:
    :type y0:
    
    :param tau:
    :type tau:
    
    :returns:
    :rtype:
    '''
    return y0 + A * np.exp(-t / tau)


def ddho_phase(t, A, tau1, tau2):
    """
    :param t:
    :type t:
    
    :param A:
    :type A:
    
    :param tau1:
    :type tau1:
    
    :param tau2:
    :type tau2:
    
    :returns:
    :rtype:
    
    """
    prefactor = tau2 / (tau1 + tau2)

    return A * tau1 * np.exp(-t / tau1) * (-1 + prefactor * np.exp(-t / tau2)) + A * tau1 * (1 - prefactor)


'''
Fit functions

Product: product of two exponential functions (default)
Sum: sum of two exponential functions
Exp: Single exponential decay
Ringdown: Same as Exp but with different bounds
Phase: integrated product of two exponential functions

'''


def fit_product(Q, drive_freq, t, inst_freq):
    """
    :param Q:
    :type Q:
    
    :param drive_freq:
    :type drive_freq:
    
    :param t:
    :type t:
    
    :param inst_freq:
    :type inst_freq:
    
    :returns:
    :rtype:
    
    """
    # Initial guess for relaxation constant.
    inv_beta = Q / (np.pi * drive_freq)

    # Cost function to minimize.
    cost = lambda p: np.sum((ddho_freq_product(t, *p) - inst_freq) ** 2)

    # bounded optimization using scipy.minimize
    pinit = [inst_freq.min(), 1e-4, inv_beta]

    popt = minimize(cost, pinit, method='TNC', options={'disp': False},
                    bounds=[(-10000, -1.0),
                            (5e-7, 0.1),
                            (1e-5, 0.1)])

    return popt.x


def fit_product_unbound(Q, drive_freq, t, inst_freq):
    """
    Fit without any bound constraints
    
    :param Q:
    :type Q:
    
    :param drive_freq:
    :type drive_freq:
    
    :param t:
    :type t:
    
    :param inst_freq:
    :type inst_freq:
    
    :returns:
    :rtype:
    """

    # Initial guess for relaxation constant.
    inv_beta = Q / (np.pi * drive_freq)

    # Cost function to minimize.
    cost = lambda p: np.sum((ddho_freq_product(t, *p) - inst_freq) ** 2)

    # bounded optimization using scipy.minimize
    pinit = [inst_freq.min(), 1e-4, inv_beta]

    popt = minimize(cost, pinit, method='TNC', options={'disp': False})

    return popt.x


def fit_sum(Q, drive_freq, t, inst_freq):
    """
    Fit without any bound constraints
    
    :param Q:
    :type Q:
    
    :param drive_freq:
    :type drive_freq:
    
    :param t:
    :type t:
    
    :param inst_freq:
    :type inst_freq:
    
    :returns:
    :rtype:
    """
    # Initial guess for relaxation constant.
    inv_beta = Q / (np.pi * drive_freq)

    # Cost function to minimize.
    cost = lambda p: np.sum((ddho_freq_sum(t, *p) - inst_freq) ** 2)

    # bounded optimization using scipy.minimize
    pinit = [inst_freq.min(), inst_freq.min(), 1e-4, inv_beta]

    popt = minimize(cost, pinit, method='TNC', options={'disp': False},
                    bounds=[(-10000, -1.0),
                            (-10000, -1.0),
                            (5e-7, 0.1),
                            (1e-5, 0.1)])

    return popt.x


def fit_exp(t, inst_freq):
    """
    :param t:
    :type t:
    
    :param inst_freq:
    :type inst_freq:
    
    :return:
    :rtype:
    """
    # Cost function to minimize.
    cost = lambda p: np.sum((cut_exp(t, *p) - inst_freq) ** 2)

    pinit = [inst_freq.max() - inst_freq.min(), inst_freq.min(), 1e-4]

    #    popt = minimize(cost, pinit, method='TNC', options={'disp': False},
    #                    bounds=[(1e-5, 1000),
    #                            (np.abs(inst_freq.min()) * -2, np.abs(inst_freq.max()) * 2),
    #                             (1e-6, 0.1)])

    popt = minimize(cost, pinit, method='TNC', options={'disp': False})

    return popt.x


def fit_ringdown(t, cut):
    """
    
    :param t:
    :type t:
    
    :param cut:
    :type cut:
    
    :returns:
    :rtype:
    """
    # Cost function to minimize. Faster than normal scipy optimize or lmfit
    cost = lambda p: np.sum((cut_exp(t, *p) - cut) ** 2)

    pinit = [cut.max() - cut.min(), cut.min(), 1e-4]

    popt = minimize(cost, pinit, method='TNC', options={'disp': False},
                    bounds=[(0, 5 * (cut.max() - cut.min())),
                            (0, cut.min()),
                            (1e-8, 1)])
    return popt.x


def fit_phase(Q, drive_freq, t, phase):
    """
    
    :param Q:
    :type Q:
    
    :param drive_freq:
    :type drive_freq:
    
    :param t:
    :type t:
    
    :param phase:
    :type phase:
    
    :returns:
    :rtype:
    """
    # Initial guess for relaxation constant.
    inv_beta = Q / (np.pi * drive_freq)

    # Cost function to minimize.
    cost = lambda p: np.sum((ddho_phase(t, *p) - phase) ** 2)

    # bounded optimization using scipy.minimize
    pinit = [phase.max() - phase.min(), 1e-4, inv_beta]

    maxamp = phase[-1] / (1e-4 * (1 - inv_beta / (inv_beta + 1e-4)))

    popt = minimize(cost, pinit, method='TNC', options={'disp': False},
                    bounds=[(0, 5 * maxamp),
                            (5e-7, 0.1),
                            (1e-5, 0.1)])

    return popt.x
