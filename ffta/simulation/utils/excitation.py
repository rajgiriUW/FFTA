'''
Contains excitation functions used in simulations.

These functions all assume an output scaled from 0 to 1. In MechanicalDrive,
these would all be passed to omega0 and to to Fe to change with these time-dependent
conditions, where t is passed relative to the trigger Cantilever.trigger

'''

import numpy as np


def single_exp(t, tau):
    '''
    Resonance frequency exhibits single exponential decay to a new offset

    Parameters
    ----------
    t : float or ndarray
        Time axis

    tau : float
        Time constant for decay

    '''

    return -np.expm1(-t / tau)


def bi_exp(t, tau1, tau2):
    '''
    Resonance frequency exhibits bi-exponential decay to a new offset


    '''

    A = np.exp(-t / tau1)
    B = np.exp(-t / tau2)

    return -0.5 * (A + B - 2)


def str_exp(t, tau, beta):
    '''
    Resonance frequency exhibits stretched exponential decay to a new offset
    '''

    return - (np.exp(-t / tau) ** beta - 1)


def step(t):
    '''
    Heaviside function
    '''
    return np.heaviside(t, 0)
