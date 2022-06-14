# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:27:55 2021

@author: Raj
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import medfilt

from ffta.pixel_utils.load import configuration
from .mechanical_drive import MechanicalDrive
from .utils.load import params_from_experiment as load_parm
from .utils.load import simulation_configuration as load_sim_config


def cal_curve(can_path, param_cfg, taus_range=[-7, -3], plot=True, **kwargs):
    '''
    Generates a calibration curve for a given cantilever given some particular
    parameters.

    Ideally you would have a tip parameters file as well.

    Usage:
    ------
    >>> param_cfg = 'path'
    >>> can_params = 'path'
    >>> taus, tfp, spl = cal_curve(param_cfg, can_params)
    >>> from matplotlib import pyplot as plt
    >>> plt.plot(tfp, taus, 'bX-')
        
    If you want to change the fit parameters per tau
    taus, tfp, spl = cal_curve(param_cfg, can_params, roi=0.001, n_taps=199)

    :param can_path: Path to cantilever parameters.txt file 
    :type can_path: str

    :param params_cfg: Path to parameters.cfg file (from FFtrEFM experiment, in the data folder)
    :type params_cfg: string
        
    :param taus_range: taus_range to set a range for the simulations, taken as [low, high]
    :type taus_range: ndarray (2-index array), optional
        
    :param plot: Plots the last taus vs tfps for verification
    :type plot: bool, optional
        
    :param kwargs:
    :type kwargs:
    
    :returns: tuple (taus, tfps, spl)
        WHERE
        ndarray taus is the single exponential taus that were simulated
        ndarray tfps is the measured time to first peaks
        UnivariateSpline spl is spline object of the calibration curve. To scale an image, type spl(x)
            
    '''
    if isinstance(can_path, str):
        can_params, force_params, sim_params, _, parms = load_parm(can_path, param_cfg)
    elif isinstance(can_path, tuple):
        can_params, force_params, sim_params = load_sim_config(can_path)
        _, parms = configuration(param_cfg)
        can_params['drive_freq'] = parms['drive_freq']
        can_params['res_freq'] = parms['drive_freq']
        sim_params['trigger'] = parms['trigger']
        sim_params['total_time'] = parms['total_time']
        sim_params['sampling_rate'] = parms['sampling_rate']

    if len(taus_range) != 2 or (taus_range[1] <= taus_range[0]):
        raise ValueError('Range must be ascending and 2-items')

    # Check if given as log or actual values
    if taus_range[0] < 0 or taus_range[1] < 0:
        _rlo = taus_range[0]
        _rhi = taus_range[1]

    else:
        _rlo = np.log10(taus_range[0])
        _rhi = np.log10(taus_range[1])

    taus = np.logspace(_rlo, _rhi, 50)
    tfps = []

    for t in taus:
        force_params['tau'] = t
        cant = MechanicalDrive(can_params, force_params, sim_params)
        Z, _ = cant.simulate()
        try:
            pix = cant.analyze(plot=False, **kwargs)
            tfps.append(pix.tfp)
        except:
            print('Error', t)

    # sort the arrays
    taus = taus[np.argsort(tfps)]
    tfps = np.sort(tfps)

    # Splines work better on shorter lengthscales
    taus = np.log(taus)
    tfps = np.log(tfps)

    # Error corrections
    # negative x-values (must be monotonic for spline)
    dtfp = np.diff(tfps)
    tfps = np.array(tfps)
    taus = np.array(taus)
    tfps = np.delete(tfps, np.where(dtfp < 0)[0])
    taus = np.delete(taus, np.where(dtfp < 0)[0])

    # "hot" pixels in the cal-curve
    hotpixels = np.abs(taus - medfilt(taus))
    taus = np.delete(taus, np.where(hotpixels > 0))
    tfps = np.delete(tfps, np.where(hotpixels > 0))

    # Negative slopes
    neg_slope = np.diff(taus) / np.diff(tfps)
    while any(np.where(neg_slope < 0)[0]):
        tfps = np.delete(tfps, np.where(neg_slope < 0)[0])
        taus = np.delete(taus, np.where(neg_slope < 0)[0])
        neg_slope = np.diff(taus) / np.diff(tfps)

    # Infinite slops (tfp saturation at long taus)
    while (any(np.where(neg_slope == np.inf)[0])):
        tfps = np.delete(tfps, np.where(neg_slope == np.inf)[0])
        taus = np.delete(taus, np.where(neg_slope == np.inf)[0])
        neg_slope = np.diff(taus) / np.diff(tfps)

    try:
        spl = ius(tfps, taus, k=4)
    except:
        print('=== Error generating cal-curve. Check manually ===')
        spl = None
        print(taus)
        print(tfps)

    if plot:
        pix.plot()
        fig, ax = plt.subplots(facecolor='white')
        ax.loglog(np.exp(tfps), np.exp(taus), 'bX-')
        try:
            ax.loglog(np.exp(tfps), np.exp(spl(tfps)), 'r--')
        except:
            pass
        ax.set_xlabel('$t_{fp}$ (s)')
        ax.set_ylabel(r'$\tau$ (s)')
        ax.set_title('Calibration curve')

    # Save Calibration Curve
    df = pd.DataFrame(index=taus, data=tfps)
    df = df.rename(columns={0: 'tfps'})
    df.index.name = 'taus'
    df.to_csv('Calibration_Curve.csv')

    print('Do not forget that the spline is in log-space')

    return taus, tfps, spl


def reconstruct_cal_curve(csv_file):
    '''
    Reconstruct the calibration curve from a saved CSV
    
    :param csv_file: path to CSV file
    :type csv_file: str
    
    :returns: tuple (taus, tfps, spl)
        WHERE
        ndarray taus is the single exponential taus that were simulated
        ndarray tfps is the measured time to first peaks
        UnivariateSpline spl is spline object of the calibration curve. To scale an image, type spl(x)
               
    '''

    df = pd.read_csv(csv_file)

    taus = df['taus']
    tfps = df['tfps']
    spl = ius(tfps, taus, k=4)

    print('Note that taus, tfps, and cal_curve are in log-space')

    return taus, tfps, spl
