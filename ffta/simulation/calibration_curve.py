# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:27:55 2021

@author: Raj
"""

import numpy as np

from .mechanical_drive import MechanicalDrive
from .utils.load import params_from_experiment as load_parm
from scipy.interpolate import UnivariateSpline

from matplotlib import pyplot as plt
import pandas as pd

'''
Generates a calibration curve for a given cantilever given some particular
parameters.

Ideally you would have a tip parameters file as well.

Usage:
    param_cfg = 'path'
    can_params = 'path'
    taus, tfp, spl = cal_curve(param_cfg, can_params)
    from matplotlib import pyplot as plt
    plt.plot(tfp, taus, 'bX-')
    
    If you want to change the fit parameters per tau
    taus, tfp, spl = cal_curve(param_cfg, can_params, roi=0.001, n_taps=199)
'''


def cal_curve(can_path, param_cfg, plot=True, **kwargs):
    '''
    Parameters
    ----------
    can_params : string
		Path to cantilever parameters file (from Force Calibration tab)

	params_cfg : string
		Path to parameters.cfg file (from FFtrEFM experiment, in the data folder)
        
    Returns:
    --------
    taus : ndArray
        The single exponential taus that were simulated
    tfps : ndArray
        The measured time to first peaks
    spl : UnivariateSpline
        spline object of the calibration curve. To scale an image, type spl(x)
            
	'''

    can_params, force_params, sim_params, _, parms = load_parm(can_path, param_cfg)

    taus = np.logspace(-7, -3, 50)
    tfps = []

    for t in taus:
        force_params['tau'] = t
        cant = MechanicalDrive(can_params, force_params, sim_params)
        Z, _ = cant.simulate()
        pix = cant.analyze(plot=False, **kwargs)
        tfps.append(pix.tfp)

    dtfp = np.diff(tfps)
    tfps = np.array(tfps)
    taus = np.array(taus)
    tfps = np.delete(tfps, np.where(dtfp < 0)[0])
    taus = np.delete(taus, np.where(dtfp < 0)[0])
    
    negs = np.diff(taus) / np.diff(tfps)
    tfps = np.delete(tfps, np.where(negs < 0)[0])
    taus = np.delete(taus, np.where(negs < 0)[0])

    try:
        spl = UnivariateSpline(tfps, taus)
    except:
        spl = None
        print(taus)
        print(tfps)

    if plot:
        pix.plot()
        fig, ax = plt.subplots(facecolor='white')
        ax.loglog(tfps, taus, 'bX-')
        ax.set_xlabel('$t_{fp}$ (s)')
        ax.set_ylabel(r'$\tau$ (s)')
        ax.set_title('Calibration curve')

    # Save Calibration Curve
    df = pd.DataFrame(index=taus, data=tfps)
    df = df.rename(columns={0: 'tfps'})
    df.index.name = 'taus'
    df.to_csv('Calibration_Curve.csv')

    return taus, tfps, spl
