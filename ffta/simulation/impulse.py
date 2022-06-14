# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:59:18 2022

@author: Raj
"""

from .mechanical_drive import MechanicalDrive
from .utils.load import params_from_experiment as load_parm


def impulse(can_path, param_cfg=None, voltage=None, plot=False, **kwargs):
    '''
    Generates the impulse force response for a given cantilever given some particular
    parameters.
    
    :param can_path: Path to cantilever parameters.txt file 
    :type can_path: str

    :param params_cfg: Path to parameters.cfg file (from FFtrEFM experiment, in the data folder)
    :type params_cfg: string
        
    :param voltage: Voltage to simulate the impulse response at
    :type param: float
    
    :param plot: Whether the plot the processed instantaneous frequency/Pixel response
    :type param: bool
    
    :param kwargs: Passed to pixel.analyze (n_taps, method, etc)
    :type kwargs:
    
    '''
    if isinstance(can_path, str) and isinstance(param_cfg, str):
        can_params, force_params, sim_params, _, parms = load_parm(can_path, param_cfg)
    elif isinstance(can_path, list) and isinstance(param_cfg, dict):  # passing dictionaries directly
        can_params, force_params, sim_params = can_path
        parms = param_cfg
        can_params['drive_freq'] = parms['drive_freq']
        can_params['res_freq'] = parms['drive_freq']
        sim_params['trigger'] = parms['trigger']
        sim_params['total_time'] = parms['total_time']
        sim_params['sampling_rate'] = parms['sampling_rate']
    else:
        raise ValueError('Missing correct parameters path')
    # Use input voltage to calculate expected frequency shift with which to simulate
    if voltage:
        k = can_params['k']  # N/m
        resf = can_params['res_freq']
        dFdz = force_params['dFdz']  # I think these values are off by 1000 somewhere
        # dFdz = dFdz * 4 *k / resf , note dFdz is already saved scaled!!!

        delta_f = resf / (4 * k) * dFdz * voltage ** 2  # Marohn and others
        force_params['delta_freq'] = delta_f
        force_params['es_force'] = (voltage / (force_params['lift_height'] * 1e-9)) * 1.6e-19
        # #omega0 = resf * np.ones(len(voltage)) - delta_f

    force_params['tau'] = 1e-8  # "impulse"
    cant = MechanicalDrive(can_params, force_params, sim_params)
    Z, _ = cant.simulate()
    pix = cant.analyze(plot=plot, **kwargs)

    return pix, cant
