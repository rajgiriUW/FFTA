# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:59:18 2022

@author: Raj
"""

import numpy as np

from .mechanical_drive import MechanicalDrive
from .utils.load import params_from_experiment as load_parm
from .utils.load import simulation_configuration as load_sim_config
from ffta.pixel_utils.load import configuration


def impulse(can_path, param_cfg, voltage = 5, **kwargs):
    '''
    Generates the impulse force response for a given cantilever given some particular
    parameters.
    
    :param can_path: Path to cantilever parameters.txt file 
    :type can_path: str

    :param params_cfg: Path to parameters.cfg file (from FFtrEFM experiment, in the data folder)
    :type params_cfg: string
        
    :param voltage: Voltage to simulate the impulse response at
    :type param: float
    
    :param kwargs: Passed to pixel.analyze (n_taps, method, etc)
    :type kwargs:
    
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

    # Use input voltage to calculate expected frequency shift with which to simulate
    # k = can_params['k'] # N/m 
    # resf = can_params['res_freq']
    # dFdz = force_params['dFdz']
    # dFdz = dFdz * 4 *k / resf
    
    # #delta_f = 0.25 * resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    # delta_f =  resf / (4*k) * dFdz * voltage**2  # Marohn and others
    # force_params['delta_freq'] = delta_f
    # #omega0 = resf * np.ones(len(voltage)) - delta_f
    
    force_params['tau'] = 1e-8 #"impulse"
    cant = MechanicalDrive(can_params, force_params, sim_params)
    Z, _ = cant.simulate()
    pix = cant.analyze(plot=False, **kwargs)
    
    return