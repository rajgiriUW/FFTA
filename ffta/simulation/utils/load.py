# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:23:17 2020

@author: Raj
"""

import configparser
import urllib

from ffta.pixel_utils.load import cantilever_params
from ffta.pixel_utils.load import configuration


def simulation_configuration(path, is_url=False):
    """
    Reads an ASCII file with relevant parameters for simulation.

    Parameters
    ----------
    path : string
        Path to ASCII file.
    is_url : bool, optional
        Set to True if path is a URL

    Returns
    -------
    can_params : dict
        Parameters for cantilever properties. The dictionary contains:

        amp_invols = float (in m/V)
        def_invols = float (in m/V)
        soft_amp = float (in V)
        drive_freq = float (in Hz)
        res_freq = float (in Hz)
        k = float (in N/m)
        q_factor = float

    force_params : dict
        Parameters for forces. The dictionary contains:

        es_force = float (in N)
        delta_freq = float (in Hz)
        tau = float (in seconds)
        v_dc = float (in V)
        v_ac = float (in V)
        v_cpd = float (in V)
        dcdz = float (in F/m)

    sim_params : dict
        Parameters for simulation. The dictionary contains:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)

    """

    # Create a parser for configuration file and parse it.
    config = configparser.RawConfigParser()
    if not is_url:
        config.read(path)
    else:
        f = urllib.request.urlopen(path)
        fl = ''.join([l.decode('utf-8') for l in f])
        config.read_string(fl)

    sim_params = {}
    can_params = {}
    force_params = {}

    can_keys = ['amp_invols', 'def_invols', 'soft_amp', 'drive_freq',
                'res_freq', 'k', 'q_factor']
    force_keys = ['es_force', 'ac_force', 'dc_force', 'delta_freq', 'tau',
                  'v_dc', 'v_ac', 'v_cpd', 'dCdz', 'v_step']
    sim_keys = ['trigger', 'total_time', 'sampling_rate']

    for key in can_keys:

        if config.has_option('Cantilever Parameters', key):
            _str = config.get('Cantilever Parameters', key).split(';')
            can_params[key] = float(_str[0])

    for key in force_keys:

        if config.has_option('Force Parameters', key):
            _str = config.get('Force Parameters', key).split(';')
            force_params[key] = float(_str[0])

    for key in sim_keys:

        if config.has_option('Simulation Parameters', key):
            _str = config.get('Simulation Parameters', key).split(';')
            sim_params[key] = float(_str[0])

    return can_params, force_params, sim_params


def params_from_experiment(can_params_file, params_cfg):
    '''
    Generates a simulation-compatible configuration given a Cantilever Parameters file
    (typically acquired in the experiment) and a Params.cfg file saved with FFtrEFM data

    can_params : string
        Path to cantilever parameters file (from Force Calibration tab)

    params_cfg : string
        Path to parameters.cfg file (from FFtrEFM experiment, in the data folder)
    '''

    can = cantilever_params(can_params_file)
    _, par = configuration(params_cfg)

    if isinstance(params_cfg, dict):
        par = params_cfg

    can_params = {}
    can_params['amp_invols'] = can['Initial']['AMPINVOLS']
    can_params['def_invols'] = can['Initial']['DEFINVOLS']
    can_params['soft_amp'] = 0.3
    can_params['drive_freq'] = par['drive_freq']
    can_params['res_freq'] = par['drive_freq']
    can_params['k'] = can['Initial']['SpringConstant']
    can_params['q_factor'] = can['Initial']['Q']

    force_params = {}
    force_params['es_force'] = can['Differential']['ElectroForce']
    force_params['ac_force'] = can['Initial']['DrivingForce']
    force_params['dc_force'] = 0  # only for GKPFM
    force_params['delta_freq'] = can['Differential']['ResFrequency']
    force_params['tau'] = 1e-5
    force_params['dFdz'] = can['Differential']['dFdZ']
    force_params['lift_height'] = can['Initial']['LiftHeight']

    sim_params = {}
    sim_params['trigger'] = par['trigger']
    sim_params['total_time'] = par['total_time']
    sim_params['sampling_rate'] = par['sampling_rate']

    return can_params, force_params, sim_params, can, par
