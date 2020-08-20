# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:23:17 2020

@author: Raj
"""

import configparser


def simulation_configuration(path):
	"""
	Reads an ASCII file with relevant parameters for simulation.

	Parameters
	----------
	path : string
		Path to ASCII file.

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
	config.read(path)

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
