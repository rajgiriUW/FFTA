"""simulate.py: Contains Cantilever class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2020, Ginger Lab"
__email__ = "rgiri@uw.edu"
__status__ = "Production"

import numpy as np
from scipy.integrate import odeint
from scipy.signal import chirp

from .cantilever import Cantilever

# Set constant 2 * pi.
PI2 = 2 * np.pi


class BroadbandPulse(Cantilever):
	"""Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers  
	Uses a broadband pulse to excite the cantilever

	Parameters
	----------
	can_params : dict
		Parameters for cantilever properties. See Cantilever

	force_params : dict
		Parameters for forces. Beyond Cantilever, the dictionary contains:

		es_force = float (in N)
		delta_freq = float (in Hz)
		tau = float (in seconds)

	sim_params : dict
		Parameters for simulation. The dictionary contains:

		trigger = float (in seconds)
		total_time = float (in seconds)
		sampling_rate = int (in Hz)

	chirp_lo, chirp_hi : float
		Control the lower and upper frequencies for defining the chirp applied
		
	Attributes
	----------
	Z : ndarray
		ODE integration of the DDHO response

	Method
	------
	simulate(trigger_phase=180)
		Simulates the cantilever motion with excitation happening
		at the given phase.

	See Also
	--------
	pixel: Pixel processing for FF-trEFM data.
	Cantilever: base class

	Examples
	--------
	>>> from ffta.simulation import mechanical_dirve, load
	>>>
	>>> params_file = '../examples/sim_params.cfg'
	>>> params = load.simulation_configuration(params_file)
	>>>
	>>> c = mechanical_dirve.MechanicalDrive(*params)
	>>> Z, infodict = c.simulate()
	>>> c.analyze()
	>>> c.analyze(roi=0.004) # can change the parameters as desired
	"""

	def __init__(self, can_params, force_params, sim_params, chirp_lo=10, chirp_hi=1e6):
		parms = [can_params, force_params, sim_params]
		super(BroadbandPulse, self).__init__(*parms)

		t_Z = self.t_Z
		self.chirp = chirp(t_Z, chirp_lo, t_Z[-1], chirp_hi)
		self.delta_w = 0  # no trigger event

		return

	def force(self, t, t0=0, tau=0):
		"""
		Force on the cantilever at a given time. 

		Parameters
		----------
		t : float
			Time in seconds.

		Returns
		-------
		f : float
			Force on the cantilever at a given time, in N/kg.

		"""

		p = int(t * self.sampling_rate)
		n_points = int(self.total_time * self.sampling_rate)

		_ch = self.chirp[p] if p < n_points else self.chirp[-1]
		driving_force = self.f0 * _ch

		return driving_force
