__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2020, Ginger Lab"
__email__ = "rgiri@uw.edu"
__status__ = "Production"

from math import pi

import numpy as np

from .cantilever import Cantilever
from .utils import excitation

# Set constant 2 * pi.
PI2 = 2 * pi


class MechanicalDrive_Arb(Cantilever):
    """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers under
    Mechanial drive (i.e. conventional DDHO) with an arbitrary resonance frequency
    pattern supplied by an array

    Simulates a DDHO under excitation with given parameters and a change to resonance
    and electrostatic force

    Time-dependent change must be specifiedby explicitly defining v_array, which
    is already scaled by the user in Hz

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
    >>> from ffta.simulation import mechanical_drive_arb
    >>> from ffta.simulation.utils import load
    >>>
    >>> params_file = '../examples/sim_params.cfg'
    >>> params = load.simulation_configuration(params_file)
    >>>
    >>> # Example applying an arbitrary frequency shift pattern
    >>> n_points = int(params[2]['total_time'] * params[2]['sampling_rate'])
    >>> v_array = np.ones(n_points) # create just a flat excitation
    >>> c = mechanical_drive_arb.MechanicalDrive_Arb(*params, v_array)
    >>> Z, _ = c.simulate()
    >>> c.analyze()
    >>> c.analyze(roi=0.004) # can change the parameters as desired

    :param can_params: Parameters for cantilever properties. See Cantilever
    :type can_params: dict
        
    :param force_params: Parameters for forces. Beyond Cantilever, the dictionary contains:
        es_force = float (in N)
        delta_freq = float (in Hz)
        tau = float (in seconds)
    :type force_params: dict
    
    :param sim_params: Parameters for simulation. The dictionary contains:
        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)  
    :type sim_params: dict
        
    :param v_array: If supplied, v_array is the time-dependent excitation to the resonance
        frequency and electrostatic force, explicitly defined.
        
        v_array must be the exact length and sampling of the desired signal
    :type v_array: ndarray, optional
    
    :param func:
    :type func: function, optional

    """

    def __init__(self, 
                 can_params, 
                 force_params, 
                 sim_params,
                 v_array):

        parms = [can_params, force_params, sim_params]
        super(MechanicalDrive_Arb, self).__init__(*parms)

        # Did user supply a voltage pulse themselves
        if len(v_array) != int(self.total_time * self.sampling_rate):
            raise ValueError('v_array must match sampling rate/length of parameters')

        self.v_array = v_array

        return

    def __gamma__(self, t):
        """
        Controls how the cantilever behaves after a trigger.
        Default operation is an exponential decay to omega0 - delta_freq with
        time constant tau.

        If supplying an explicit v_array, then this function will call the values
        in that array

        :param t: Time in seconds.
        :type t: float
            
        :returns: Value of the function at the given time.
        :rtype: float
            
        """

        p = int(t * self.sampling_rate)
        n_points = int(self.total_time * self.sampling_rate)
        t0 = self.t0

        if t >= t0:

            _g = self.v_array[p] if p < n_points else self.v_array[-1]

            return _g

        return 0

    def omega(self, t, t0, tau):
        """
        Exponentially decaying resonance frequency.

        :param t: Time in seconds.
        :type t: float
        
        :param t0: Event time in seconds.
        :type t0: float
        
        :param tau: Decay constant in the exponential function, in seconds.
        :type tau: float
            
        :returns: Resonance frequency of the cantilever at a given time, in rad/s.
        :rtype: float
            
        """

        return self.w0 + self.__gamma__(t)

    def force(self, t, t0, tau):
        """
        Force on the cantilever at a given time. It contains driving force and
        electrostatic force.

        :param float: time in seconds
        :type t: float
        
        :param t0: event time in seconds
        :type t0: float
        
        :param tau: Decay constant in the exponential function, in seconds.
        :type tau: float
            
        :returns: Force on the cantilever at a given time, in N/kg.
        :rtype: float
            
        """

        driving_force = self.f0 * np.sin(self.wd * t)
        scale = [np.max(self.v_array) - np.min(self.v_array), np.min(self.v_array)]
        electro_force = self.fe * (self.__gamma__(t) - scale[1])/scale[0]
        
        return driving_force - electro_force