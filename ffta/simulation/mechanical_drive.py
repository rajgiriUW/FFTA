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


class MechanicalDrive(Cantilever):
    """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers under
    Mechanial drive (i.e. conventional DDHO)

    Simulates a DDHO under excitation with given parameters and a change to resonance
    and electrostatic force

    Time-dependent change can be specified in two ways:
        1) explicitly defining v_array, a scale from 0 to 1 of the same length as
            the desired integration
        2) using a defined function and parameter, passed to parameter "func"
            This approach will call self.func(t, \*self.func_args)
            By default, this will call excitation.single_exp, a single exponential
            decay.
            For this approach to work, you must supply or set self.func_args = []

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
    >>> from ffta.simulation import mechanical_drive
    >>> from ffta.simulation.utils import load
    >>>
    >>> params_file = '../examples/sim_params.cfg'
    >>> params = load.simulation_configuration(params_file)
    >>>
    >>> c = mechanical_dirve.MechanicalDrive(*params)
    >>> Z, infodict = c.simulate()
    >>> c.analyze()
    >>> c.analyze(roi=0.004) # can change the parameters as desired
    >>>
    >>> # To supply an arbitary v_array
    >>> n_points = int(params[2]['total_time'] * params[2]['sampling_rate'])
    >>> v_array = np.ones(n_points) # create just a flat excitation
    >>> c = mechanical_dirve.MechanicalDrive(*params, v_array = v_array)
    >>> Z, _ = c.simulate()
    >>> c.analyze()
    >>>
    >>> # To use a function instead of artbitary array, say stretch exponential
    >>> c = mechanical_dirve.MechanicalDrive(*params, func=excitation.str_exp, func_args=[1e-3, 0.8])
    >>> Z, _ = c.simulate()
    >>> c.analyze()
    >>> c.func_args = [1e-3, 0.7] # change beta value in stretched exponential
    >>> Z, _ = c.simulate()
    >>> c.analyze()

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
        frequency and the electrostatic force, scaled from 0 to 1.
        v_array must be the exact length and sampling of the desired signal
    :type v_array: ndarray, optional
    
    :param func:
    :type func: function, optional

    """

    def __init__(self, can_params, force_params, sim_params,
                 v_array=[], func=excitation.single_exp, func_args=[]):

        parms = [can_params, force_params, sim_params]
        super(MechanicalDrive, self).__init__(*parms)

        # Did user supply a voltage pulse themselves
        self.use_varray = False
        if any(v_array):

            if len(v_array) != int(self.total_time * self.sampling_rate):
                raise ValueError('v_array must match sampling rate/length of parameters')

            if np.min(v_array) != 0 or np.max(v_array) != 1:

                raise ValueError('v_array must scale from 0 to 1')

            else:

                self.use_varray = True
                self.v_array = v_array

        self.func = func
        self.func_args = func_args

        # default case set a single tau for a single exponential function
        if not np.any(func_args):
            self.func_args = [self.tau]

        try:
            _ = self.func(0, *self.func_args)
        except:
            print('Be sure to correctly set func_args=[params]')

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

            if not self.use_varray:

                return self.func(t - t0, *self.func_args)

            else:

                _g = self.v_array[p] if p < n_points else self.v_array[-1]

                return _g

        else:

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

        # return self.w0 + self.delta_w * self.__gamma__(t, t0, tau)
        return self.w0 + self.delta_w * self.__gamma__(t)

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
        # electro_force = self.fe * self.__gamma__(t, t0, tau)
        electro_force = self.fe * self.__gamma__(t)

        return driving_force - electro_force
