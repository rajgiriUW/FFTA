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


class MechanicalDrive_Simple(Cantilever):

    def __init__(self, 
                 can_params={}, 
                 force_params={}, 
                 sim_params={},
                 w_array=[]):
        """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers under 
        Mechanical drive (i.e. conventional DDHO)

        Simulates a DDHO under excitation with explicitly supplied resonance 
        frequency shift and NO electrostatic force change

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
        >>> c = mechanical_dirve.MechanicalDrive_Simple(*params)
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
        parms = [can_params, force_params, sim_params]
        super(MechanicalDrive_Simple, self).__init__(*parms)

        # Did user supply a voltage pulse themselves 
        if len(w_array) != int(self.total_time * self.sampling_rate):
            print(int(self.total_time * self.sampling_rate))
            print(len(w_array))
            raise ValueError('w_array must match sampling rate/length of parameters')

        self.w_array = w_array

        return

    def omega(self, t):
        """
        Exponentially decaying resonance frequency.

        :param float: time in seconds
        :type t: float

        :returns: Resonance frequency of the cantilever at a given time, in rad/s.
        :rtype: float
            
        """

        # return self.w0 + self.delta_w * self.__gamma__(t, t0, tau)
        p = int(np.floor(t * self.sampling_rate))
        n_points = int(self.total_time * self.sampling_rate)

        w = self.w_array[p] if p < n_points else self.w_array[-1]

        return w

    def force(self, t, t0, tau):
        """
        Force on the cantilever at a given time. It contains driving force and
        electrostatic force.

        :param t: time in seconds
        :type t: float
        
        :param t0: Event time in seconds.
        :type t0: float
        
        :param tau: Decay constant in the exponential function, in seconds.
        :type tau: float
            
        :returns: Force on the cantilever at a given time, in N/kg.
        :rtype: float
            

        """

        driving_force = self.f0 * np.sin(self.wd * t)
        # electro_force = self.fe * self.__gamma__(t)
        electro_force = self.fe

        return driving_force - electro_force

    def dZ_dt(self, Z, t=0):
        """
        Takes the derivative of the given Z with respect to time.

        :param Z: Z[0] is the cantilever position, and Z[1] is the cantilever
            velocity.
        :type Z: (2, ) array_like
        
        :param t: time
        :type t: float

        :returns: Zdot[0] is the cantilever velocity, and Zdot[1] is the cantilever
            acceleration.
        :rtype: (2, ) array_like
            

        """

        t0 = self.t0
        tau = self.tau

        v = Z[1]
        vdot = (self.force(t, t0, tau) -
                self.omega(t) * Z[1] / self.q_factor -
                self.omega(t) ** 2 * Z[0])

        return np.array([v, vdot])
