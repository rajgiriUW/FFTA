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
    """DDHO simulator with an explicitly supplied resonance frequency array and no electrostatic force.

    Simulates a cantilever under mechanical drive where the resonance frequency
    shift is provided directly as an array rather than computed from an
    exponential or arbitrary function.

    Attributes
    ----------
    Z : ndarray
        ODE integration of the DDHO response.

    Methods
    -------
    simulate(trigger_phase=180)
        Simulates the cantilever motion with excitation at the given phase.

    See Also
    --------
    MechanicalDrive : standard DDHO with function- or array-based excitation.
    Cantilever : base class.

    Examples
    --------
    >>> from ffta.simulation import mechanical_drive_simple
    >>> from ffta.simulation.utils import load
    >>>
    >>> params_file = '../examples/sim_params.cfg'
    >>> params = load.simulation_configuration(params_file)
    >>>
    >>> n_points = int(params[2]['total_time'] * params[2]['sampling_rate'])
    >>> w_array = np.ones(n_points) * 2 * np.pi * 300e3  # flat resonance frequency
    >>> c = mechanical_drive_simple.MechanicalDrive_Simple(*params, w_array=w_array)
    >>> Z, infodict = c.simulate()
    >>> c.analyze()
    >>> c.analyze(roi=0.004)
    """

    def __init__(self,
                 can_params={},
                 force_params={},
                 sim_params={},
                 w_array=[]):
        """
        Parameters
        ----------
        can_params : dict
            Cantilever properties. See :class:`Cantilever`.
        force_params : dict
            Force parameters. Beyond Cantilever, contains:

            es_force : float (N)
            delta_freq : float (Hz)
            tau : float (s)
        sim_params : dict
            Simulation parameters. Contains:

            trigger : float (s)
            total_time : float (s)
            sampling_rate : int (Hz)
        w_array : ndarray
            Time-dependent resonance frequency in rad/s. Must match the length
            implied by total_time * sampling_rate.
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
        Returns the resonance frequency at time t from the supplied w_array.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Resonance frequency at time t, in rad/s.
        """

        # return self.w0 + self.delta_w * self.__gamma__(t, t0, tau)
        p = int(np.floor(t * self.sampling_rate))
        n_points = int(self.total_time * self.sampling_rate)

        w = self.w_array[p] if p < n_points else self.w_array[-1]

        return w

    def force(self, t, t0, tau):
        """
        Force on the cantilever at a given time (driving force only; no electrostatic term).

        Parameters
        ----------
        t : float
            Time in seconds.
        t0 : float
            Event time in seconds.
        tau : float
            Decay constant in seconds (unused here, kept for interface compatibility).

        Returns
        -------
        float
            Force on the cantilever at time t, in N/kg.
        """

        driving_force = self.f0 * np.sin(self.wd * t)
        electro_force = self.fe

        return driving_force - electro_force

    def dZ_dt(self, Z, t=0):
        """
        Derivative of the state vector Z with respect to time.

        Parameters
        ----------
        Z : array_like, shape (2,)
            Z[0] is cantilever position, Z[1] is cantilever velocity.
        t : float, optional
            Time in seconds. Default 0.

        Returns
        -------
        ndarray, shape (2,)
            Zdot[0] is cantilever velocity, Zdot[1] is cantilever acceleration.
        """

        t0 = self.t0
        tau = self.tau

        v = Z[1]
        vdot = (self.force(t, t0, tau) -
                self.omega(t) * Z[1] / self.q_factor -
                self.omega(t) ** 2 * Z[0])

        return np.array([v, vdot])
