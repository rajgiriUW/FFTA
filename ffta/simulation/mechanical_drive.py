"""simulate.py: Contains Cantilever class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2020, Ginger Lab"
__email__ = "rgiri@uw.edu"
__status__ = "Production"

import numpy as np
from scipy.integrate import odeint

from .cantilever import Cantilever

# Set constant 2 * pi.
PI2 = 2 * np.pi


class MechanicalDrive(Cantilever):
    """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers under 
    Mechanial drive (i.e. conventional DDHO)

    Simulates a DDHO under excitation with given parameters.

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

    v_array : ndarray, optional
        If supplied, v_array is the time-dependent excitation to the resonance 
        frequency and the electrostatic force, scaled from 0 to 1.
        v_array must be the exact length and sampling of the desired signal

    Attributes
    ----------
    amp : float
        Amplitude of the cantilever in meters.
    beta : float
        Damping factor of the cantilever in rad/s.
    delta : float
        Initial phase of the cantilever in radians.
    delta_freq : float
        Frequency shift of the cantilever under excitation.
    mass : float
        Mass of the cantilever in kilograms.

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
    >>> from ffta.simulation import simulate, load
    >>>
    >>> params_file = '../examples/sim_params.cfg'
    >>> params = load.simulation_configuration(params_file)
    >>>
    >>> c = simulate.Cantilever(*params)
    >>> Z, infodict = c.simulate()
    >>> c.analyze()
    >>> c.analyze(roi=0.004) # can change the parameters as desired

    """

    def __init__(self, can_params, force_params, sim_params, v_array=[]):

        parms = [can_params, force_params, sim_params]
        super(MechanicalDrive, self).__init__(*parms)

        # Did user supply a voltage pulse themselves (Electrical drive only)
        self.use_varray = False
        if any(v_array):

            if len(v_array) != int(self.total_time*self.sampling_rate):

                raise ValueError('v_array must match sampling rate/length of parameters')

            if np.min(v_array) != 0 or np.max(v_array) != 1:
                
                raise ValueError('v_array must scale from 0 to 1')

            else:

                self.use_varray = True
                self.v_array = v_array
       
        return

    def __gamma__(self, t, t0, tau):
        """
        Exponential decay function for force and resonance frequency.

        Parameters
        ----------
        t : float
            Time in seconds.
        t0: float
            Event time in seconds.
        tau : float
            Decay constant in the exponential function, in seconds.

        Returns
        -------
        value : float
            Value of the function at the given time.

        """

        if t >= t0:
            
            if not self.use_varray:

                return -np.expm1(-(t - t0) / tau)

            else:
                
                p = int(t * self.sampling_rate)
                n_points = int(self.total_time * self.df)
                _g = self.v_array[p] if p <= n_points else self.v_array[-1]
                
                return _g

        else:

            return 0

    def omega(self, t, t0, tau):
        """
        Exponentially decaying resonance frequency.

        Parameters
        ----------
        t : float
            Time in seconds.
        t0: float
            Event time in seconds.
        tau : float
            Decay constant in the exponential function, in seconds.

        Returns
        -------
        w : float
            Resonance frequency of the cantilever at a given time, in rad/s.

        """

        return self.w0 + self.delta_w * self.__gamma__(t, t0, tau)

    def force(self, t, t0, tau):
        """
        Force on the cantilever at a given time. It contains driving force and
        electrostatic force.

        Parameters
        ----------
        t : float
            Time in seconds.
        t0: float
            Event time in seconds.
        tau : float
            Decay constant in the exponential function, in seconds.

        Returns
        -------
        f : float
            Force on the cantilever at a given time, in N/kg.

        """
 
        driving_force = self.f0 * np.sin(self.wd * t)
        electro_force = self.fe * self.__gamma__(t, t0, tau)
            
        return driving_force - electro_force