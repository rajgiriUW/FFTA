"""simulate.py: Contains Cantilever class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2020, Ginger Lab"
__email__ = "rgiri@uw.edu"
__status__ = "Production"

import numpy as np
from math import pi
from scipy.integrate import odeint

from .cantilever import Cantilever
from . import excitation

# Set constant 2 * pi.
PI2 = 2 * pi


class ElectricDrive(Cantilever):
    """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers under Electric drive

    Simulates a DDHO under excitation with given parameters.

    Parameters
    ----------
    can_params : dict
        Parameters for cantilever properties. See Cantilever

    force_params : dict
        Parameters for forces. The dictionary contains:

        es_force = float (in N)
        delta_freq = float (in Hz)
        tau = float (in seconds)
        v_dc = float (in Volts)
        v_ac = float (in Volts)
        v_cpd = float (in Volts)
        dCdz = float (in F/m)

    sim_params : dict
        Parameters for simulation. The dictionary contains:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)

    v_array : ndarray, optional
        If provided, supplies the time-dependent voltage to v_cpd
        v_array must be the exact length and sampling of the desired signal
        v_array only functionally does anything after the trigger.

    v_step : float, optional
        If v_array not supplied, then a voltage of v_step is applied at the trigger

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
    >>> from ffta.simulation import electric_drive, load
    >>>
    >>> params_file = '../examples/sim_params.cfg'
    >>> params = load.simulation_configuration(params_file)
    >>>
    >>> c = electric_drive.ElectricDrive(*params)
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
    >>> # To supply an arbitary voltage step
    >>> step = -7 #-7 Volt step
    >>> c = mechanical_dirve.MechanicalDrive(*params, v_step = step)
    >>> Z, _ = c.simulate()
    >>> c.analyze()

    """

    def __init__(self, can_params, force_params, sim_params, v_array=[], v_step=np.nan,
                 func=excitation.single_exp, func_args=[]):

        parms = [can_params, force_params, sim_params]
        super(ElectricDrive, self).__init__(*parms)

        # Did user supply a voltage pulse themselves (Electrical drive only)
        self.use_varray = False
        self.use_vstep = False
        if any(v_array):

            if len(v_array) != int(self.total_time * self.sampling_rate):

                raise ValueError('v_array must match sampling rate/length of parameters')

            else:

                self.use_varray = True
                self.v_array = v_array
                self.scale = [np.max(v_array) - np.min(v_array), np.min(v_array)]

        if not np.isnan(v_step):
            self.v_step = v_step  # if applying a single DC step
            self.use_vstep = True

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

        return

    def __gamma__(self, t):
        """
        Controls how the cantilever behaves after a trigger.
        Default operation is an exponential decay to omega0 - delta_freq with
        time constant tau.

        If supplying an explicit v_array, then this function will call the values
        in that array

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        value : float
            Value of the function at the given time.

        """

        p = int(t * self.sampling_rate)
        n_points = int(self.total_time * self.sampling_rate)
        t0 = self.t0

        if t >= t0:

            if not self.use_varray:

                return self.func(t - t0, *self.func_args)

            else:

                _g = self.v_array[p] if p < n_points else self.v_array[-1]
                _g = (_g - self.scale[1]) / self.scale[0]

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

        return self.w0 + self.delta_w * self.__gamma__(t)

    def dc_step(self, t, t0):
        """
        Adds a DC step at the trigger point for electrical drive simulation

        Parameters
        ----------
        t : float
            Time in seconds.
        t0: float
            Event time in seconds.
        """

        if t > t0:

            return self.v_step

        else:

            return self.v_dc

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

        # explicitly define voltage at each time step
        if self.use_varray:

            p = int(t * self.sampling_rate)
            n_points = int(self.total_time * self.sampling_rate)

            _g = self.v_array[p] if p < n_points else self.v_array[-1]

            driving_force = 0.5 * self.dCdz / self.mass * ((_g - self.v_cpd) \
                                                           + self.v_ac * np.sin(self.wd * t)) ** 2
        else:  # single voltage step

            driving_force = 0.5 * self.dCdz / self.mass * ((self.dc_step(t, t0) - self.v_cpd) \
                                                           + self.v_ac * np.sin(self.wd * t)) ** 2

        return driving_force

    def set_conditions(self, trigger_phase=180):
        """
        Sets initial conditions and other simulation parameters. Using 2w given
        the squared term in electric driving

        Parameters
        ----------
        trigger_phase: float, optional
           Trigger phase is in degrees and wrt cosine. Default value is 180.

        """
        self.delta = np.abs(np.arctan(np.divide(2 * (2 * self.wd) * self.beta,
                                                self.w0 ** 2 - (2 * self.wd) ** 2)))

        self.trigger_phase = np.mod(np.pi * trigger_phase / 180, PI2)
        self.n_points = int(self.total_time * self.sampling_rate)

        # Add extra cycles to the simulation to find correct phase at trigger.
        cycle_points = int(2 * self.sampling_rate / self.res_freq)
        self.n_points_sim = cycle_points + self.n_points

        # Create time vector and find the trigger wrt phase.
        self.t = np.arange(self.n_points_sim) / self.sampling_rate

        # Current phase at trigger.
        current_phase = np.mod(self.wd * self.trigger - self.delta, PI2)
        phase_diff = np.mod(self.trigger_phase - current_phase, PI2)

        self.t0 = self.trigger + phase_diff / self.wd  # modified trigger point

        # Set the initial conditions at t=0.
        z0 = self.amp * np.sin(-self.delta)
        v0 = self.amp * self.wd * np.cos(-self.delta)

        self.Z0 = np.array([z0, v0])

        return
