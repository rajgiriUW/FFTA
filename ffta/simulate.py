"""simulate.py: Contains Cantilever class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2015, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"

import numpy as np
from scipy.integrate import odeint

# Set constant 2 * pi.
PI2 = 2 * np.pi


class Cantilever(object):
    """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers.

    Simulates a DDHO under excitation with given parameters.

    Parameters
    ----------
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

    sim_params : dict
        Parameters for simulation. The dictionary contains:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)

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

    Examples
    --------
    >>> from ffta import simulate, utils
    >>>
    >>> params_file = '../examples/sim_params.cfg'
    >>> params = utils.load.configuration(params_file)
    >>>
    >>> c = simulate.Cantilever(*params)
    >>> Z, infodict = c.simulate()

    """

    def __init__(self, can_params, force_params, sim_params):

        # Initialize cantilever parameters and calculate some others.
        for key, value in can_params.items():

            setattr(self, key, value)

        self.w0 = PI2 * self.res_freq  # Radial resonance frequency.
        self.wd = PI2 * self.drive_freq  # Radial drive frequency.

        self.beta = self.w0 / (2 * self.q_factor)  # Damping factor.
        self.mass = self.k / (self.w0 ** 2)  # Mass of the cantilever in kg.
        self.amp = self.soft_amp * self.amp_invols  # Amplitude in meters.

        # Calculate reduced driving force and phase in equilibrium.
        self.f0 = self.amp * np.sqrt((self.w0 ** 2 - self.wd ** 2) ** 2 +
                                     4 * self.beta ** 2 * self.wd ** 2)
        self.delta = np.arctan(np.divide(2 * self.wd * self.beta,
                                         (self.w0 ** 2 - self.wd ** 2)))

        # Initialize force parameters and calculate some others.
        for key, value in force_params.items():

            setattr(self, key, value)

        self.delta_w = PI2 * self.delta_freq  # Frequency shift in radians.
        self.fe = self.es_force / self.mass  # Reduced electrostatic force.

        # Initialize simulation parameters.
        for key, value in sim_params.items():

            setattr(self, key, value)

        return

    def set_conditions(self, trigger_phase=180):
        """
        Sets initial conditions and other simulation parameters.

        Parameters
        ----------
        trigger_phase: float, optional
           Trigger phase is in degrees and wrt cosine. Default value is 180.

        """

        self.trigger_phase = np.mod(np.pi * trigger_phase / 180, PI2)
        self.n_points = self.total_time * 1e8

        # Add extra cycles to the simulation to find correct phase at trigger.
        cycle_points = int(2 * 1e8 / self.res_freq)
        self.n_points_sim = cycle_points + self.n_points

        # Create time vector and find the trigger wrt phase.
        self.t = np.arange(self.n_points_sim) / 1e8

        # Current phase at trigger.
        current_phase = np.mod(self.wd * self.trigger - self.delta, PI2)
        phase_diff = np.mod(trigger_phase - current_phase, PI2)

        self.t0 = self.trigger + phase_diff / self.wd

        # Set the initial conditions at t=0.
        z0 = self.amp * np.sin(-self.delta)
        v0 = self.amp * self.wd * np.cos(-self.delta)

        self.Z0 = np.array([z0, v0])

        return

    @staticmethod
    def __gamma__(t, t0, tau):
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

            return -np.expm1(-(t - t0) / tau)

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

    def dZ_dt(self, Z, t=0):
        """
        Takes the derivative of the given Z with respect to time.

        Parameters
        ----------
        Z : (2, ) array_like
            Z[0] is the cantilever position, and Z[1] is the cantilever
            velocity.
        t : float
            Time.

        Returns
        -------
        Zdot : (2, ) array_like
            Zdot[0] is the cantilever velocity, and Zdot[1] is the cantilever
            acceleration.

        """

        t0 = self.t0
        tau = self.tau

        v = Z[1]
        vdot = (self.force(t, t0, tau) -
                self.omega(t, t0, tau) * Z[1] / self.q_factor -
                self.omega(t, t0, tau) ** 2 * Z[0])

        return np.array([v, vdot])

    def simulate(self, trigger_phase=180):
        """
        Simulates the cantilever motion.

        Parameters
        ----------
        trigger_phase: float, optional
           Trigger phase is in degrees and wrt cosine. Default value is 180.

        Returns
        -------
        Z : (n_points, 1) array_like
            Cantilever position in Volts.
        infodict : dict
            Information about the ODE solver.

        """
        self.set_conditions(trigger_phase)
        Z, infodict = odeint(self.dZ_dt, self.Z0, self.t, full_output=True)

        t0_idx = int(self.t0 * 1e8)
        tidx = int(self.trigger * 1e8)

        Z_cut = Z[(t0_idx - tidx):(t0_idx + self.n_points - tidx), 0]

        step = int(1e8 / self.sampling_rate)
        n_points = self.total_time * self.sampling_rate

        self.Z = Z_cut[0::step].reshape(n_points, 1) / self.def_invols
        self.infodict = infodict

        return self.Z, self.infodict
