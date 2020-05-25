"""simulate.py: Contains Cantilever class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2020, Ginger Lab"
__email__ = "rgiri@uw.edu"
__status__ = "Production"

import numpy as np
from scipy.integrate import odeint

import ffta 

# Set constant 2 * pi.
PI2 = 2 * np.pi


class ElectricDrive(Cantilever):
    """Damped Driven Harmonic Oscillator Simulator for AFM Cantilevers under Electric drive

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
        v_dc = float (in Volts)
        v_ac = float (in Volts)
        v_cpd = float (in Volts)
        dCdz = float (in F/m)

    sim_params : dict
        Parameters for simulation. The dictionary contains:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)

    elec_drive : bool, optional 
        If True (default), uses AC voltage drive. Requires that force_params
        contain v_dc, v_ac, v_cpd, and dCdz

    v_array : ndarray, optional
        If supplied and elec_drive is True, supplies the time-dependent voltage to v_cpd
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

    def __init__(self, can_params, force_params, sim_params, elec_drive=False, v_array=[]):

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
        self.delta = np.abs(np.arctan(np.divide(2 * self.wd * self.beta,
                                                self.w0 ** 2 - self.wd ** 2)))

        # Initialize force parameters and calculate some others.
        for key, value in force_params.items():

            setattr(self, key, value)

        self.delta_w = PI2 * self.delta_freq  # Frequency shift in radians.
        self.fe = self.es_force / self.mass  # Reduced electrostatic force.

        # Initialize simulation parameters.
        for key, value in sim_params.items():

            setattr(self, key, value)

        # Calculate time axis for simulated tip motion without extra cycles
        num_pts = int(self.total_time*self.sampling_rate)
        self.t_Z = np.linspace(0, self.total_time, num=num_pts)

        # Calculate frequency axis for simulated tip_motion without extra cycles.
        self.freq_Z = np.linspace(0, int(self.sampling_rate/2), num=int(num_pts/2 + 1))

        # Did user supply a voltage pulse themselves (Electrical drive only)
        self.elec_drive = elec_drive
        self.use_varray = False
        if any(v_array):

            if len(v_array) != num_pts:

                raise ValueError('v_array must match sampling rate/length of parameters')

            else:

                self.use_varray = True
                self.v_array = v_array
        
        # Create a Pixel class-compatible params file
        self.fit_params = {}
        self.parameters = force_params
        self.parameters.update(**sim_params)
        self.can_params = can_params
        self.create_parameters(self.parameters, self.can_params)
        
        # Define simulation sampling rate, default = 100 MHz
        self.df = 1e8
        
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
        if self.elec_drive:
            
            # explicitly define voltage at each time step
            if self.use_varray:
                
                p = int(t * self.sampling_rate)
                try:
                    driving_force = 0.5 * self.dCdz/self.mass * ((self.v_array[p] - self.v_cpd) \
                                                             + self.v_ac * np.sin(self.wd * t))**2
                except:
                    driving_force = 0.5 * self.dCdz/self.mass * ((self.v_array[-1] - self.v_cpd) \
                                                             + self.v_ac * np.sin(self.wd * t))**2
            else:
                driving_force = 0.5 * self.dCdz/self.mass * ((self.dc_step(t, t0) - self.v_cpd) \
                                                             + self.v_ac * np.sin(self.wd * t))**2
            
            return driving_force
        
        #mechanical driving
        else: 
            
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

    def simulate(self, trigger_phase=180, Z0=None):
        """
        Simulates the cantilever motion.

        Parameters
        ----------
        trigger_phase: float, optional
           Trigger phase is in degrees and wrt cosine. Default value is 180.
        Z0 : list, optional
            Z0 = [z0, v0], the initial position and velocity
            If not specified, is calculated from the analytical solution to DDHO
            (using "set_conditions")
        Returns
        -------
        Z : (n_points, 1) array_like
            Cantilever position in Volts.
        infodict : dict
            Information about the ODE solver.

        """
        
        if Z0:
            
            if not isinstance(Z0, (np.ndarray, list)):
                raise TypeError('Must be 2-size array or list')
            if len(Z0) != 2:
                raise ValueError('Must specify exactly [z0, v0]')
            
            self.n_points = int(self.total_time * self.df)
            self.t = np.arange(self.n_points) / self.df
            self.t0 = self.trigger 
            self.Z0 = Z0
        
        else:
            self.set_conditions(trigger_phase) 
        
        Z, infodict = odeint(self.dZ_dt, self.Z0, self.t, full_output=True)

        t0_idx = int(self.t0 * self.df)
        tidx = int(self.trigger * self.df)
        n_points = int(self.total_time * self.sampling_rate)
        Z_cut = Z[(t0_idx - tidx):(t0_idx + self.n_points - tidx), 0]

        step = int(self.df / self.sampling_rate)

        self.Z = Z_cut[0::step].reshape(n_points, 1) / self.def_invols
        self.infodict = infodict

        return self.Z, self.infodict

    def create_parameters(self, params={}, can_params={}, fit_params={}):
        '''
        Creates a Pixel class-compatible parameters and cantilever parameters Dict
        
        Parameters
        ----------
        params : dict, optional
            Contains analysis parameters for the Pixel cass
        
        can_params : dict, optional
            Contains cantilever parameters for the Pixel class. These data are
            optional for the analysis.
        
        fit_params : dict, optional
            Contains various parameters for fitting and analysis. See Pixel class.
        '''
        
        # default seeding of parameters
        _parameters = {'bandpass_filter': 1.0,
                       'drive_freq': 277261,
                       'filter_bandwidth': 10000.0,
                       'n_taps': 799,
                       'roi': 0.0003,
                       'sampling_rate': 1e7,
                       'total_time': 0.002,
                       'trigger': 0.0005,
                       'window': 'blackman',
                       'wavelet_analysis': 0}
        
        _can_params = {'amp_invols' : 5.52e-08,
                       'def_invols' : 5.06e-08, 
                       'k' : 26.2,
                       'q_factor' : 432}     
     
        _fit_params = {'filter_amplitude': True,
                       'method': 'hilbert',
                       'fit': True,
                       'fit_form': 'product'}
     
        for key, val in _parameters.items():
            if key not in params:
                if hasattr(self, key):
                    params[key] = self.__dict__[key]
                else:
                    params[key] = val
        
        for key, val in _can_params.items():
            if key not in can_params:
                if hasattr(self, key):
                    can_params[key] = self.__dict__[key]
                else:
                    can_params[key] = val
        
        for key, val in _fit_params.items():
            if key not in fit_params:
                if hasattr(self, key):
                    fit_params[key] = self.__dict__[key]
                else:
                    fit_params[key] = val
        
        # then write to the Class
        self.parameters.update(**params)
        self.can_params.update(**can_params)
        self.fit_params.update(**fit_params)
        
        return 
    
    def analyze(self, plot=True, **kwargs):
        '''
        Converts output to a Pixel class and analyzes
        
        Parameters
        ----------
        plot : bool, optional
            If True, calls Pixel.plot() to display the results
        

        Returns
        -------
        None.

        '''
        param_keys = ['bandpass_filter', 'drive_freq', 'filter_bandwidth', 'n_taps',
                       'roi', 'sampling_rate', 'total_time', 'trigger', 'window', 'wavelet_analysis']
        
        can_param_keys = ['amp_invols','def_invols', 'k', 'q_factor']
        
        fit_param_keys = ['filter_amplitude', 'method', 'fit', 'fit_form']
        
        params = {}
        can_params = {}
        fit_params = {}
        
        for k, v in kwargs.items():
            if k in param_keys:
                params[k] = v
            elif k in can_param_keys:
                can_params[k] = v
            elif k in fit_param_keys:
                fit_params[k] = v
        
        self.create_parameters(params, can_params, fit_params)
        
        pix = ffta.pixel.Pixel(self.Z, self.parameters, self.can_params, **self.fit_params)

        pix.analyze()        
        
        if plot:
            
            pix.plot()
        
        return
        