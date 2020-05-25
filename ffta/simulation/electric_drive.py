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

    def __init__(self, can_params, force_params, sim_params, v_array=[], v_step=np.nan):

        parms = [can_params, force_params, sim_params]
        super(ElectricDrive, self).__init__(*parms)        

        # Did user supply a voltage pulse themselves (Electrical drive only)
        self.use_varray = False
        self.use_vstep = False
        if any(v_array):

            if len(v_array) != int(self.total_time*self.sampling_rate):

                raise ValueError('v_array must match sampling rate/length of parameters')

            else:

                self.use_varray = True
                self.v_array = v_array
                self.scale = [np.max(v_array)-np.min(v_array), np.min(v_array)]
        
        if not np.isnan(v_step):
        
            self.v_step = v_step # if applying a single DC step
            self.use_vstep = True
        
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

                if self.use_vstep:
                    
                    return 1 # multiplies by delta_freq
                
                else:
                    
                    return -np.expm1(-(t - t0) / tau)

            else:
                
                p = int(t * self.sampling_rate)
                n_points = int(self.total_time * self.df)
                _g = self.v_array[p] if p <= n_points else self.v_array[-1]
                
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
            
        # explicitly define voltage at each time step
        if self.use_varray:
            
            p = int(t * self.sampling_rate)
            n_points = int(self.total_time * self.df)
            _g = self.v_array[p] if p <= n_points else self.v_array[-1]
            
            driving_force = 0.5 * self.dCdz/self.mass * ((_g - self.v_cpd) \
                                                         + self.v_ac * np.sin(self.wd * t))**2
        else: # single voltage step
        
            driving_force = 0.5 * self.dCdz/self.mass * ((self.dc_step(t, t0) - self.v_cpd) \
                                                         + self.v_ac * np.sin(self.wd * t))**2
        
        return driving_force