#!/usr/bin/env python3
"""biexponential.py: Simulates the response of an AFM cantilever to a biexponential force 
(like a mixed ionic-electronic photoconductor's capacitive force on an AFM tip)"""
__author__ = "Jake T. Precht"
__copyright__ = "Copyright 2018, Ginger Lab"
__credits__ = "Rajiv Giridharagopal, Durmus Karatay"
__maintainer__ = "Jake T. Precht"
__email__ = "jprecht@uw.edu"
__status__ = "Development"

"""Known issues as of 2018_08_07:
    --Minimum tau is 1e-6.  This is because when creating the exponential rise
    or decays in exponential_rise_or_decay() function, you generate floats on
    the order of 1e-15 which is the numpy 64bit float resolution limit.  Possible
    fixes are detailed in Lab Notebook 3 page 172
    
    --Unsure if Code is currently good for negative phase (i.e. when drive
    frequency > resonant frequency).  Need to think about and test if going
    to drive off-resonance.
    
    --Assuming cantilever is tuned to 1V.  Will need to add term to 
    EQ_amplitude if this is not the case.

    """

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import configparser


class biexp():

    def __init__(self, exponential_rise_or_decay, tau_e, beta_e, tau_i, beta_i,
                 electronic_fraction, path = 'C:/Users/Jake/DDHO/data/parameters2.cfg'):
        
        """when initializing this class, you will need to provide the following parameters:
            exponential_rise_or_decay = "rise" or "decay" (what form is the light/voltage?)
            rise = turning on voltage/light
            decay = turning off voltage/light
            
            tau_e = lifetime for faster component (electronic) of biexponential in us
            beta_e = stretched exponential value for faster component (beta = 1 for normal, unstretched exponentials)
            
            tau_i = lifetime for slower component (ionic) of biexponential in us
            beta_i = stretched exponential value for slower component (beta = 1 for normal, unstretched exponentials)
            
            electronic_fraction = float between 0-1 and is the percentage of the voltage/light pulse amplitude from the fast component (0.5 if equal magnitude fast/slow)
            
            path = path to .cfg file with all the parameters.  should usually be same folder as that used for pixel.py
            all simulation parameters go under [Parameters] in the .cfg
            """
        
        self.tau_electronic = tau_e
        self.beta_e = beta_e
        
        self.tau_ionic = tau_i
        self.beta_i = beta_i
        
        self.electronic_fraction = electronic_fraction
        self.ionic_fraction = 1 - electronic_fraction
        
        if str.lower(exponential_rise_or_decay) == "rise":
            self.is_exponential_rise = 1
        elif str.lower(exponential_rise_or_decay) == "decay":
            self.is_exponential_rise = 0
        else:
            raise ValueError("Input for exponential_rise_or_decay must be ''rise'' or ''decay''")
        
        
        #load other parameters
        config = configparser.RawConfigParser()
        config.read(path)
        
        for (each_key,each_value) in config.items('Parameters'):
            setattr(self,each_key,config.getfloat('Parameters',each_key))
        
        if self.resonant_frequency == self.drive_freq:
            self.cantilever_phase = np.pi/2
        else:
            self.cantilever_phase = np.arctan(np.divide((self.resonant_frequency * self.drive_freq), 
                               (self.q * (self.resonant_frequency**2 - self.drive_freq**2))))
        
        
        #preprocess some values
        self.drive_freq = 2 * np.pi * self.drive_freq
        self.resonant_frequency = 2 * np.pi * self.resonant_frequency
        
        self.electrostatic_force_electronic = self.total_electro_force * self.electronic_fraction #in N
        self.electrostatic_force_ionic = self.total_electro_force * self.ionic_fraction #in N
        
        self.frequency_shift_electronic = 2 * np.pi * self.total_freq_shift * self.electronic_fraction #in Hz and radial frequency
        self.frequency_shift_ionic = 2 * np.pi * self.total_freq_shift * self.ionic_fraction #in Hz and radial frequency
        
        #Calculate some simulation values
        self.simulation_trigger_index = int(self.trigger * self.simulation_resolution)
        self.trigger_phase = np.pi #in radians -- Durmus' paper suggests maximum sensitivity at 180 degrees trigger
        
        self.simulation_time = self.total_time# + 5*(1/self.resonant_frequency) 
        #I used to add extra simulation cycles on so that I could trigger at appropriate phase and then delete some extra points so trigger was at trigger time and total time was conserved
        self.number_of_simulation_points = self.simulation_time * self.simulation_resolution #making simulations 10x or 20x resolution of real data (5 or 10 MHz gage gard).  then I reshape data at end to match sampling_rate
        self.t = np.arange(self.number_of_simulation_points) / self.simulation_resolution
        
        
        
        """Calculate other necessary parameters:"""
        #Calculate cantilever equilibrium amplitude (assuming tuned to 1V)
        self.EQ_amplitude = self.soft_amplitude * self.amp_invols #in meters
        self.mass = self.k / (self.resonant_frequency**2) #in kg
        
        #Calculate equilibrium reduced driving force magnitude (see Lab Notebook 3 pages 167-168 for derivation)
        self.driving_force = self.mass * self.EQ_amplitude * (
            np.sqrt((self.resonant_frequency**2 - self.drive_freq**2)**2 
            + ((self.resonant_frequency**2 * self.drive_freq**2)/self.q**2))) #in N

        
        return
        
    
    def set_t0(self):
        """Calculates where in a cantilever oscillation cycle the trigger 
        occurs at based upon time, and moves the trigger slightly forward in 
        time to instead occur at appropriate trigger phase
        
        The function Z_reshape later will shift all of my data back by this
        small amountof time we moved so as to maintain trigger time at the appropriate
        real time.  Together this allows trigger phase and time to be properly set."""
        
        current_cycle_phase = np.mod(self.drive_freq * self.trigger - self.cantilever_phase, 2 * np.pi)
        phase_offset = np.mod(self.trigger_phase - current_cycle_phase, 2 * np.pi) #amount of a cycle (in radians) until the cantilever phase reaches trigger phase
        
        self.t0 = self.trigger + phase_offset / self.drive_freq #this line adds a sub-cantilever oscillation cycle amount of time such that t0 occurs when the cantilever oscillation cycle phase is equal to trigger phase 
        return
    
    
    def set_initial_conditions(self):
        """Calculate initial position and velocity for numeric integrations"""
        z0 = self.EQ_amplitude * np.sin(0 - self.cantilever_phase) #z = A*sin(w * t - phase)
        v0 = self.EQ_amplitude * self.drive_freq * np.cos(0 - self.cantilever_phase) #derivative of z0
        
        self.z0v0 = np.array([z0,v0])
        return
        
        
        
    def stretched_exponential_rise_or_decay(self, t, t0, tau, beta):
        """Stretched Exponential rise or decay function 
        (simulates turning light/voltage on or off)
        
        This function gets called for both the time-evolving force and force gradient"""
       
        
        if self.tau_electronic != 0:
            #if exponential rise (light on):
            if self.is_exponential_rise == 1:
                if t>= t0:
                    return -np.expm1(-((t-t0) / tau)**beta)
                else:
                    return 0
        
            
            #if exponential decay (light off):
            else: 
                if t>= t0:
                    return np.exp(-((t-t0) / tau)**beta)
                else:
                    return 0
        else: 
            return 0


    def omega_stretched_biexponential(self, t, t0, tau_e, beta_e, tau_i, beta_i):
        """Resonance frequency/force gradient that shifts with time for stretched monoexponential force"""
        return self.resonant_frequency - 1 * (self.frequency_shift_electronic * self.stretched_exponential_rise_or_decay(t, t0, tau_e, beta_e)  + self.frequency_shift_ionic * self.stretched_exponential_rise_or_decay(t, t0, tau_i, beta_i))
    
    def force_stretched_biexponential(self, t, t0, tau_e, beta_e, tau_i, beta_i):
        """Monoexponential force that shifts with time"""
        driving_force = self.driving_force * np.sin(self.drive_freq * t)
        
        if self.tau_electronic == 0:
            electrostatic_force_bi = 0
        else:
            electrostatic_force_bi = (self.electrostatic_force_electronic * self.stretched_exponential_rise_or_decay(t, t0, tau_e, beta_e) + 
                                      self.electrostatic_force_ionic * self.stretched_exponential_rise_or_decay(t, t0, tau_i, beta_i))
                                      
        return driving_force - electrostatic_force_bi
    
        
#    def force_monoexponential(self, t, t0, tau): #special version for Raj's paper
 #       """Monoexponential force that shifts with time and squares the (driving force - electrostatic force) while maintaining proper units"""
  #      driving_tau = (np.sqrt(self.driving_force) * np.sin(self.drive_freq * t) + 
   #     np.sqrt(self.driving_force) * np.sin(self.drive_freq * 2 * t) / 2 )
        #np.sqrt(self.driving_force) * np.sin(self.drive_freq * 3 * t) / 4 +
        #np.sqrt(self.driving_force) * np.sin(self.drive_freq * 4 * t) / 8)
        
    #    electro_force_tau = np.sqrt(self.electrostatic_force_electronic) * self.exponential_rise_or_decay(t, t0, tau)
        
     #   total_tau = (driving_tau - electro_force_tau)**2
      #  return total_tau      
    
    def force_gmode_biexponential(self, t, t0, tau_electronic, tau_ionic, beta_e = 1, beta_i = 1):
        """biexponential force that shifts with time and squares the total force while maintaining proper units"""
        driving_force = (np.sqrt(self.driving_force) * np.sin(self.drive_freq * t))
        elec_force = np.sqrt(self.electrostatic_force_electronic) * self.stretched_exponential_rise_or_decay(t, t0, tau_electronic, beta_e)
        ion_force = np.sqrt(self.electrostatic_force_ionic) * self.stretched_exponential_rise_or_decay(t, t0, tau_ionic, beta_i)
        
        total_force = (driving_force - elec_force - ion_force)**2
        return total_force
    
    def force_array(self):
        """function for playing with stuff for Raj's paper
        NOT FUNCTIONAL!!!"""
        
        self.t = np.arange(self.number_of_simulation_points) / self.simulation_resolution
        t = self.t[0::20]
        t0 = self.t0
        tau_electronic = self.tau_electronic
        tau_ionic = self.tau_ionic
        beta_e = self.beta_e
        beta_i = self.beta_i
        
        force = np.zeros(int(len(t)))
        for i in range(int(len(t))):
            force[i] = self.force_gmode_biexponential(t[i],t0,tau_electronic,tau_ionic,beta_e,beta_i)
        return force
    
    def voltage_array(self):
        """function for playing with stuff for Raj's paper
        NOT FUNCTIONAL"""
        
        self.t = np.arange(self.number_of_simulation_points) / self.simulation_resolution
        t = self.t
        t0 = self.t0
        tau_electronic = self.tau_electronic
        tau_ionic = self.tau_ionic
        beta_e = self.beta_e
        beta_i = self.beta_i
        
        voltage = np.zeros(len(t))
        for i in range(len(t)):
            voltage[i] = self.t
        return voltage
   

    def DDHO_dzdt_stretched_biexponential_force(self, z, t):
       """calculate acceleration for odeint purposes (odeint requires you to 
       convert a 2nd order ODE into a system of two first order ODEs)
       
       "z" is thing we are going to integrate to get"""
       
       t0 = self.t0
       tau_e = self.tau_electronic
       beta_e = self.beta_e
       tau_i = self.tau_ionic
       beta_i = self.beta_i
       
       velocity1 = z[1]
       acceleration1 = ((self.force_stretched_biexponential(t, t0, tau_e,beta_e,tau_i,beta_i) / self.mass)
                       - (self.omega_stretched_biexponential(t,t0,tau_e,beta_e,tau_i,beta_i) * velocity1 / self.q)
                       - self.omega_stretched_biexponential(t,t0,tau_e,beta_e,tau_i,beta_i) ** 2 * z[0])
       return np.array([velocity1, acceleration1])
   
    def DDHO_dzdt_Gmode_force(self, z, t):
       """calculate acceleration for odeint purposes (odeint requires you to 
       convert a 2nd order ODE into a system of two first order ODEs)
       
       "z" is thing we are going to integrate to get"""
       
       t0 = self.t0
       tau_e = self.tau_electronic
       beta_e = self.beta_e
       tau_i = self.tau_ionic
       beta_i = self.beta_i
       
       velocity1 = z[1]
       acceleration1 = ((self.force_gmode_biexponential(t, t0, tau_e,beta_e,tau_i,beta_i) / self.mass)
                       - (self.omega_stretched_biexponential(t,t0,tau_e,beta_e,tau_i,beta_i) * velocity1 / self.q)
                       - self.omega_stretched_biexponential(t,t0,tau_e,beta_e,tau_i,beta_i) ** 2 * z[0])
       """acceleration1 = ((self.force_monoexponential(t, t0, tau1) / self.mass)
                       - (self.omega_monoexponential(t,t0,tau1) * velocity1 / self.q)
                       - self.omega_monoexponential(t,t0,tau1) ** 2 * z[0])"""
       return np.array([velocity1, acceleration1])
    
    
    def run_stretched_simulation(self):
        """Perform simulations of cantilever z motion using supplied parameters"""
        #self.init_params()
        self.set_t0()
        self.set_initial_conditions()

        Z_biexponential_force = odeint(self.DDHO_dzdt_stretched_biexponential_force, self.z0v0, self.t)
        #Note these Z's have too many points relative to our experimental sampling rate
        #The output of this function should be next sent to Z_reshape()
        
        return Z_biexponential_force
    
    def run_Gmode_simulation(self):
        """Perform simulations of cantilever z motion using supplied parameters"""
        #self.init_params()
        self.set_t0()
        self.set_initial_conditions()

        #Z_monoexponential_force = odeint(self.DDHO_dzdt_stretched_monoexponential_force, self.z0v0, self.t)
        Z_biexponential_force = odeint(self.DDHO_dzdt_Gmode_force, self.z0v0, self.t)
        #Note these Z's have too many points relative to our experimental sampling rate
        #The output of this function should be next sent to Z_reshape()
        
        return Z_biexponential_force
        
    
    def Z_reshape(self, simulated_Z):
        """Take simulated cantilever motion from run_simulation() and 
        resample it to be at same sampling rate as Gage card/experiment.
        This REQUIRES that the simulation is ran with 
        frequency > Gage card frequency (1e7).  Default simulation sampling rate is 1e8
        
        ***This function also offsets the simulation such that 
        the trigger occurs at the appropriate place in real time
        (it undoes the offset performed in set_t0()"""
        
        simulation_t0_index = int(self.t0 * self.simulation_resolution)
        simulation_offset = simulation_t0_index - self.simulation_trigger_index
        
        #This next line undoes the time shift performed bn set_t0() #and cuts
        #the extraneous cycles from the end of the data
        Z_cut = simulated_Z[simulation_offset:(int((self.total_time*self.simulation_resolution) + simulation_offset)),0]
        
        #The below lines resample the data such that it is outputted at the
        #proper experimental sampling rate
        reshape_step_size = int(self.simulation_resolution / self.sampling_rate)
        Z_reshape = Z_cut[0::reshape_step_size] #/ self.def_invols #if you want it in volts
        t_reshape = self.t[0::reshape_step_size]
        
        return Z_reshape, t_reshape
    
    
    def noise_on_z(self, simulated_Z, noise_magnitude):
        """Noise magnitude should be standard deviation of noise in meters
        This function applies normal, Gaussian noise"""
        noise = np.random.normal(0, noise_magnitude, simulated_Z.shape)
        noisy_signal = simulated_Z + noise
        return noisy_signal
    
    def Z_reshape_ROI_only(self, simulated_Z):
        """not used in production.  was used for simulations"""
        simulation_t0_index = int(self.t0 * self.simulation_resolution)
        simulation_offset = simulation_t0_index - self.simulation_trigger_index
        
        #This next line undoes the time shift performed bn set_t0() and cuts
        #the extraneous cycles from the end of the data
        Z_cut = simulated_Z[simulation_offset:(int((self.total_time*self.simulation_resolution) + simulation_offset)),0]
        
        #The next line is the key difference between Z_reshape and Z_reshape_ROI_only
        #It truncates the Z data to only return the ROI after the trigger
        Z_cut = Z_cut[self.simulation_trigger_index:]
        
        #The below lines resample the data such that it is outputted at the
        #proper experimental sampling rate
        reshape_step_size = int(self.simulation_resolution / self.sampling_rate)
        Z_reshape = Z_cut[0::reshape_step_size] #/ self.def_invols #if you want it in volts
        #t_reshape = self.t[0::reshape_step_size]
        
        return Z_reshape
    
    

def line_loop(taus):
    """requires .txt file in root with each line being a tau in us"""
    array_of_taus = np.loadtxt(str(taus))
    
    #initialize output data
    output_array = np.array([])
    output_array_noisy = np.array([])
    
    for tau in array_of_taus:
        #Initialize new cantilever object
        tau_in_s = tau * 1e-6
        Canti = biexp("rise", tau_in_s, 1, tau_in_s, 1, 1)
        
        #Run simulations
        data_mono = Canti.run_stretched_simulation()
        #data_mono_noisy = Canti.noise_on_z(data_mono,0.3e-9)
        
        #Reshape to proper data output sampling rate
        data_mono_reshape, t_mono_reshape = Canti.Z_reshape(data_mono)
        #data_noisy_mono_reshape, t_noisy_mono_reshape = Canti.Z_reshape(data_mono_noisy)
        
        #Concatenate to output array
        output_array = np.append(output_array,data_mono_reshape)
        #output_array_noisy = np.append(output_array_noisy,data_noisy_mono_reshape)
        
        #Saving
        #savestr = 'z_data_' + str(int(tau)) + '.txt'
        #savestrnoisy = 'z_data_' + str(int(tau)) + '_noisy_.txt'
        #np.savetxt(savestr,data_mono_reshape,delimiter=',')
        #np.savetxt(savestrnoisy,data_noisy_mono_reshape,delimiter=',')
        
        #plotting for debugging purposes only.  Don't run with a large list of taus!!!!!!
        #plt.figure()
        #plt.plot(t_mono_reshape[0:8000], data_mono_reshape[0:8000],c='r', label='Cantilever z Position', lw=4,)
        #plt.plot(t_mono_reshape[3000:4000], data_mono_reshape[3000:4000],c='r', label='Cantilever z Position', lw=4,)
        #plt.axvline(x=Canti.trigger, c='k', lw = 3) #this is the trigger
    
    #np.savetxt('thisisalinetest.txt',output_array,delimiter=',')
    #savestrline = 
    #np.save('nptest2.npy',output_array)
    return output_array#, output_array_noisy

def matrix_loop(number_of_rows, name_prefix):
    """requires a .txt file for each row of data that contains all the taus to be calculated for that row
    name_prefix is what the text part of that .txt file is
    e.g. if I have a 3x3 matrix,
    file1.txt = 
    10
    20
    30
    file2.txt = ...
    file3.txt = ...
    
    number_of_rows = 3
    name_prefix = 'file' """
    outputfile = np.array([])
    outputfile_noisy = np.array([])
    
    for i in range(1,number_of_rows+1):
        if i == 1:
            linestr = str(name_prefix) + str(i) + '.txt'
            print(linestr)
            #line, line_noisy = line_loop(linestr)
            line = line_loop(linestr)
        
            outputfile = np.append(outputfile,line)
            #outputfile_noisy = np.append(outputfile_noisy,line)
            
        else:
            linestr = str(name_prefix) + str(i) + '.txt'
            print(linestr)
            #line, line_noisy = line_loop(linestr)
            line = line_loop(linestr)
        
            outputfile = np.vstack((outputfile,line))
            #outputfile_noisy = np.vstack((outputfile_noisy,line))
    
    np.save('25kernel_1x1_outlier_v2.npy',outputfile)
   # np.save('25kernel_noisy_3x3_outlier_v1.npy',outputfile_noisy)
    return   

def main(tau_e, tau_i, beta_e = 1, beta_i = 1, electronic_fraction = 0.5, color = 'r'):
    """taus in above parameters are in us"""
    #Create cantilever object
    Canti = biexp("rise", tau_e * 1e-6, beta_e, tau_i * 1e-6, beta_i, electronic_fraction)
    
    #Generate synthetic z data for cantilever
    #data_mono, data_bi = Canti.run_simulation()
    data_bi = Canti.run_stretched_simulation()
    
    #Apply Gaussian noise
    data_bi_noisy = Canti.noise_on_z(data_bi,0.3e-9)
    
    #Reshape synthetic data to be appropriate resolution
    data_bi_reshape, t_bi_reshape = Canti.Z_reshape(data_bi) 
    data_noisy_bi_reshape, t_noisy_bi_reshape = Canti.Z_reshape(data_bi_noisy)
    
    #Save data if desired by uncommenting below lines and changing to desired name and data to be saved
    #np.savetxt('time.txt',t_mono_reshape,delimiter=',')
    np.savetxt('z_data_for_aaron.txt',data_bi_reshape,delimiter=',')
    np.savetxt('z_data_300_noisy.txt',data_noisy_bi_reshape,delimiter=',')
    
    #Optional plotting of data
    plt.plot(t_bi_reshape[3000:4000], data_bi_reshape[3000:4000],c=color, label='Cantilever z Position', lw=4,)
    #plt.plot(t_bi_reshape[0:8000], data_bi_reshape[0:8000],c=color, label='Cantilever z Position', lw=4,)
    
    #Plot trigger
    plt.axvline(x=Canti.trigger, c='k', lw = 3) #this is the trigger
    
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Cantilever z position about equilibrium', fontsize=10, x=-1.0)

    return

def gmode_sims(tau_e, tau_i, beta_e = 1, beta_i = 1, electronic_fraction = 0.5, color = 'r'):
    """taus in above parameters are in us
    also note this is super buggy and was just something I'm playing with for Raj.  
    THIS IS NOT FUNCTIONAL!!!!!!!!"""
    #Create cantilever object
    Canti = biexp("rise", tau_e * 1e-6, beta_e, tau_i * 1e-6, beta_i, electronic_fraction)
    
    #Generate synthetic z data for cantilever
    #data_mono, data_bi = Canti.run_simulation()
    data_bi = Canti.run_Gmode_simulation()
    
    #Apply Gaussian noise
    data_bi_noisy = Canti.noise_on_z(data_bi,0.3e-9)
    
    #Reshape synthetic data to be appropriate resolution
    data_bi_reshape, t_bi_reshape = Canti.Z_reshape(data_bi) 
    data_noisy_bi_reshape, t_noisy_bi_reshape = Canti.Z_reshape(data_bi_noisy)
    
    #Save data if desired by uncommenting below lines and changing to desired name and data to be saved
    #np.savetxt('time.txt',t_mono_reshape,delimiter=',')
    np.savetxt('z_data_for_aaron.txt',data_bi_reshape,delimiter=',')
    np.savetxt('z_data_300_noisy.txt',data_noisy_bi_reshape,delimiter=',')
    
    #Optional plotting of data
    #plt.plot(t_bi_reshape[3000:4000], data_bi_reshape[3000:4000],c=color, label='Cantilever z Position', lw=4,)
    #plt.plot(t_bi_reshape[0:8000], data_bi_reshape[0:8000],c=color, label='Cantilever z Position', lw=4,)
    force = np.sqrt(Canti.force_array())
    print(force[0])
    print(force[1])
    print(force[2])
    print(force[3])
    #plt.plot(force[3000:4000], data_bi_reshape[3000:4000],c=color, label='Cantilever z Position', lw=4,)
    t0 = int(Canti.simulation_trigger_index/20)
    plt.plot(force[t0:(t0+60)], data_bi_reshape[t0:(t0+60)],c=color, label='Cantilever z Position', lw=4,)
    #plt.plot(force[0:100], data_bi_reshape[0:100],c=color, label='Cantilever z Position', lw=4,)
    
    #Plot trigger
    #plt.axvline(x=Canti.trigger, c='k', lw = 3) #this is the trigger
    
    #plt.xlabel('Time (s)', fontsize=10)
    #plt.ylabel('Cantilever z position about equilibrium', fontsize=10, x=-1.0)
    plt.xlabel('Force (N)', fontsize=10)
    plt.ylabel('Cantilever z position about equilibrium', fontsize=10, x=-1.0)
    return


if __name__ == "__main__":
    #gmode_sims(50,300)
    #matrix_loop(25,'pca')
    #line_loop('row2.txt')
    main(10, 300, electronic_fraction = 1)