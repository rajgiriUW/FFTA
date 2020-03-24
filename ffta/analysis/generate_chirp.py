# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:23:12 2020

@author: Raj
"""


import scipy.signal as sps
import numpy as np

def gen_chirp(f_center, f_width = 100e3, n_pts=1000):
    '''
    Based on the Agilent manual, the max-frequency is 250 MHz/number_of_points
    The minimum number of points is 8, maximum is 1e6
    
    The default case above yields a signal of max-frequency 1.25 MHz
    
    '''
    max_freq = np.floor(250e6 / n_pts)
    
    # if max_freq < f_center + f_width:
    #     raise ValueError('Choose lower number of points')
    
    f_hi = f_center + f_width
    f_lo = f_center - f_width

    tx = np.arange(0, 1e-2, 1/10e7)
    
    chirp = sps.chirp(tx, f_lo, tx[-1], f_hi)
    
    name = "chirp.dat"
    np.savetxt(name, chirp, delimiter='\n', fmt ='%.10f')
    
    return chirp


def GeneratePulse(pulse_time,voltage,total_time):
    
    sample_rate = 1.0e7
    total_samples = sample_rate * total_time
    pulse_samples = sample_rate * pulse_time
    
    data = np.zeros(total_samples)
    
    data[:pulse_samples] = voltage
    
    
    
    fo = open("Pulse.dat","wb")
    
    for i in range(int(total_samples)):
        fo.write(str(data[i])+"\r")
    fo.close()

def GenerateTaus(tau, beta, sfx =''):
    
    sample_rate = 1.0e8 # sampling rate used in Wavegenerator code
    total_samples = 800000
    pulse_samples = 700000
    
    data = np.arange(total_samples)/sample_rate
    
    data[:pulse_samples] = np.exp(-data[:pulse_samples]/tau)**beta - 1
    data[pulse_samples:] = 0
    
    name = "taub" + sfx + ".dat"
    np.savetxt(name, data, delimiter='\n', fmt ='%.10f')

if __name__ == '__main__':

	print 'Generating! This is slow, so hang on'
	GenerateTaus(1e-7, 0.4, '0')
	GenerateTaus(1e-7, 0.6, '1')
	GenerateTaus(1e-7, 0.8, '2')
	GenerateTaus(1e-6, 0.4, '3')
	GenerateTaus(1e-6, 0.6, '4')
	GenerateTaus(1e-6, 0.8, '5')
	GenerateTaus(1e-5, 0.4, '6')
	GenerateTaus(1e-5, 0.6, '7')
	GenerateTaus(1e-5, 0.8, '8')
	GenerateTaus(1e-4, 0.4, '9')
	GenerateTaus(1e-4, 0.6, '10')
	GenerateTaus(1e-4, 0.8, '11')