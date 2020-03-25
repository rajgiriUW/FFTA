# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:23:12 2020

@author: Raj
"""


import scipy.signal as sps
import numpy as np
import argparse

def GenChirp(f_center, f_width = 100e3):
    '''
    Based on the Agilent manual, the max-frequency is 250 MHz/number_of_points
    The minimum number of points is 8, maximum is 1e6
		
	This creates 3 chirp signals around the first three mechanical resonances
    '''


    tx = np.arange(0, 1e-2, 1/10e7) # fixed 10 MHz sampling rate
	
    f_hi = f_center + f_width
    f_lo = np.max([f_center - f_width, 100]) # to ensure a positive number
   
    print(f_lo, 'to',f_hi)
    chirp = sps.chirp(tx, f_lo, tx[-1], f_hi)
    
    name = "chirp_w.dat"  # first electrical resonance
    np.savetxt(name, chirp, delimiter='\n', fmt ='%.10f')

    f_hi = 2*f_center + f_width
    f_lo = np.max([2*f_center - f_width, 100]) # to ensure a positive number
   
    print(f_lo, 'to',f_hi)
    chirp = sps.chirp(tx, f_lo, tx[-1], f_hi)
    
    name = "chirp_2w.dat" # second electrical resonance
    np.savetxt(name, chirp, delimiter='\n', fmt ='%.10f')
	
    f_hi = 3*f_center + f_width
    f_lo = np.max([3*f_center - f_width, 100]) # to ensure a positive number
   
    print(f_lo, 'to',f_hi)
    chirp = sps.chirp(tx, f_lo, tx[-1], f_hi)
    
    name = "chirp_3w.dat"  # third electrical resonance
    np.savetxt(name, chirp, delimiter='\n', fmt ='%.10f')
    
    f_hi = 6.25*f_center + f_width
    f_lo = np.max([6.25*f_center - f_width, 100]) # to ensure a positive number
   
    print(f_lo, 'to',f_hi)
    chirp = sps.chirp(tx, f_lo, tx[-1], f_hi)
    
    name = "chirp_w2.dat"  # second mechanical resonance
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
    
    '''
    From command line, usage:
        python generate_chirp.py 350000 100000 1000
        
        Generates a 350 kHz +/- 100 kHz chirp 1000 points long. This would be ~14 MB on disk
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('freq_center', help='Resonance Frequency (Hz)')
    parser.add_argument('freq_width', help='Frequency width (Hz)')

    f_center = float(parser.parse_args().freq_center)
    f_width = float(parser.parse_args().freq_width)
    chirp = GenChirp(f_center, f_width)