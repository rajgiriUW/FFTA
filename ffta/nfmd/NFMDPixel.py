import numpy as np

from .NFMD import NFMD


class NFMDMode:

    def __init__(self, IF, IA, A, t):
        '''
        Initialize an NFMDMode object containing the data relevant to
        a Fourier mode in the data. This is a data storage object.
        
        :param IF: frequency vector of instantaneous frequencies, one for each time in t
        :type IF: float
        
        :param IA: vector of instantaneous amplitudes
        :type IA: numpy.ndarray
        
        :param A: vector of fourier mode coefficients
            (note: different from amplitude, but related)
        :type A: numpy.ndarray
            
        :param t: time vector, same length as the IF and IA vectors.
        :type t: numpy.ndarray
            
        '''
        self.IF = IF
        self.IA = IA
        self.A = A
        self.t = t


class NFMDPixel:

    def __init__(self, signal,
                 nfmd_options={'num_freqs': 3,
                               'window_size': 320}):

        '''
        Initialize an NFMDPixel object for decomposing a signal into Fourier modes.
    
        :param signal: Temporal signal to be analyzed.
        :type signal: numpy.ndarray
    
        :param nmfd_options: options passed to the NFMD analysis class
        :type nfmd_options: dict
        
        '''
        # Signal
        self.signal = signal

        # self.signal = (signal-np.mean(signal))
        # self.signal /= np.std(signal)

        # Signal Decomposition options
        self.nfmd_options = nfmd_options

    def analyze(self, dt=1, update_freq: int = None):
        '''
        Initialize an NFMDMode object containing the data relevant to
        a Fourier mode in the data. This is a data storage object.
        
        :param dt: timestep between datapoints in the signal array
        :type dt: float, optional
        
        :param update_freq:
        :type update_freq:
        '''
        # Initialize the NFMD object
        nfmd = NFMD(self.signal, **self.nfmd_options)
        t = np.arange(nfmd.n) * dt

        # Decompose the signal using NFMD
        freqs, A, losses, indices = nfmd.decompose_signal(update_freq)

        # Compute corrected frequencies (scaled by dt) and instantaneous amplitudes
        self.freqs = nfmd.correct_frequencies(dt=dt)
        self.amps = nfmd.compute_amps()

        # Slice for each window, and then the center point of each window!
        self.indices = indices
        self.mid_idcs = nfmd.mid_idcs

        # Compute frequencies, amplitudes and mean
        self.mean = nfmd.compute_mean()
        self.mean_t = t[nfmd.mid_idcs]

        # Organize the other modes into Mode objects:
        # Compute mean IF for each mode over entire time vector
        mean_freqs = np.mean(freqs, axis=0)

        # Organize the other modes:
        self.modes = []
        for i in range(nfmd.num_freqs):
            # If it's not the lowest-freq mode (assumed to be the mean)
            if i != np.argmin(mean_freqs):
                # Extract IFs, IAs, and A vector for the mode:
                IF = self.freqs[:, i]
                IA = self.amps[:, i]
                A = A[:, i::nfmd.num_freqs]
                # Initialize the Mode object:
                mode = NFMDMode(IF, IA, A, t[nfmd.mid_idcs])
                # Store the mode in the 
                self.modes.append(mode)
