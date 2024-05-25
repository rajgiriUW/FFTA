import cupy as cp
import numpy as np
import torch
from scipy.optimize import minimize

import time


class CUNFMD:
    def __init__(self, signal, num_freqs, window_size,
                 windows=None,
                 optimizer=torch.optim.SGD,
                 optimizer_opts={'lr': 1e-4},
                 max_iters=1000,
                 target_loss=1e-4,
                 device='cuda',
                 verbose=False,
                 zerodata=False,
                 drive_freq=None,
                 sampling_rate=None):
        '''
        Initialize the object

        :param signal: temporal signal to be analyzed (should be 1-D)
        :type signal: numpy.ndarray
        
        :param num_freqs: number of frequencies to fit to signal.
            (Note: The 'mean' mode counts as a frequency mode)
        :type num_freqs: integer
        
        :param window_size:
        :type window_size:
        
        :param windows:
        :type windows:
        
        :param optimizer: Optimization algorithm to employ for learning.
        :type optimizer: optimizer object (torch.optim)
        
        :param optimizer_opts: Parameters to pass to the optimizer class.
        :type optimizer_opts: dict
        
        :param max_iters: number of steps for optimizer to take (maximum)
        :type max_iters: int
        
         the loss value at which the window is considered sufficiently 'fit'
            (note: setting this too low can cause issues by pushing freqs to 0):param target_loss:
        :type target_loss: float
        
        :param device: device to use for optimization
            (Note: default 'cpu', but could be 'cuda' with GPU)
        :type device: string
            

        '''
        # Signal -- assumed 1D, needs to be type double
        self.x = signal.astype(cp.double).flatten()
        self.n = signal.shape[0]

        # Signal Decomposition options
        self.num_freqs = num_freqs
        self.window_size = window_size
        self.windows = windows

        if not windows:
            self.windows = self.n

        # Stochastic Gradient Descent Options
        self.optimizer = optimizer
        self.optimizer_opts = optimizer_opts
        # If the learning rate is specified, scale it by
        # window size
        if 'lr' in optimizer_opts:
            self.optimizer_opts['lr'] /= window_size
        self.max_iters = max_iters
        self.target_loss = target_loss
        self.device = device
        self.verbose = verbose
        self.zerodata = zerodata
        self.drive_freq = drive_freq
        self.sampling_rate = sampling_rate
        
    def decompose_signal(self, update_freq: int = None, method='sgd'):
        '''
        Compute the slices of the windows used in the analysis.
        Note: this is equivalent to computing rectangular windows.

        :param update_freq: The number of optimizer steps between printed update statements.
        :type update_freq: int
        
        :param method: Either 'sgd' or 'scipy' for gradient descent method
        :type method: str
        
        :returns: tuple (freqs, A, losses, indices)
            WHERE
            numpy.ndarray freqs is frequency vector
            numpy.ndarray A is coefficient vector
            numpy.ndarray losses is fit loss (MSE) for each window
            List indices is list of slice objects. each slice describes fit window indices`

        '''

        t1 = time.time()
        # self.compute_window_indices()
        
        window_size = self.window_size
        self.indices = [self.n-window_size+1]

        # Determine if printing updates
        verbose = self.verbose

        # lists for results
        self.freqs = []
        self.A = []
        self.losses = []
        self.window_fits = []  # Save the model fits

        # Tracker variables for previous freqs and A
        prev_freqs = None
        prev_A = None
        
        drive_freq = self.drive_freq
        sampling_rate = self.sampling_rate

        if verbose:
            print(len(self.indices), 'indices')
            print(method)
        # Determine number of SGD iterations to allow
        self.times = []
        
        # iterate through each window:
        for i in range(self.n-window_size+1):    
            t  = time.time()
            
            # If update frequency is requested, print an update
            # at window <x>
            if update_freq is not None:
                if i % update_freq == 0:
                    print("{}/{}".format(i, len(self.indices)), end="|")

            # Access data slice
            x_i = self.x[i:i+window_size]
            
            if self.zerodata:
                x_i = x_i - cp.mean(x_i)

            # Fit data in window to model
            if method == 'sgd':
                loss, freqs, A, i = self.fit_window(x_i,
                                                    freqs=prev_freqs,
                                                    A=prev_A,
                                                    drive_freq=drive_freq,
                                                    sampling_rate=sampling_rate)
            elif method == 'scipy':
                loss, freqs, A, i = self.calc_window(x_i,
                                                     freqs=prev_freqs,
                                                     A=prev_A,
                                                     drive_freq=drive_freq,
                                                     sampling_rate=sampling_rate)

            # Store the results
            self.freqs.append(freqs)
            self.A.append(A)
            self.losses.append(loss)

            # Set the previous freqs and A variables
            prev_freqs = freqs
            prev_A = A
            
            if self.verbose: 
                self.times.append(time.time() - t)
            
        if verbose:
            print(time.time() - t1, 's for decompose_signal')
        
        t2 = time.time()
        self.freqs = cp.array(self.freqs)
        self.A = cp.array(self.A)
        if method == 'sgd':
            if self.device == 'cpu':
                self.losses = [loss.detach().numpy() for loss in self.losses]
            else:
                self.losses = [loss.detach().cpu().numpy() for loss in self.losses]

        if verbose:
            print(time.time() - t2, 's for detach')

        return self.freqs, self.A, self.losses, self.indices

    def compute_window_indices(self):
        '''
        Auto-sets the windowed indices assuming single index steps
        '''
        
        t1 = time.time()

        window_size = self.window_size        
        self.indices = [slice(x, x + window_size, None) for x in range(self.n-window_size+1)]
        
        if self.verbose:                
            print(time.time() - t1, 's for compute')

    def calc_window(self, xt, freqs=None, A=None, max_iters=None, 
                    drive_freq=None, sampling_rate=1e7):
        '''
        Uses cost function minimization instead of SGD
        '''
        if freqs is None:
            if drive_freq is not None:
                freqs, A = self.fast_init(drive_freq, xt, sampling_rate)
            else:
                freqs, A = self.fft(xt)
            
        # Time indices
        tx = np.arange(len(xt))+1
        
        # Determine how many iterations will be used
        if not max_iters:
            max_iters = self.max_iters

        xt = xt.get()
        if isinstance(A, cp.ndarray):
            A = A.get()
        if isinstance(freqs, cp.ndarray):
            freqs = freqs.get()
        
        def omega_calc(freqs, tx):
            if freqs.ndim == 2:
                freqs = freqs[0,:]
            omega1 = np.vstack([[np.cos(tx * 2 * np.pi * freqs[x])] 
                                for x in range(self.num_freqs)])
            omega2 = np.vstack([[np.sin(tx * 2 * np.pi * freqs[x])] 
                                for x in range(self.num_freqs)])
            omega = np.vstack([omega1, omega2]).T
            return omega
            # return np.vstack( (np.cos(tx * 2 * np.pi * freqs[0]),
            #                    np.cos(tx * 2 * np.pi * freqs[1]),
            #                    np.sin(tx * 2 * np.pi * freqs[0]),
            #                    np.sin(tx * 2 * np.pi * freqs[1]))).T

        #cost = lambda p: np.sum(((omega_calc(p,tx) @ np.array(A)) - xt)**2)
        cost = lambda p: np.mean(((omega_calc(p,tx) @ np.array(A)) - xt)**2)
        
        try:
            pinit = [float(x.get()) for x in freqs]
        except:
            pinit = freqs
        popt = minimize(cost, pinit, method='TNC')
        freqs = popt.x
        
        omega = omega_calc(freqs, tx)
        A = np.matmul(np.linalg.pinv(omega), xt)
        xhat = omega @ A
        loss = np.mean((xhat - xt)**2)
        
        # Store the model fit:
        self.window_fits.append(xhat)
        self.popt = popt

        return loss, freqs, A, 0        

    def fit_window(self, xt, freqs=None, A=None, drive_freq=None, sampling_rate=1e7):
        '''
        Fits a set of instantaneous frequency and component coefficient vectors
        to the provided data.

        :param xt: Temporal data of dimensions [T, ...]
        :typer xt: numpy.ndarray
            
        :param freqs: 1D vector of (guess) instantaneous frequencies
            (Note: assumes dt=1 in xt data array)
        :type freqs: numpy.ndarray, optional
        
        :param A: 1D vector of cosine/sine coefficients
        :type A: numpy.ndarray, optional
        
        :returns: tuple (loss, freqs, A)
            WHERE
            float loss is the loss for the fit window (mean squared error)
            numpy.ndarray freqs is frequency vector of instantaneous frequencies
            numpy.ndarray A is coefficient vector of component (sine/cosine) coefficients

        '''
        # If no frequency is provided, generate initial frequency guess:
        if freqs is None:
            if drive_freq is not None:
                freqs, A = self.fast_init(drive_freq, xt, sampling_rate)
            else:
                freqs, A = self.fft(xt)

        # Then begin SGD
        loss, freqs, A, i = self.sgd(xt, freqs, A, max_iters=self.max_iters)

        return loss, freqs, A, i

    def fast_init(self, drive_freq, xt, sampling_rate = 1e7):
        '''
        Uses matrix math and given drive_frequency to initialize, since 
        these are usually known for our system        
        '''
        
        k = self.num_freqs
        freqs = cp.zeros(k)
        freqs[0] = drive_freq / sampling_rate
        tx = cp.arange(len(xt))+1
        
        omega1 = cp.vstack([[cp.cos(tx * 2 * cp.pi * freqs[x])] 
                            for x in range(self.num_freqs)])
        omega2 = cp.vstack([[cp.sin(tx * 2 * cp.pi * freqs[x])] 
                            for x in range(self.num_freqs)])
        omega = cp.vstack([omega1, omega2]).T
        # omega = cp.vstack( (cp.cos(tx * 2 * cp.pi * freqs[0]),
        #                     cp.cos(tx * 2 * cp.pi * freqs[1]),
        #                     cp.sin(tx * 2 * cp.pi * freqs[0]),
        #                     cp.sin(tx * 2 * cp.pi * freqs[1]))).T
        
        A = cp.matmul(cp.linalg.pinv(omega), xt)
        
        return freqs, A
        

    def fft(self, xt):
        '''
        Given temporal data xt, fft performs the initial guess of the
        frequencies contained in the data using the FFT.

        :param xt: Temporal data of dimensions [T, ...]
        :type xt: numpy.array

        :returns: tuple (freqs, A)
            WHERE
            numpy.ndarray freqs is vector of instantaneous frequency estimates for each timepoint
            numpy.ndarray A is vector of component coefficients

        '''
        
        t1 = time.time()
        # Ensure input signal is 1D:
        if len(xt.shape) == 1:
            xt = xt.reshape(-1, 1)

        # Gather model-fitting parameters
        k = self.num_freqs
        N = xt.shape[0]

        # Initialize a list of frequencies:
        freqs = []

        for i in range(k):

            if len(freqs) == 0:
                residual = xt
            else:
                t = cp.expand_dims(cp.arange(N) + 1, -1)
                ws = cp.asarray(freqs)
                Omega = cp.concatenate([cp.cos(t * 2 * cp.pi * ws),
                                        cp.sin(t * 2 * cp.pi * ws)], -1)
                A = cp.dot(cp.linalg.pinv(Omega), xt)

                pred = cp.dot(Omega, A)

                residual = pred - xt

            ffts = 0

            for j in range(xt.shape[1]):
                ffts += cp.abs(cp.fft.fft(residual[:, j])[:N // 2])

            w = cp.fft.fftfreq(N, 1)[:N // 2]
            idxs = cp.argmax(ffts)

            freqs.append(w[idxs])
            ws = cp.asarray(freqs)

            t = cp.expand_dims(cp.arange(N) + 1, -1)

            Omega = cp.concatenate([cp.cos(t * 2 * cp.pi * ws),
                                    cp.sin(t * 2 * cp.pi * ws)], -1)

            A = cp.dot(cp.linalg.pinv(Omega), xt)
        
        if self.verbose:
            print ('fft', time.time()-t1)
        return freqs, A

    def sgd(self, xt, freqs, A, max_iters=None):
        '''
        Given temporal data xt, sgd improves the initial guess of omega
        by SGD. It uses the pseudo-inverse to obtain A.

        :param xt: Temporal data of dimensions [T, ...]
        :type xt: numpy.ndarray
        
        :param freqs: frequency vector
        :type freqs: numpy.ndarray
        
        :param A: Component coefficient vector
        :type A: numpy.ndarray
            
        :param max_iters: Number of optimizer steps to take (maximum)
        :type max_iters:
            
        :returns: tuple (loss, freqs, A)
            WHERE
            float loss is the loss for the fit window (mean squared error)
            numpy.ndarray freqs is frequency vector of instantaneous frequencies
            numpy.ndarray A is coefficient vector of component (sine/cosine) coefficients

        '''
        # Set up PyTorch tensors for SGD
        A = torch.tensor(A, requires_grad=False, device=self.device)
        freqs = torch.tensor(cp.asarray(freqs), requires_grad=True, device=self.device)
        xt = torch.tensor(xt, requires_grad=False, device=self.device)

        # Set up PyTorch Optimizer
        o2 = self.optimizer([freqs], **self.optimizer_opts)

        # Time indices
        t = torch.unsqueeze(torch.arange(len(xt),
                                         dtype=torch.get_default_dtype(),
                                         device=self.device) + 1, -1)

        # Determine how many iterations will be used
        if not max_iters:
            max_iters = self.max_iters

        # Below is the method for using the Fourier->Koopman
        # SGD to determine solution
        for i in range(max_iters):
            # Compute new model
            Omega = torch.cat([torch.cos(t * 2 * cp.pi * freqs),
                               torch.sin(t * 2 * cp.pi * freqs)], -1)

            A = torch.matmul(torch.pinverse(Omega.data), xt)

            xhat = torch.matmul(Omega, A)

            # Compute Loss function
            loss = torch.mean((xhat - xt) ** 2)

            # Take a step
            o2.zero_grad()
            loss.backward()
            o2.step()

            # If loss is below fit threshold, end learning
            if loss < self.target_loss:
                break

        # Store the model fit:
        xhat = xhat.cpu().detach().numpy()
        self.window_fits.append(xhat)

        # Prepare the results
        A = A.cpu().detach().numpy()
        freqs = freqs.cpu().detach().numpy()

        return loss, freqs, A, i

    def predict(self, T):
        '''
        Predicts the data from 1 to T.

        :param T: Prediction horizon (number of timepoints T)
        :type T: int
            
        :returns: xhat from 0 to T.
        :rtype: numpy.array
            
        '''
        t = cp.expand_dims(cp.arange(T) + 1, -1)

        for i, idx_slice in enumerate(self.indices):
            local_freqs = self.freqs[i]
        Omega = cp.concatenate([cp.cos(t * 2 * cp.pi * self.freqs),
                                cp.sin(t * 2 * cp.pi * self.freqs)], -1)
        return cp.dot(Omega, self.A)

    def correct_frequencies(self, dt):
        '''
        Compute corrected frequency vector that takes into account the
        timestep dt in the signal

        :param dt: The time step between samples in the signal
        :type dt: float
            
        :returns: Timestamp-corrected frequency vector
        :rtype: numpy.ndarray
            
        '''
        corrected_freqs = []
        for freq in self.freqs:
            corrected_freqs.append(freq / dt)
        corrected_freqs = cp.asarray(corrected_freqs)
        return corrected_freqs

    def compute_amps(self):
        '''
        Compute the 'amplitude' of the Fourier mode.
        Amplitude = sqrt(A_1^2 + A_2^2)

        :returns: Amplitude vector, length = num_freqs
        :rtype: numpy.ndarray
            
        '''
        # initialize amps list
        Amps = cp.ndarray((self.A.shape[0], self.num_freqs))
        # print(Amps.shape)
        # Populate amps list
        for i, A in enumerate(self.A):
            # print(A.shape)
            # Reshape the As list into a 2 x k matrix of
            # cosine and sine coefficients
            AsBs = A.reshape(-1, self.num_freqs)
            # Compute amplitude of each mode:
            for j in range(AsBs.shape[-1]):
                Amp = complex(*AsBs[:, j])
                Amps[i, j] = abs(Amp)
        Amps = cp.asarray(Amps)
        return Amps

    def compute_mean(self, lf_mode=None):
        '''
        Computes the value of the mean mode. The mode is constructed by
        taking the value of the fit mean mode at the center of the window
        for each window data was fit in, and concatenating the center values.

        :param lf_mode: The index of the mode that is known to represent the mean.
            Note: if not provided, the lowest-average-IF mode is assumed to be
            the mean.
        :type lf_mode: optional, integer
            
        :returns: The reconstructed mean signal at each time point.
        :rtype: numpy.ndarray
            
        '''
        # Initialize empty array
        means = cp.ndarray(len(self.mid_idcs))
        
        # Identify the low-frequency mode based on initial frequency estimate
        if lf_mode is None:
            lf_mode = cp.argmin(cp.mean(self.freqs[:, :], axis=0))
        mid_idx = int(self.window_size / 2)
        
        # Iterate through each fourier object and compute the mean
        for i in range(len(self.mid_idcs)):
        
            # Grab the frequency and the amplitudes
            freq = self.freqs[i, lf_mode]
            A = self.A[i, lf_mode::self.num_freqs]
            
            # Compute the estimate
            t = cp.expand_dims(cp.arange(self.window_size) + 1, -1)
            Omega = cp.concatenate([cp.cos(t * 2 * cp.pi * freq),
                                    cp.sin(t * 2 * cp.pi * freq)], -1)
            fit = cp.dot(Omega, A)
            
            # Grab the centerpoint and add it to the means list
            means[i] = fit[mid_idx]
        
        return means

    def predict_window(self, i):
        '''
        Show the sum of the modes fit to a window of index i.

        :param i: index of the window to retrieve the fit for.
        :type i: integer
            
        :returns: The sum of the reconstructed modes for the given window.
        :rtype: numpy.ndarray
            
        '''
        return self.window_fits[i]
