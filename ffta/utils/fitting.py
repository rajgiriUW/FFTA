       
import numpy as np
import scipy.optimize as spo



def fit_tfp(Q,drive_freq,t,inst_freq):

        def __fit_func__(t, A, tau1, tau2):

            decay = np.exp(-t / tau1)
            relaxation = np.expm1(-t / tau2)
            
            return -A * decay * relaxation

        # Initial relaxation constant
        beta = Q / (np.pi * drive_freq)

        # mean squared error wrt.
        err = lambda p: np.sum((__fit_func__(t,*p)-inst_freq)**2)

        # bounded optimization using scipy.minimize
        p_init = [inst_freq.min(), 1e-4, beta]
         
        p_opt, n_eval, returncode = spo.fmin_tnc(err,p_init,approx_grad=True,
                                     bounds = [(None,-1.0), (5e-7, 0.1), (1e-5, 0.1)])


        return p_opt