import numpy as np
from scipy import integrate as sci


class DDHO(object):

    def __init__(self, sim_params, can_params, force_params):

        self.sim = sim_params
        self.can = can_params
        self.force = force_params

        self.w0 = 2 * np.pi * self.can['drive_freq']
        self.T0 = self.w0 * self.sim['trigger']
        self.Tau = self.w0 * self.force['tau']

        self.ratio = self.force['delta_freq'] / self.can['drive_freq']
        self.fd = self.force['drive'] / self.can['k'] / self.can['amp_invols']
        self.fe = self.force['electrostatic'] / self.can['k'] / self.can['amp_invols']

        self.T = self.w0 * np.arange(0, self.sim['total_time'],
                                     1.0 / self.sim['sampling_rate'])

        return

    def set_initial_conditions(self, z0, v0):

        self.Z0 = np.array([z0, v0])

        return

    def exp_decay(self, T):

        if T > self.T0:

            arg = -(T - self.T0) / self.Tau

            return (1 - np.exp(arg))

        else:

            return 0

    def gamma(self, T):

        exp_decay = np.vectorize(self.exp_decay, otypes=[np.float])

        return (1 + self.ratio * exp_decay(T))

    def F(self, T):

        exp_decay = np.vectorize(self.exp_decay, otypes=[np.float])

        return self.fd * np.cos(T) + self.fe * exp_decay(T)

    def dZ_dt(self, Z, T=0):

        a = self.gamma(T) / self.can['q_factor']
        b = self.gamma(T) ** 2

        z = Z[0]
        zdot = Z[1]
        vdot = self.F(T) - a * zdot - b * z

        return np.array([zdot, vdot])

    def solve(self, full_output=True):

        Z, infodict = sci.odeint(self.dZ_dt, self.Z0, self.T, full_output=True)

        if full_output:

            print infodict['message']

        self.z, self.v = Z.T

        return





