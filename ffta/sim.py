import numpy as np
from scipy import integrate as sci


class DDHO(object):

    def __init__(self, sim_params, can_params, force_params):

        self.sim = sim_params
        self.can = can_params
        self.force = force_params

        self.w0 = 2 * np.pi * self.can['drive_freq']
        self.T0 = self.w0 * self.sim['trigger'] * 2
        self.Tau = self.w0 * self.force['tau']

        mw2 = (self.can['k'] * self.can['amp_invols'])
        self.ratio = self.force['delta_freq'] / self.can['drive_freq']
        self.fd = self.force['drive'] / mw2
        self.fe = self.force['electrostatic'] / mw2

        self.T = self.w0 * np.arange(0, 2 * self.sim['total_time'],
                                     1.0 / self.sim['sampling_rate'])

        z0 = self.can['amplitude'] / self.can['amp_invols'] * np.sin(0)
        v0 = self.can['amplitude'] / self.can['amp_invols'] * np.cos(0)

        self.Z0 = np.array([z0, v0])

        return

    def set_trigger_phase(self, delta):

        current_phase = np.mod(self.T0, 2 * np.pi)
        diff = delta - current_phase

        self.T0 = self.T0 + diff

        return

    def exp_decay(self, T):

        if T > self.T0:

            arg = -(T - self.T0) / self.Tau

            return 1 - np.exp(arg)

        else:

            return 0

    def gamma(self, T):

        exp_decay = np.vectorize(self.exp_decay, otypes=[np.float])

        return 1 + self.ratio * exp_decay(T)

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

        tidx = np.searchsorted(self.T, self.T0)
        before_trigger = self.sim['trigger'] * self.sim['sampling_rate']
        total = self.sim['total_time'] * self.sim['sampling_rate']

        z = self.z[(tidx - before_trigger):(tidx + total - before_trigger)]
        v = self.v[(tidx - before_trigger):(tidx + total - before_trigger)]

        return z, v
