from ffta import simulation, pixel
from ffta.utils import load
from matplotlib import pyplot as plt

path = 'sim_parameters.cfg'
can_params, force_params, sim_params = load.simulation_configuration(path)

c = simulation.Cantilever(can_params, force_params, sim_params)
c.simulate(trigger_phase=0)

parameters = {'bandpass_filter': 1.0,
              'drive_freq': 277261,
              'filter_bandwidth': 10000.0,
              'n_taps': 499,
              'roi': 0.0003,
              'sampling_rate': 1e7,
              'total_time': 0.0008192,
              'trigger': 0.0004096,
              'window': 'blackman',
              'wavelet_analysis': 0}

p = pixel.Pixel(c.Z, parameters)
p.analyze()

plt.figure()
plt.plot(p.inst_freq[p.tidx:(p.tidx + 2000)])