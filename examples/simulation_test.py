from ffta import pixel, sim
from ffta.utils import load
from matplotlib import pyplot as plt

path = 'parameters.cfg'
sim_params, can_params, force_params = load.simulation_configuration(path)

ddho = sim.DDHO(sim_params, can_params, force_params)
ddho.set_trigger_phase(0)
z, v = ddho.solve()

plt.figure()
plt.plot(z)

params = {"trigger": .0004096, "total_time": .0008192, "sampling_rate": 1e7,
          "drive_freq": 272325, "roi": .0003, "window": 'blackman',
          "bandpass_filter": 1, "filter_bandwidth": 10000, "n_taps": 999}

p = pixel.Pixel(z.reshape(8192, 1), params)
p.analyze()


plt.figure()
plt.plot(p.inst_freq[4096:8000])
