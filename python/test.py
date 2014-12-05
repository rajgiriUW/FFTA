import pixel
import logging

from utils import load
from matplotlib import pyplot as plt

logging.basicConfig(filename='C:/Users/DurmusU/SkyDrive/Research/FF-trEFM/example.log', level=logging.DEBUG, filemode='w')

signal_file = 'C:/Users/DurmusU/SkyDrive/Research/FF-trEFM/SW_0000.ibw'
params_file = 'C:/Users/DurmusU/SkyDrive/Research/FF-trEFM/SW.cfg'

signal_array = load.signal(signal_file)
n_pixels, params = load.configuration(params_file)

p = pixel.Pixel(signal_array, params)
p.get_tfp()

plt.plot(p.inst_freq[(p.tidx - 2000):(p.tidx + 2000)])