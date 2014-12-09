import os
import pixel
import logging

from utils import load
from matplotlib import pyplot as plt

cwd = os.getcwd()
log_file = os.path.join(cwd, 'example.log')

logging.basicConfig(level=logging.DEBUG,
                    format='\n%(asctime)s - %(levelname)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    filename=log_file,
                    filemode='w')


signal_file = 'C:/Users/DurmusU/SkyDrive/Projects/FFTA/data/SW_0000.ibw'
params_file = 'C:/Users/DurmusU/SkyDrive/Projects/FFTA/data/SW.cfg'

signal_array = load.signal(signal_file)
n_pixels, params = load.configuration(params_file)

p = pixel.Pixel(signal_array, params)
p.analyze()

plt.plot(p.inst_freq[(p.tidx - 2000):(p.tidx + 2000)])
