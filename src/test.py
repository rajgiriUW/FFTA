import pixel
from utils import load

signal_array = load.signal('C:/Users/DurmusU/Downloads/scopeWaveAvg4.txt', skiprows=0)
n_pixels, parameters = load.configuration('C:/Users/DurmusU/SkyDrive/Research/FF-trEFM/SW.cfg')

p = pixel.Pixel(signal_array, parameters)
tfp, shift = p.get_tfp()
