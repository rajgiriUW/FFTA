import pixel
from utils import load

signal_array = load.signal('C:/Users/DurmusU/Desktop/FF-trEFM/SW_0000.ibw')
n_pixels, parameters = load.configuration('C:/Users/DurmusU/Desktop/FF-trEFM/SW.cfg')

p = pixel.Pixel(signal_array, parameters)
tfp, shift = p.get_tfp()