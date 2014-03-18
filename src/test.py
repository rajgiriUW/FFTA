from utils import load, noise, cwavelet
from scipy import signal as sps

signal_array = load.ibw('D:/TestData/x03_680_preox_ND00/SW_0000.ibw')
n_pixels, parameters = load.configuration('D:/TestData/x03_680_preox_ND00/config.cfg')
import pixel
p = pixel.Pixel(parameters)
p.load_signals(signal_array)