import numpy as np
from scipy.signal import  argrelextrema

def get_peaks(x):

	maxpeaks = argrelextrema(x, np.greater, order=2)
	minpeaks = argrelextrema(x, np.less, order=2)

	return maxpeaks[0], minpeaks[0]