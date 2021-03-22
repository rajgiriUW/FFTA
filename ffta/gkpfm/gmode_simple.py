# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:07:34 2018

@author: Raj
"""

import numpy as np
import ffta

from scipy import signal as sig

class F3R(object):
	"""
	Simple G-Mode analyzer for F3R. This does not take into account transfer
	function of the tip at all.
	
	7/25/2018 : hard coded for only single pixel work
	
	"""
	
	def __init__(self, signal_array, params, n_pixels=1):
		
		for key, value in params.items():

			setattr(self, key, value)
		
		self.N_points = n_pixels
		self.N_points_per_pixel = self.sampling_rate * self.total_time
		self.time_per_osc = 1/self.drive_freq
		self.pnts_per_period = self.sampling_rate * self.time_per_osc
		self.pxl_time = self.N_points_per_pixel/self.sampling_rate
		self.num_periods = int(self.pxl_time/self.time_per_osc)
		
		xpts = np.arange(0,self.total_time, 1/self.sampling_rate)[:-1]
		self.pixel_ex_wfm = np.sin(xpts * self.drive_freq*2*np.pi)
		
		self.signal = signal_array
		
		if len(self.signal.shape) != 1: # needs to be averaged
			
			self.signal = self.signal.mean(axis=0)
		
		self.signal -= np.mean(self.signal) #DC offset
		
		return
	
	def t_div(self):
		'''
		Simple divide by the transfer function to get Force. Doesn't really work
		'''
	
		h = self.pixel_ex_wfm
		y = self.signal
		
		H = np.fft.fftshift(np.fft.fft(h))
		Y = np.fft.fftshift(np.fft.fft(y))
		F = np.divide(H,Y)
		
		self.signal = np.real(np.fft.ifft(np.fft.ifftshift(F)))
		
		return    
	
	def lia(self, tc=2):
		'''
		Simple lockin integration
		
		time constant tc = number of points
		
		Does not auto calculate for you based on time. 
		'''
		
		xpts = np.arange(0,self.total_time, 1/self.sampling_rate)[:-1]
		sini = np.sin(2*np.pi*self.drive_freq*xpts)
		cosi = np.cos(2*np.pi*self.drive_freq*xpts)
		
		UoutX = self.signal * sini
		UoutY = self.signal * cosi
		
		X = np.array([])
		Y = np.array([])
		for i in np.arange(0,int(self.N_points_per_pixel), tc):
			
			t = xpts[i:i+tc]
			Uout = np.trapz(UoutX[i:i+tc], t)
			X = np.append(X,Uout)
			Uout = np.trapz(UoutY[i:i+tc], t)
			Y = np.append(Y,Uout)
		
		self.X = X
		self.Y = Y
		self.amp = np.sqrt(X**2 + Y**2)
		self.phase = np.arctan(Y/X)
		
		return
	
	def analyze(self, periods=4):
	 
		complete_periods = True
		 
		num_periods_per_sample = int(np.floor(self.num_periods / periods))
		pnts_per_sample = int(np.floor(self.pnts_per_period * periods))
		
		if complete_periods == False:
 
			# new approach since it's base-2 samples and can curve-fit to less than full cycle
			decimation = 2**int(np.floor(np.log2(pnts_per_sample)))
			self.pnts_per_CPDpix = int(self.N_points_per_pixel/decimation)
			remainder = 0
			
		else:

			# old approach, but add section for missing period at the end
			decimation = int(np.floor(pnts_per_sample))
			self.pnts_per_CPDpix = int(self.N_points_per_pixel/decimation)
			remainder = self.N_points_per_pixel - self.pnts_per_CPDpix*decimation
		
		# second degree polynomial fitting
		deg = 2
#        deg = 1
	   
		# Holds all the fit parameters
		self.wHfit3 = np.zeros((1, self.pnts_per_CPDpix, deg+1))
		
		for p in range(self.pnts_per_CPDpix-min(1,remainder)): 

			resp = np.float32(self.signal[decimation*p:decimation*(p+1)])
			resp = resp-np.mean(resp)
			V_per_osc = self.pixel_ex_wfm[decimation*p:decimation*(p+1)]
			popt, _ = np.polynomial.polynomial.polyfit(V_per_osc, resp, deg, full=True)
			self.wHfit3[0,p,:] = popt
		
		if remainder > 0:

			resp = np.float32(self.signal[(self.pnts_per_CPDpix-1)*decimation:])
			resp = (resp-np.mean(resp))
			V_per_osc = self.pixel_ex_wfm[(self.pnts_per_CPDpix-1)*decimation:]
			popt, _ = np.polynomial.polynomial.polyfit(V_per_osc, resp, deg, full=True)
		
		self.wHfit3[0,-1,:] = popt
		
		self.CPD = -0.5*np.divide(self.wHfit3[:,:,1],self.wHfit3[:,:,2])[0,:]
#        self.CPD = self.wHfit3[:,:,1][0,:]
		
		return
	
	def smooth(self, kernel):
		
		self.CPD_filt = np.convolve(self.CPD, kernel, mode='valid')
		
		return
		