# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:19:26 2018

@author: Raj
"""

import numpy as np
import sklearn as sk
import pycroscopy as px
from pycroscopy.processing.cluster import Cluster
import pyUSID as usid
from pyUSID.processing.process import Process
from pyUSID.io.write_utils import build_ind_val_matrices, Dimension
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from ffta.analysis import mask_utils
from ffta.hdf_utils import hdf_utils
from ffta.load import get_utils
from ffta import pixel

"""
Creates a Class with various data grouped based on distance to masked edges

Typical usage:

>> mymask = mask_utils.load_mask_txt('Path_mask.txt')
>> tfp_clust = dist_cluster.dist_cluster(h5_main, mask=mymask, data_avg='tfp')
>> tfp_clust.analyze()
>> tfp_clust.plot_img()
>> tfp_clust.kmeans()
>> tfp_clust.plot_kmeans()

To do: 
	Implement dist_cluster as implementation of pycroscopy Cluster class
"""


class dist_cluster:

	def __init__(self, h5_main, data_avg, mask, results=None, isCPD=False, parms_dict=None):
		"""
		Parameters
		----------
		h5_main : h5Py dataset or str
			File type to be processed.

		mask : ndarray
			mask file to be loaded, is expected to be of 1 (transparent) and 0 (opaque)
			loaded e.g. mask = mask_utils.loadmask('E:/ORNL/20191221_BAPI/nice_masks/Mask_BAPI20_0008.txt')

		parms_dict : dict
			Parameters file used for the image. Must contain at minimum:
				num_rows, num_cols, sampling_rate, FastScanSize, SlowScanSize, total_time

		data_avg : str
			The file to use as the "averaged" data upon which to apply the mask and do calculations
			e.g. 'tfp' searches within the h5_main parent folder for 'tfp'
			
		results : clustering results
			If you want to pass some previously-calculated results..
			
		isCPD : bool, optional,
			toggle between GMode and FFtrEFM data, this is just for plotting

		"""
		self.h5_main = h5_main

		if isinstance(data_avg, str):
			self.data_avg = usid.hdf_utils.find_dataset(h5_main.parent, data_avg)[0].value
		elif isinstance(data_avg, np.ndarray):
			self.data_avg = usid.hdf_utils.find_dataset(h5_main.parent, 'tfp')[0].value
		else:
			raise ValueError('Wrong format for data_avg')

		if results:
			self.results = results

		self.isCPD = isCPD

		# Set up datasets data
		self.data = self.h5_main[()]
		self.parms_dict = parms_dict
		if parms_dict == None:
			print('get params')
			self.parms_dict = get_utils.get_params(h5_main)

		self._params()

		# Create mask for grain boundaries
		if not mask.any():
			mask = np.ones([self.num_rows, self.num_cols])

		self.mask = mask
		self.mask_nan, self.mask_on_1D, self.mask_off_1D = mask_utils.load_masks(self.mask)
		self._idx_1D = np.copy(self.mask_off_1D)

		return

	def _params(self):
		""" creates CPD averaged data and extracts parameters """

		parms_dict = self.parms_dict
		self.num_rows = parms_dict['num_rows']
		self.num_cols = parms_dict['num_cols']
		self.sampling_rate = parms_dict['sampling_rate']
		self.FastScanSize = parms_dict['FastScanSize']
		self.SlowScanSize = parms_dict['SlowScanSize']

		self.xvec = np.linspace(0, self.FastScanSize, self.num_cols)
		self.yvec = np.linspace(0, self.SlowScanSize, self.num_rows)

		#        IO_rate = parms_dict['IO_rate_[Hz]']     #sampling_rate
		self.pxl_time = parms_dict['total_time']  # seconds per pixel
		self.dt = self.pxl_time / self.data.shape[1]
		self.aspect = self.num_rows / self.num_cols

		return

	def analyze(self):
		"""
		Creates 1D arrays of data and masks
		Then, calculates the distances and saves those.

		This also creates data_scatter within the distances function

		"""
		# Create 1D arrays
		self._data_1D_values(self.mask)
		self._make_distance_arrays()

		self.data_dist, _ = self._distances(self.data_1D_pos, self.mask_on_1D_pos)

		return

	def _data_1D_values(self, mask):
		"""
		Uses 1D Mask file (with NaN and 0) and generates data of non-grain boundary regions

		Parameters
		----------
		mask : ndarray, 2D
			Unmasked locations (indices) as 1D location

		data_1D_vals : data as a 1D array with data points (num_rows*num_cols X pnts_per_data)
		data_avg_1D_vals : Average data (CPD_on_avg/tfP_fixed, say) that is 1D

		"""

		ones = np.where(mask == 1)

		self.data_avg_1D_vals = np.zeros(ones[0].shape[0])
		self.data_1D_vals = np.zeros([ones[0].shape[0], self.data.shape[1]])

		for r, c, x in zip(ones[0], ones[1], np.arange(self.data_avg_1D_vals.shape[0])):
			self.data_avg_1D_vals[x] = self.data_avg[r][c]
			self.data_1D_vals[x, :] = self.data[self.num_cols * r + c, :]

		return

	def plot_img(self):

		if self.data_avg is not None:
			fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
			_, cbar = usid.plot_utils.plot_map(ax, self.data_avg * 1e6,
											   x_vec=self.xvec * 1e6, y_vec=self.yvec * 1e6,
											   cmap='inferno', aspect=self.aspect)
			cbar.set_label('tFP (us)', rotation=270, labelpad=16)
			ax.imshow(self.mask_nan)

		return fig, ax

	def _make_distance_arrays(self):
		"""
		Generates 1D arrays where the coordinates are scaled to image dimensions

		Generates
		-------
		mask_on_1D_pos : ndarray Nx2
			Where mask is applied (e.g. grain boundaries)

		mask_off_1D_pos : ndarray Nx2
			Where mask isn't applied (e.g. grains)

		data_1D_pos : ndarray Nx2
			Identical to mask_off_1D_scaled, this exists just for easier bookkeeping
			without fear of overwriting one or the other

		"""

		csz = self.FastScanSize / self.num_cols
		rsz = self.SlowScanSize / self.num_rows

		mask_on_1D_pos = np.zeros([self.mask_on_1D.shape[0], 2])
		mask_off_1D_pos = np.zeros([self.mask_off_1D.shape[0], 2])

		for x, y in zip(self.mask_on_1D, np.arange(mask_on_1D_pos.shape[0])):
			mask_on_1D_pos[y, 0] = x[0] * rsz
			mask_on_1D_pos[y, 1] = x[1] * csz

		for x, y in zip(self.mask_off_1D, np.arange(mask_off_1D_pos.shape[0])):
			mask_off_1D_pos[y, 0] = x[0] * rsz
			mask_off_1D_pos[y, 1] = x[1] * csz

		self.data_1D_pos = np.copy(mask_off_1D_pos)  # to keep straight, but these are the same
		self.mask_on_1D_pos = mask_on_1D_pos
		self.mask_off_1D_pos = mask_off_1D_pos

		return

	def _distances(self, data_1D_pos, mask_on_1D_pos):
		"""
		Calculates pairwise distance between CPD array and the mask on array.
		For each pixel, this generates a minimum distance that defines the "closeness" to
		a grain boundary in the mask

		Returns to Class
		----------------
		data_scatter : distances x data_points (full data at each distance)
		data_avg_scatter : distances x 1 (data_average at each distance)
		data_dist : minimum pairwise distances to mask 
		data_avg_dist : mean of pairwise distances to maks

		"""
		self.data_dist = np.zeros(data_1D_pos.shape[0])
		self.data_avg_dist = np.zeros(data_1D_pos.shape[0])

		# finds distance to nearest mask pixel
		for i, x in zip(data_1D_pos, np.arange(self.data_dist.shape[0])):
			d = sk.metrics.pairwise_distances([i], mask_on_1D_pos)
			self.data_dist[x] = np.min(d)
			self.data_avg_dist[x] = np.mean(d)

		# create single [x,y] dataset
		self.data_avg_scatter = np.zeros([self.data_dist.shape[0], 2])
		for x, y, z in zip(self.data_dist, self.data_avg_1D_vals, np.arange(self.data_dist.shape[0])):
			self.data_avg_scatter[z] = [x, y]

		self.data_scatter = np.copy(self.data_1D_vals)
		self.data_scatter = np.insert(self.data_scatter, 0, self.data_dist, axis=1)

		return self.data_dist, self.data_avg_dist

	def write_results(self, verbose=False, name='inst_freq_masked'):
		''' Writes a new main data set'''

		h5_dist_clust_group = px.hdf_utils.create_indexed_group(self.h5_main.parent, 'dist-cluster')

		# Create dimensions
		pos_desc = [Dimension('Grain Distance', 'm', self.data_dist)]
		ds_pos_ind, ds_pos_val = build_ind_val_dsets(pos_desc, is_spectral=False, verbose=verbose)
		spec_desc = [Dimension('Time', 's', np.linspace(0, self.pxl_time, self.parms_dict['pnts_per_avg']))]
		ds_spec_inds, ds_spec_vals = build_ind_val_dsets(spec_desc, is_spectral=True, verbose=verbose)

		# Writes main dataset
		h5_clust = px.hdf_utils.write_main_dataset(h5_dist_clust_group,
												   self.data_scatter[:, 1:],
												   name,  # Name of main dataset
												   'Frequency',  # Physical quantity contained in Main dataset
												   'Hz',  # Units for the physical quantity
												   pos_desc,  # Position dimensions
												   spec_desc,  # Spectroscopic dimensions
												   dtype=np.float32,  # data type / precision
												   main_dset_attrs=self.parms_dict)

		# Adds mask and grain min distances and mean distances
		grp = px.io.VirtualGroup(h5_dist_clust_group.name)
		mask = px.io.VirtualDataset('mask', self.mask, parent=self.h5_main.parent)
		dist_min = px.io.VirtualDataset('dist_min', self.data_dist, parent=self.h5_main.parent)
		dist_mean = px.io.VirtualDataset('dist_mean', self.data_avg_dist, parent=self.h5_main.parent)
		data_pos = px.io.VirtualDataset('coordinates', self.mask_off_1D_pos, parent=self.h5_main.parent)
		data_avg = px.io.VirtualDataset('data_avg', self.data_avg_1D_vals, parent=self.h5_main.parent)

		grp.add_children([mask])
		grp.add_children([dist_min])
		grp.add_children([dist_mean])
		grp.add_children([data_pos])
		grp.add_children([data_avg])

		# Find folder, write to it
		hdf = px.io.HDFwriter(self.h5_main.file)
		h5_refs = hdf.write(grp, print_log=verbose)

		return h5_clust

	def kmeans(self, clusters=3, show_results=False, plot_mid=[]):

		""""
		Simple k-means

		Data typically is self.CPD_scatter
		
		Parameters
		----------
		plot_pts : list
			Index of where to plot (i.e. when light is on). Defaults to p_on:p_off

		Returns
		-------
		self.results : KMeans type

		self.segments : dict, Nclusters
			Contains the segmented arrays for displaying

		"""

		data = self.data_scatter[:, 1:]

		# create single [x,y] dataset
		estimators = sk.cluster.KMeans(clusters)
		self.results = estimators.fit(data)

		if show_results:
			ax, fig = self.plot_kmeans(plot_mid=plot_mid)

			return self.results, ax, fig

		return self.results

	def plot_kmeans(self, plot_mid=[]):

		labels = self.results.labels_
		cluster_centers = self.results.cluster_centers_
		labels_unique = np.unique(labels)
		self.segments = {}
		self.clust_tfp = []

		if not any(plot_mid):
			plot_mid = [0, int(self.data_scatter.shape[1] / 2)]

		# color defaults are blue, orange, green, red, purple...
		colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
				  '#d62728', '#9467bd', '#8c564b',
				  '#e377c2', '#7f7f7f', '#bcbd22',
				  '#17becf']

		fig, ax = plt.subplots(nrows=1, figsize=(8, 4))

		ax.tick_params(labelsize=15)

		for i in labels_unique:

			if not self.isCPD:
				# FFtrEFM data
				ax.plot(self.data_dist[labels == i] * 1e6,
						self.data_avg_scatter[labels == i, 1] * 1e6,
						c=colors[i], linestyle='None', marker='.')

				#                pix = pixel.Pixel(cluster_centers[i],self.parms_dict)
				#                pix.inst_freq = cluster_centers[i]
				#                pix.fit_freq_product()
				#                self.clust_tfp.append(pix.tfp)
				pix_tfp = np.mean(self.data_avg_scatter[labels == i, 1])
				ax.plot(np.mean(self.data_dist[labels == i] * 1e6),
						pix_tfp * 1e6,
						marker='o', markerfacecolor=colors[i], markersize=8,
						markeredgecolor='k')
				ax.set_xlabel('Distance to Nearest Boundary (um)', fontsize=16)
				ax.set_ylabel('tfp (us)', fontsize=16)

			elif self.isCPD:
				# CPD data
				ax.plot(self.data_dist[labels == labels_unique[i]] * 1e6,
						self.data_avg_scatter[labels == labels_unique[i], 1] * 1e3,
						c=colors[i], linestyle='None', marker='.')
				xp_0 = int(
					(self.parms_dict['light_on_time'][0] * 1e-3 / self.parms_dict['total_time']) * self.data_avg.shape[
						1])
				xp_1 = int(
					(self.parms_dict['light_on_time'][1] * 1e-3 / self.parms_dict['total_time']) * self.data_avg.shape[
						1])
				pix = cluster_centers[i][xp_0:xp_1]
				ax.plot(np.mean(self.data_dist[labels == labels_unique[i]] * 1e6),
						np.mean(pix) * 1e3,
						marker='o', markerfacecolor=colors[i], markersize=8,
						markeredgecolor='k')
				ax.set_xlabel('Distance to Nearest Boundary (um)', fontsize=16)
				ax.set_ylabel('CPD (mV)', fontsize=16)

		return fig, ax

	def plot_centers(self):

		fig, ax = plt.subplots(nrows=1, figsize=(6, 4))

		for i in self.results.cluster_centers_:
			ax.plot(np.linspace(0, self.parms_dict['total_time'], i.shape[0]), i)

		return fig, ax

	def elbow_plot(self, data=None, clusters=10):
		""""
		Simple k-means elbow plot, over 10 clusters. Note that this can take
		a long time for big data sets.

		Data defaults to self.data_scatter

		Returns
		-------
		score : KMeans type

		"""

		data = self.data_scatter if data is None else data

		Nc = range(1, clusters)
		km = [sk.cluster.KMeans(n_clusters=i) for i in Nc]

		score = [km[i].fit(data).score(data) for i in range(len(km))]

		fig, ax = plt.subplots(nrows=1, figsize=(6, 4))
		ax.plot(Nc, score, 's', markersize=8)
		ax.set_xlabel('Clusters')
		ax.set_ylabel('Score')
		fig.tight_layout()

		return score

	def segment_maps(self, results=None):

		"""
		This creates 2D maps of the segments to overlay on an image

		Returns to Class:
		
		segments is in actual length
		segments_idx is in index coordinates
		segments_data is the full CPD trace (i.e. vs time)
		segments_data_avg is for the average CPD value trace (not vs time)

		To display, make sure to do [:,1], [:,0] given row, column ordering
		Also, segments_idx is to display since pyplot uses the index on the axis

		"""

		if not results:
			results = self.results
		labels = results.labels_
		cluster_centers = results.cluster_centers_
		labels_unique = np.unique(labels)

		self.segments = {}
		self.segments_idx = {}
		self.segments_data = {}
		self.segments_data_avg = {}

		for i in range(len(labels_unique)):
			self.segments[i] = self.data_1D_pos[labels == labels_unique[i], :]
			self.segments_idx[i] = self._idx_1D[labels == labels_unique[i], :]
			self.segments_data[i] = self.data_1D_vals[labels == labels_unique[i], :]
			self.segments_data_avg[i] = self.data_avg_1D_vals[labels == labels_unique[i]]

		# the average value in that segment
		self.data_time_avg = {}
		for i in range(len(labels_unique)):
			self.data_time_avg[i] = np.mean(self.segments_data[i], axis=0)

		return

	def plot_segment_maps(self, ax, newImage=False):
		""" Plots the segments using a color map on given axis ax"""

		colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
				  '#d62728', '#9467bd', '#8c564b',
				  '#e377c2', '#7f7f7f', '#bcbd22',
				  '#17becf']

		if newImage:
			fig, ax = plt.subplots(nrows=1, figsize=(8, 6))
			im0, _ = usid.plot_utils.plot_map(ax, self.data_on_avg, x_vec=self.FastScanSize,
											  y_vec=self.SlowScanSize, show_cbar=False,
											  cmap='inferno')

		for i in self.segments_idx:
			im1, = ax.plot(self.segments_idx[i][:, 1], self.segments_idx[i][:, 0],
						   color=colors[i], marker='s', linestyle='None', label=i)

		ax.legend(fontsize=14, loc=[-0.18, 0.3])

		if newImage:
			return im0, im1

		return im1

	def heat_map(self, bins=50):

		"""
		Plots a heat map using CPD_avg_scatter data
		"""

		heatmap, _, _ = np.histogram2d(self.data_avg_scatter[:, 1], self.data_avg_scatter[:, 0], bins)

		fig, ax = plt.subplots(nrows=1, figsize=(8, 4))
		ax.set_xlabel('Distance to Nearest Boundary (um)')

		if not self.isCPD:
			ax.set_ylabel('tfp (us)')
			xr = [np.min(self.data_avg_scatter[:, 0]) * 1e6, np.max(self.data_avg_scatter[:, 0]) * 1e6]
			yr = [np.min(self.data_avg_scatter[:, 1]) * 1e6, np.max(self.data_avg_scatter[:, 1]) * 1e6]
			aspect = ((xr[1] - xr[0]) / (yr[1] - yr[0]))
			ax.imshow(heatmap, origin='lower', extent=[xr[0], xr[1], yr[0], yr[1]],
					  cmap='viridis', aspect=aspect)
			fig.tight_layout()
		else:
			ax.set_ylabel('CPD (mV))')
			xr = [np.min(self.data_avg_scatter[:, 0]) * 1e6, np.max(self.data_avg_scatter[:, 0]) * 1e6]
			yr = [np.min(self.data_avg_scatter[:, 1]) * 1e3, np.max(self.data_avg_scatter[:, 1]) * 1e3]
			aspect = ((xr[1] - xr[0]) / (yr[1] - yr[0]))
			ax.imshow(heatmap, origin='lower', extent=[xr[0], xr[1], yr[0], yr[1]],
					  cmap='viridis', aspect=aspect)
			fig.tight_layout()

		return fig, ax

	def animated_clusters(self, clusters=3, one_color=False):

		plt.rcParams[
			'animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'

		fig, a = plt.subplots(nrows=1, figsize=(13, 6))

		time = np.arange(0, self.pxl_time, 2 * self.dtCPD)
		idx = np.arange(1, self.CPD.shape[1], 2)  # in 2-slice increments

		colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
				  '#d62728', '#9467bd', '#8c564b',
				  '#e377c2', '#7f7f7f', '#bcbd22',
				  '#17becf']

		if one_color:
			colors = ['#1f77b4', '#1f77b4', '#1f77b4',
					  '#1f77b4', '#1f77b4', '#1f77b4',
					  '#1f77b4', '#1f77b4', '#1f77b4']

		a.set_xlabel('Distance to Nearest Boundary (um)')
		a.set_ylabel('CPD (V)')

		labels = self.results.labels_
		cluster_centers = self.results.cluster_centers_
		labels_unique = np.unique(labels)

		ims = []
		for t in idx:

			data = np.zeros([self.CPD_scatter.shape[0], 3])
			data[:, 0] = self.CPD_scatter[:, 0]
			data[:, 1:3] = self.CPD_scatter[:, t:t + 2]

			_results = self.kmeans(data, clusters=clusters)

			labels = _results.labels_
			cluster_centers = _results.cluster_centers_
			labels_unique = np.unique(labels)

			km_ims = []

			for i in range(len(labels_unique)):
				tl0, = a.plot(data[labels == labels_unique[i], 0] * 1e6, data[labels == labels_unique[i], 1],
							  c=colors[i], linestyle='None', marker='.')

				tl1, = a.plot(cluster_centers[i][0] * 1e6, cluster_centers[i][1],
							  marker='o', markerfacecolor=colors[i], markersize=8,
							  markeredgecolor='k')

				ims.append([tl0, tl1])

			km_ims = [i for j in km_ims for i in j]  # flattens
			ims.append(km_ims)

		ani = animation.ArtistAnimation(fig, ims, interval=120, repeat_delay=10)

		ani.save('kmeans_graph_.mp4')

		return

	def animated_image_clusters(self, clusters=5):
		"""
		Takes an image and animates the clusters over time on the overlay

		As of 4/3/2018 this code is just a placeholder

		"""

		plt.rcParams[
			'animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'

		fig, ax = plt.subplots(nrows=1, figsize=(13, 6))
		im0 = px.plot_utils.plot_map(ax, self.CPD_on_avg, x_size=self.FastScanSize,
									 y_size=self.SlowScanSize, show_cbar=False,
									 cmap='inferno')

		time = np.arange(0, self.pxl_time, 2 * self.dtCPD)
		idx = np.arange(1, self.CPD.shape[1], 2)  # in 2-slice increments

		ims = []
		for t in idx:
			data = np.zeros([self.CPD_scatter.shape[0], 3])
			data[:, 0] = self.CPD_scatter[:, 0]
			data[:, 1:3] = self.CPD_scatter[:, t:t + 2]

			_results = self.kmeans(data, clusters=clusters)
			self.segment_maps(results=_results)
			im1 = self.plot_segment_maps(ax)

			ims.append([im1])

		ani = animation.ArtistAnimation(fig, ims, interval=120, repeat_delay=10)

		ani.save('img_clusters_.mp4')

		return


def plot_clust(h5_main, labels, mean_resp, x_pts, data_avg=None, tidx_off=0):
	try:
		_labels = np.array(labels.value).T[0]
	except:
		_labels = np.array(labels)
	labels_unique = np.unique(_labels)
	parms_dict = hdf_utils.get_params(h5_main)
	clust_tfp = []

	# color defaults are blue, orange, green, red, purple...
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
			  '#d62728', '#9467bd', '#8c564b',
			  '#e377c2', '#7f7f7f', '#bcbd22',
			  '#17becf']

	fig, ax = plt.subplots(nrows=1, figsize=(12, 6))
	ax.set_xlabel('Distance to Nearest Boundary (um)')
	ax.set_ylabel('tfp (us)')

	for i in labels_unique:

		labels_tfp = []

		if not data_avg.any():

			for sig in h5_main[_labels == labels_unique[i], :]:
				pix = pixel.Pixel(sig, parms_dict)
				pix.inst_freq = sig
				pix.fit_freq_product()
				labels_tfp.append(pix.tfp)

			labels_tfp = np.array(labels_tfp)

		else:
			labels_tfp = data_avg[_labels == labels_unique[i]]

		ax.plot(x_pts[_labels == labels_unique[i]] * 1e6, labels_tfp * 1e6,
				c=colors[i], linestyle='None', marker='.')

		parms_dict['trigger'] += tidx_off / parms_dict['sampling_rate']
		pix = pixel.Pixel(mean_resp[i], parms_dict)
		pix.inst_freq = mean_resp[i]
		pix.fit_freq_product()
		clust_tfp.append(pix.tfp)
		parms_dict['trigger'] -= tidx_off / parms_dict['sampling_rate']

		ax.plot(np.mean(x_pts[_labels == labels_unique[i]] * 1e6), pix.tfp * 1e6,
				marker='o', markerfacecolor=colors[i], markersize=8,
				markeredgecolor='k')

	print('tfp of clusters: ', clust_tfp)

	return ax, fig
