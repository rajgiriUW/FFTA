# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:19:26 2018

@author: Raj
"""

import numpy as np
import sklearn as sk
import pycroscopy as px
from pycroscopy.processing.cluster import Cluster
from pycroscopy.processing.process import ProcessIf this deck goes anywhere are we going to call it
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import mask_utils

""" 
Creates a Class with various CPD data grouped based on distance to a grain boundary

"""

class CPD_cluster(object):
    
    def __init__(self, h5_file, mask=None, imgsize=[32e-6,8e-6], 
                 ds_group='/Measurement_000/Channel_000/Raw_Data-CPD',
                 light_on=[1,5]):
        """
    
        h5_file : h5Py File or str
            File type to be processed.
            
        mask : ndarray
            mask file to be loaded, is expected to be of 1 (transparent) and 0 (opaque)
            loaded e.g. mask = mask_utils.loadmask('E:/ORNL/20191221_BAPI/nice_masks/Mask_BAPI20_0008.txt')
        
        imgsize : list
            dimensions of the image
        
        ds_group : str or H5Py Group
            Where the target dataset is located
    
        light_on : list
            the time in milliseconds for when light is on
    
        """    
        hdf = px.ioHDF5(h5_file)
        self.h5_main = hdf.file[ds_group]
        
        self.FastScanSize = imgsize[0]
        self.SlowScanSize = imgsize[1]
        self.light_on_time = light_on

        # Set up CPD data
        self.CPD = self.h5_main['CPD'].value
        self.CPD_orig = self.h5_main['CPD']
        self.pnts_per_CPDpix = self.CPD.shape[1]
        
        self.CPD_params()
        
        # Create mask for grain boundaries
        
        # 
        if not mask.any():
            mask = np.ones([self.num_rows, self.num_cols])
        
        self.mask = mask
        self.mask_nan, self.mask_on_1D, self.mask_off_1D = mask_utils.load_masks(self.mask)
        self.CPD_1D_idx = np.copy(self.mask_off_1D)
        
        return

    def CPD_params(self):
        """ creates CPD averaged data and extracts parameters """
        CPD_on_time = self.h5_main['CPD_on_time']
        
        self.CPD_off_avg = np.zeros(CPD_on_time.shape)
        self.CPD_on_avg = np.zeros(CPD_on_time.shape)
        parms_dict = self.h5_main.parent.parent.attrs
        self.num_rows = parms_dict['grid_num_rows']
        self.num_cols = parms_dict['grid_num_cols']
        
        N_points_per_pixel = parms_dict['num_bins']
        IO_rate = parms_dict['IO_rate_[Hz]']     #sampling_rate
        self.pxl_time = N_points_per_pixel/IO_rate    #seconds per pixel
        
        self.dtCPD = self.pxl_time/self.CPD.shape[1] 
        p_on = int(self.light_on_time[0]*1e-3 / self.dtCPD) 
        p_off = int(self.light_on_time[1]*1e-3 / self.dtCPD) 
        
        self.p_on = p_on
        self.p_off = p_off
        
        for r in np.arange(CPD_on_time.shape[0]):
            for c in np.arange(CPD_on_time.shape[1]):
                
                self.CPD_off_avg[r][c] = np.mean(self.CPD[r*self.num_cols + c,p_off:])
                self.CPD_on_avg[r][c] = np.mean(self.CPD[r*self.num_cols + c,p_on:p_off])
        
        return
        
    def analyze_CPD(self, CPD_avg):
        """ 
        Creates 1D arrays of data and masks 
        Then, calculates the distances and saves those.
        
        This also creates CPD_scatter within the distances function
        
        """
        # Create 1D arrays 
        self.CPD_values(CPD_avg,self.mask)
        self.make_distance_arrays()
        
        self.CPD_dist, _ = self.CPD_distances(self.CPD_1D_pos, self.mask_on_1D_pos)
        
        return

    def CPD_values(self, CPD_avg, mask):
        """
        Uses 1D Mask file (with NaN and 0) and generates CPD of non-grain boundary regions
        
        h5_file : H5Py File
            commonly as hdf.file
            
        CPD : ndarray
            RowsXColumns matrix of CPD average values such as CPD_on_avg
            
        mask : ndarray, 2D
            Unmasked locations (indices) as 1D location
        
        CPD_loc : str, optional
            The path to the dataset within the h5_file
            
        CPD_1D_vals : CPD as a 1D array with pnts_per_CPD points
        CPD_avg_1D_vals : Average CPD (CPD_on_avg, say) that is 1D
        
        """
            
        ones = np.where(mask == 1)
        self.CPD_avg_1D_vals = np.zeros(ones[0].shape[0])
        self.CPD_1D_vals = np.zeros([ones[0].shape[0], self.CPD.shape[1]])
    
        for r,c,x in zip(ones[0], ones[1], np.arange(self.CPD_avg_1D_vals.shape[0])):
            self.CPD_avg_1D_vals[x] = CPD_avg[r][c]
            self.CPD_1D_vals[x,:] = self.CPD[self.num_cols*r + c,:]
    
        return 

    def make_distance_arrays(self):
        """
        Generates 1D arrays where the coordinates are scaled to image dimenions
        
        pos_val : ndarray of H5Py Dataset
            This is the Position_Values generated by pycroscopy. The last element
            contains the size of the image. Use add_position_sets to generate
            this in the folder
            
            can also be [size_x, size-y]
        
        Returns
        -------
        mask_on_1D_pos : ndarray Nx2
            Where mask is applied (e.g. grain boundaries)
            
        mask_off_1D_pos : ndarray Nx2
            Where mask isn't applied (e.g. grains)
        
        CPD_1D_pos : ndarray Nx2
            Identical to mask_off_1D_scaled, this exists just for easier bookkeeping
            without fear of overwriting one or the other
        
        """
        
        csz = self.FastScanSize / self.num_cols
        rsz = self.SlowScanSize / self.num_rows
              
        mask_on_1D_pos = np.zeros([self.mask_on_1D.shape[0],2])
        mask_off_1D_pos = np.zeros([self.mask_off_1D.shape[0],2])
        
        for x,y in zip(self.mask_on_1D, np.arange(mask_on_1D_pos.shape[0])):
            mask_on_1D_pos[y,0] = x[0] * rsz
            mask_on_1D_pos[y,1] = x[1] * csz
            
        for x,y in zip(self.mask_off_1D, np.arange(mask_off_1D_pos.shape[0])):
            mask_off_1D_pos[y,0] = x[0] * rsz
            mask_off_1D_pos[y,1] = x[1] * csz
        
        CPD_1D_pos = np.copy(mask_off_1D_pos) # to keep straight, but these are the same
        
        self.mask_on_1D_pos = mask_on_1D_pos
        self.mask_off_1D_pos = mask_off_1D_pos
        self.CPD_1D_pos = CPD_1D_pos
        
        return


    def CPD_distances(self,CPD_1D_pos, mask_on_1D_pos):
        """
        Calculates pairwise distance between CPD array and the mask on array.
        For each pixel, this generates a minimum distance that defines the "closeness" to 
        a grain boundary in the mask
        
        Returns to Class
        ----------------
        CPD_scatter : distances x CPD_points (full CPD at each distance)
        CPD_avg_scatter : distances x 1 (CPD_average at each distance)
        
        """
        CPD_dist = np.zeros(CPD_1D_pos.shape[0])
        CPD_avg_dist = np.zeros(CPD_1D_pos.shape[0])
        
        # finds distance to nearest mask pixel
        for i, x in zip(CPD_1D_pos, np.arange(CPD_dist.shape[0])):
            
            d = sk.metrics.pairwise_distances([i], mask_on_1D_pos)
            CPD_dist[x] = np.min(d)
            CPD_avg_dist[x] = np.mean(d)
        
        # create single [x,y] dataset
        self.CPD_avg_scatter = np.zeros([CPD_dist.shape[0],2])
        for x,y,z in zip(CPD_dist, self.CPD_avg_1D_vals, np.arange(CPD_dist.shape[0])):
            self.CPD_avg_scatter[z] = [x, y]

        self.CPD_scatter = np.copy(self.CPD_1D_vals)
        self.CPD_scatter = np.insert(self.CPD_scatter, 0, CPD_dist, axis=1)
        
        return CPD_dist, CPD_avg_dist

    def kmeans(self, data, clusters=3, show_results=False, light_pts=[]):
        
        """"
        
        Simple k-means
        
        Data typically is self.CPD_scatter
        
        light_pts : list
            Index of where to plot (i.e. when light is on). Defaults to p_on:p_off
        
        Returns
        -------
        self.results : KMeans type
        
        self.segments : dict, Nclusters
            Contains the segmented arrays for displaying
            
        """
        
        # create single [x,y] dataset
        estimators = sk.cluster.KMeans(clusters)
        self.results = estimators.fit(data)
        
        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        self.segments = {}
        
        if not any(light_pts):
            light_pts = [self.p_on, self.p_off]
        
        light_mid = int(light_pts[0] + (light_pts[1]-light_pts[0])/2)
        
        # color defaults are blue, orange, green, red, purple...
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 
                  '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', 
                  '#17becf']
        if show_results:
            
            plt.figure()
            plt.xlabel('Distance to Nearest Boundary (um)')
            plt.ylabel('CPD (V)')
            
            for i in range(clusters):
                
                if data.shape[1] > 2:
                    plt.plot(data[labels==labels_unique[i],0]*1e6,
                             np.mean(data[labels==labels_unique[i],light_pts[0]:light_pts[1]],axis=1),
                             c=colors[i], linestyle='None', marker='.')
            
                    plt.plot(cluster_centers[i][0]*1e6, cluster_centers[i][light_mid],
                             marker='o',markerfacecolor = colors[i], markersize=8, 
                             markeredgecolor='k')
                
                else:
                    plt.plot(data[labels==labels_unique[i],0]*1e6,
                                 data[labels==labels_unique[i],1],
                                 c=colors[i], linestyle='None', marker='.')
                
                    plt.plot(cluster_centers[i][0]*1e6, cluster_centers[i][1],
                             marker='o',markerfacecolor = colors[i], markersize=8, 
                             markeredgecolor='k')
                
        return self.results
    
    def segment_maps(self):
        
        """
        This creates 2D maps of the segments to overlay on an image
        
        Returns to Class
        ----------------
        segments is in actual length
        segments_idx is in index coordinates
        segments_CPD is the full CPD trace
        
        To display, make sure to do [:,1], [:,0] given row, column ordering
        Also, segments_idx is to display since pyplot uses the index on the axis
        
        """
        
        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        
        self.segments = {}
        self.segments_idx = {}
        self.segments_CPD = {}
        self.segments_CPD_avg = {}
        
        for i in range(len(labels_unique)):
            self.segments[i] = self.CPD_1D_pos[labels==labels_unique[i],:]
            self.segments_idx[i] = self.CPD_1D_idx[labels==labels_unique[i],:]
            self.segments_CPD[i] = self.CPD_1D_vals[labels==labels_unique[i],:]
            self.segments_CPD_avg[i] = self.CPD_avg_1D_vals[labels==labels_unique[i]]
        
        # the average CPD in that segment
        self.CPD_time_avg = {}
        for i in range(len(labels_unique)):
            
            self.CPD_time_avg[i] = np.mean(self.segments_CPD[i], axis=0)
        
        return
    
    def heat_map(self, bins=100):
        
        """
        Plots a heat map using CPD_avg_scatter data
        """
        
        heatmap, _, _ = np.histogram2d(self.CPD_avg_scatter[:,1],self.CPD_avg_scatter[:,0],bins)
        
        plt.figure()
        xr = [np.min(self.CPD_avg_scatter[:,0])*1e6, np.max(self.CPD_avg_scatter[:,0])*1e6]
        yr = [np.min(self.CPD_avg_scatter[:,1]), np.max(self.CPD_avg_scatter[:,1])]
        aspect = int((xr[1]-xr[0])/ (yr[1]-yr[0]))
        plt.imshow(heatmap, origin='lower', extent=[xr[0], xr[1], yr[0],yr[1]], 
                   cmap='hot', aspect=aspect)
        
        plt.xlabel('Distance to Nearest Boundary (um)')
        plt.ylabel('CPD (V)')
        
        return


    def elbow_plot(self):
        
#        Ks = range(1, 10)
#km = [KMeans(n_clusters=i) for i in Ks]
#score = [km[i].fit(my_matrix).score(my_matrix) for i in range(len(km))]
        return
    
    def animated_clusters(self, clusters=3, one_color=False):
        
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'
        
        fig, a = plt.subplots(nrows=1, figsize=(13,6))
        
        time = np.arange(0, self.pxl_time, 2*self.dtCPD) 
        idx = np.arange(1, self.CPD.shape[1], 2) # in 2-slice increments
                
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 
                  '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', 
                  '#17becf']
        
        if one_color:
            colors = ['#1f77b4','#1f77b4','#1f77b4',
                      '#1f77b4','#1f77b4','#1f77b4',
                      '#1f77b4','#1f77b4','#1f77b4']
        
        a.set_xlabel('Distance to Nearest Boundary (um)')
        a.set_ylabel('CPD (V)')
        
        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        
        ims = []
        for t in idx:
        
            data = np.zeros([self.CPD_scatter.shape[0], 3])
            data[:,0] = self.CPD_scatter[:,0]
            data[:,1:3] = self.CPD_scatter[:, t:t+2]
            
            _results  = self.kmeans(data, clusters = clusters)

            labels = _results.labels_
            cluster_centers = _results.cluster_centers_
            labels_unique = np.unique(labels)
            
            km_ims = []
            
            for i in range(len(labels_unique)):
        
                tl0, = a.plot(data[labels==labels_unique[i],0]*1e6,data[labels==labels_unique[i],1],
                       c=colors[i],linestyle='None', marker='.')
                
                tl1, = a.plot(cluster_centers[i][0]*1e6, cluster_centers[i][1],
                         marker='o',markerfacecolor = colors[i], markersize=8, 
                         markeredgecolor='k')
                
                km_ims.append([tl0, tl1])
                
            km_ims = [i for j in km_ims for i in j] # flattens
            ims.append(km_ims)
        
        ani = animation.ArtistAnimation(fig, ims, interval=120,repeat_delay=10)
        
        ani.save('kmeans_graph_.mp4')
        
        
        return

    
    def animated_image_clusters(self, clusters=3, one_color=False):
        """
        Takes an image and animates the clusters over time on the overlay
        
        As of 4/3/2018 this code is just a placeholder
        
        """
        
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'
        
        fig, a = plt.subplots(nrows=1, figsize=(13,6))
        
        time = np.arange(0, self.pxl_time, 2*self.dtCPD) 
        idx = np.arange(1, self.CPD.shape[1], 2) # in 2-slice increments
                
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 
                  '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', 
                  '#17becf']
        
        if one_color:
            colors = ['#1f77b4','#1f77b4','#1f77b4',
                      '#1f77b4','#1f77b4','#1f77b4',
                      '#1f77b4','#1f77b4','#1f77b4']
        
        a.set_xlabel('Distance to Nearest Boundary (um)')
        a.set_ylabel('CPD (V)')
        
        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        
        ims = []
        for t in idx:
        
            data = np.zeros([self.CPD_scatter.shape[0], 3])
            data[:,0] = self.CPD_scatter[:,0]
            data[:,1:3] = self.CPD_scatter[:, t:t+2]
            
            _results  = self.kmeans(data, clusters = clusters)

            labels = _results.labels_
            cluster_centers = _results.cluster_centers_
            labels_unique = np.unique(labels)
            
            km_ims = []
            
            for i in range(len(labels_unique)):
        
                tl0, = a.plot(data[labels==labels_unique[i],0]*1e6,data[labels==labels_unique[i],1],
                       c=colors[i],linestyle='None', marker='.')
                
                tl1, = a.plot(cluster_centers[i][0]*1e6, cluster_centers[i][1],
                         marker='o',markerfacecolor = colors[i], markersize=8, 
                         markeredgecolor='k')
                
                km_ims.append([tl0, tl1])
                
            km_ims = [i for j in km_ims for i in j] # flattens
            ims.append(km_ims)
        
        ani = animation.ArtistAnimation(fig, ims, interval=120,repeat_delay=10)
        
        ani.save('kmeans_graph_.mp4')
        
        
        return


