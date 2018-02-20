"""loadHDF5.py: Includes routines for loading into HDF5 files."""

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import configparser
import sys
from igor.binarywave import load as loadibw
from numpy.lib.npyio import loadtxt
import numpy as np
from os.path import splitext
import os

import pycroscopy as px
import h5py

from ffta.utils import load


def loadHDF5(path='', xy_scansize=[0,0], file_name='FF_H5'):
    """
    Loads .ibw folder into an HDF5 format.

    Parameters
    ----------
    path : string
        Path to signal file.
    
    xy_sscansize : 2-float array
        Width by Height in meters (e.g. [8e-6, 4e-6])
        
    file_name : str
        Desired file name, otherwise is auto-generated

    Returns
    -------
    h5_path: str
        The filename path to the H5 file created
    
    data_files: list
        List of *.ibw files in the folder to be processed
        
    parm_dict: dict
        Dictionary of relevant scan parameters

    """
    
    if any(xy_scansize ) and len(xy_scansize) != 2:
        raise Exception('XY Scan Size must be either empty (in .cfg) or length-2')
    
    if not any(path):
        folder_path = px.io_utils.uiGetFile(caption='Select Config File in FF-trEFM folder',
                                                file_filter='Config Files (*.cfg)')
        folder_path = '/'.join(folder_path.split('/')[:-1])
    
    filelist = os.listdir(folder_path)
    
    data_files = [os.path.join(folder_path, name)
                  for name in filelist if name[-3:] == 'ibw']

    config_file = [os.path.join(folder_path, name)
                   for name in filelist if name[-3:] == 'cfg'][0]
    
    n_pixels, parm_dict = load.configuration(config_file)
    parm_dict['num_rows'] = len(data_files)
    parm_dict['num_cols'] = n_pixels
    

    # Add dimensions if not in the config file
    if 'width' not in parm_dict.keys():
        if not any(xy_scansize):
            raise Exception('Need XY Scan Size! Save "Width" and "Height" in Config or pass xy_scansize')
            
        [width, height] = xy_scansize
        if width > 1e-3:    # if entering as microns
            width = width*1e-6
            height = height*1e-6
            
        parm_dict['width'] = width
        parm_dict['height'] = height
        
    # Check ratio is correct
    ratio = parm_dict['width'] / parm_dict['height']
    if n_pixels/len(data_files) != ratio:
        raise Exception('X-Y Dimensions do not match filelist')
    
    if not any(file_name):
        file_name = 'FF_H5'
        
    folder_path = folder_path.replace('/','\\')
    h5_path = os.path.join(folder_path, file_name) + '.h5'
    
    createHDF5_image(data_files, parm_dict, h5_path)
    
    return h5_path, data_files, parm_dict
    
def createHDF5_image(data_files, parm_dict, h5_path):
    """
    Generates the HDF5 file given path to files_list and parameters dictionary
    
    Creates a Datagroup FFtrEFM_Group with each line as a separate Dataset
    
    Parameters
    ----------
    h5_path : string
        Path to H5 file
    
    data_files : list
        List of the *.ibw files to be invidually scanned
        
    parm_dict : dict
        Scan parameters to be saved as attributes

    Returns
    -------
    h5_path: str
        The filename path to the H5 file created
    
    """
    
    # Uses first data set to determine parameters
    line_file = load.signal(data_files[0])
    if 'pnts_per_pixel' not in parm_dict.keys():
        parm_dict['pnts_per_avg'] = line_file.shape[0]
        parm_dict['pnts_per_pixel'] = line_file.shape[1] / parm_dict['num_cols']
        parm_dict['pnts_per_line'] = line_file.shape[1]    

    # Prepare data for writing to HDF
    dt = 1/parm_dict['sampling_rate']
    def_vec= np.arange(0, parm_dict['total_time'], dt)
    if def_vec.shape[0] != parm_dict['pnts_per_avg']:
        raise Exception('Time-per-point calculation error')

    # This takes a very long time but creates a giant NxM matrix of the data
    data_size = [line_file.shape[1]*parm_dict['num_rows'], line_file.shape[0]]

    # To do: Fix the labels/atrtibutes on the relevant data sets
    hdf = px.ioHDF5(h5_path)
    ff_group = px.MicroDataGroup('FFtrEFM_Group', parent='/')
    root_group = px.MicroDataGroup('/')
    
    ds_raw = px.MicroDataset('0000', data=line_file, dtype=np.float32,parent=ff_group)

    ff_group.addChildren([ds_raw])
    ff_group.attrs = parm_dict
    
    # Get reference for writing the data
    h5_refs = hdf.writeData(ff_group, print_log=True)
    h5_FFraw = px.io.hdf_utils.getH5DsetRefs(['0000'], h5_refs)[0]

    # Cycles through the remaining files. This takes a while (~few minutes)
    for k, num in zip(data_files[1:], np.arange(1,len(data_files))):
        
        fname = k.replace('/','\\')
        print('####',fname.split('\\')[-1],'####')
        fname = str(num).rjust(4,'0')
        
        line_file = load.signal(k)
        hdf.file[ff_group.name+'/'+fname] = line_file
        
    px.hdf_utils.print_tree(hdf.file)
    
    hdf.flush()
    
    return h5_path
    

def createHDF5_file(signal, parm_dict, h5_path=''):
    """
    Generates the HDF5 file given path to a specific file and a parameters dictionary


    Parameters
    ----------
    h5_path : string
        Path to desired h5 file.
    
    signal : str
        Path to the data file to be converted
        
    parm_dict : dict
        Scan parameters

    Returns
    -------
    h5_path: str
        The filename path to the H5 file create
    
    """
    
    sg = load.signal(signal)
    
    if not any(h5_path): # if not passed, auto-generate name
        fname = signal.replace('/','\\')    
        h5_path = fname[:-4] + '.h5'
        
    hdf = px.ioHDF5(h5_path)
    px.hdf_utils.print_tree(hdf.file)
    
    ff_group = px.MicroDataGroup('FFtrEFM_Group', parent='/')
    root_group = px.MicroDataGroup('/')
    
    fname = fname.split('\\')[-1][:-4]
    sg = px.MicroDataset(fname, data=sg, dtype=np.float32,parent=ff_group)
    
    ff_group.addChildren([sg])
    ff_group.attrs = parm_dict
    
    # Get reference for writing the data
    h5_refs = hdf.writeData(ff_group, print_log=True)
    
    hdf.flush()
    
    
def createHDF5_large(data_files, parm_dict, h5_path):
    """
    Generates the HDF5 file given path to files_list and parameters dictionary
    
    Creates a Datagroup FFtrEFM_Group with a single dataset in chunks
    
    Parameters
    ----------
    h5_path : string
        Path to H5 file
    
    data_files : list
        List of the *.ibw files to be invidually scanned
        
    parm_dict : dict
        Scan parameters to be saved as attributes

    Returns
    -------
    h5_path: str
        The filename path to the H5 file created
    
    """
    
    # Uses first data set to determine parameters
    line_file = load.signal(data_files[0])
    if 'pnts_per_pixel' not in parm_dict.keys():
        parm_dict['pnts_per_avg'] = line_file.shape[0]
        parm_dict['pnts_per_pixel'] = line_file.shape[1] / parm_dict['num_cols']
        parm_dict['pnts_per_line'] = line_file.shape[1]    

    # Prepare data for writing to HDF
    dt = 1/parm_dict['sampling_rate']
    def_vec= np.arange(0, parm_dict['total_time'], dt)
    if def_vec.shape[0] != parm_dict['pnts_per_avg']:
        raise Exception('Time-per-point calculation error')

    # This takes a very long time but creates a giant NxM matrix of the data
    data_size = [line_file.shape[0], line_file.shape[1]*parm_dict['num_rows'] ]

    # To do: Fix the labels/atrtibutes on the relevant data sets
    hdf = px.ioHDF5(h5_path)
    ff_group = px.MicroDataGroup('FFtrEFM_Group', parent='/')
    root_group = px.MicroDataGroup('/')
    
    ds_raw = px.MicroDataset('FF_raw', data=[], dtype=np.float32,
                             parent=ff_group, maxshape=data_size, 
                             chunking=(1, parm_dict['pnts_per_line']))

    # Get reference for writing the data
    ff_group.addChildren([ds_raw])
    ff_group.attrs = parm_dict
    h5_refs = hdf.writeData(ff_group, print_log=True)
    
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_line = parm_dict['pnts_per_line']

    h5_file = px.hdf_utils.getDataSet(hdf.file, 'FF_raw')[0]

    # Cycles through the remaining files. This takes a while (~few minutes)
    for k, num in zip(data_files,np.arange(0,len(data_files))):
        
        fname = k.replace('/','\\')
        print('####',fname.split('\\')[-1],'####')
        fname = str(num).rjust(4,'0')
        
        line_file = load.signal(k)

        f = hdf.file[h5_file.name]
        f[:, pnts_per_line*num:pnts_per_line*(num+1)] = line_file[:,:]
        
    px.hdf_utils.print_tree(hdf.file)
    
    hdf.flush()
    
    return h5_path