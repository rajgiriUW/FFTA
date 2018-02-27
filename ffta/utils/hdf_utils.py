# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:28 2018

@author: Raj
"""
import pycroscopy as px

from ffta.line import Line
from ffta.pixel import Pixel
import numpy as np

def get_params(h5_path, key='', verbose=False):
    """
    Gets dict of parameters from the FF-file
    Returns a specific key-value if requested
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
    key : str, optional
        Returns specific key in the parameters dictionary
        
    verbose : bool
        Prints all parameters to console
    """
    
    # path to HDF5
    if type(h5_path)== str:
        hdf = px.ioHDF5(h5_path)
        p = px.hdf_utils.findH5group(hdf.file, 'FF')[0]
    
    # h5_path is an HDF Group
    elif 'Group' in str(type(h5_path)):
        p = h5_path
    
    # h5_path is an HDF File
    else:
        p = px.hdf_utils.findH5group(h5_path, 'FF')[0]
    
    parm_dict = {}
    
    for k in p.attrs:
        parm_dict[k] = p.attrs[k]
        
    if verbose == True:
        print(parm_dict)
        
    if any(key):
        return parm_dict[key]
    
    return parm_dict


def get_line(h5_path, line_num, array_form=False):    
    """
    Gets a line of data 
    Returns a specific key if requested
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
    line_num : int
        Returns specific line in the dataset
        
    array_form : bool
        Returns the raw array contents rather than Line class
    """
    
    if type(h5_path)== str:
        hdf = px.ioHDF5(h5_path)
        p = px.hdf_utils.findH5group(hdf.file, 'FF')[0]
    else:
        p = px.hdf_utils.findH5group(h5_path, 'FF')[0]

    d = p['FF_raw']
    c = p.attrs['num_cols']
    pnts = p.attrs['pnts_per_line']
    
    if array_form == True:
        return d[:, line_num*pnts:(line_num+1)*pnts ]
    
    signal_array = d[:, line_num*pnts:(line_num+1)*pnts ]
    
    parameters = get_params(p)
    
    line_inst = Line(signal_array, parameters, c)
    
    return line_inst
    

def get_pixel(h5_path, rc, array_form=False, avg=False):    
    """
    Gets a pixel of data, returns all the averages within that pixel
    Returns a specific key if requested
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
    rc : list [r, c]
        Pixel location in terms of ROW, COLUMN
        
    array_form : bool
        Returns the raw array contents rather than Pixel class
    
    avg : bool
        Averages the pixels of n_pnts_per_pixel and then creates Pixel of that
    """
    
    if type(h5_path)== str:
        hdf = px.ioHDF5(h5_path)
        p = px.hdf_utils.findH5group(hdf.file, 'FF')[0]
    else:
        p = px.hdf_utils.findH5group(h5_path, 'FF')[0]
        
    d = p['FF_raw']
    c = p.attrs['num_cols']
    pnts = int(p.attrs['pnts_per_pixel'])
    
    if array_form == True:
        return d[:, rc[0]*c + rc[1]:rc[0]*c + rc[1]+pnts]
    
    signal_pixel = d[:, rc[0]*c + rc[1]:rc[0]*c + rc[1]+pnts]
    parameters = get_params(p)
    
    if avg == True:
        signal_pixel = signal_pixel.mean(axis=1)
        
    pixel_inst = Pixel(signal_pixel, parameters)
    
    return pixel_inst    

def h5_list(h5file, key):
    '''
    Returns list of names matching a key in the h5 group passed
    
    h5file : h5py File
        hdf.file['/Measurement_000/Channel_000'] or similar
        
    key : str
        string to search for, e.g. 'processing'
    '''
    names = []
    for i in h5file:
        if key in i:
            names.append(i)
            
    return names