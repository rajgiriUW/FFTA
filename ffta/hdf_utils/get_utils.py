# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 13:40:54 2018

@author: Raj
"""

import pycroscopy as px

from ffta.line import Line
from ffta.pixel import Pixel

import numpy as np

'''
Functions for extracting certain segments from an HDF FFtrEFM file
'''

def get_params(h5_path, key='', verbose=False, del_indices=True):
    """
    Gets dict of parameters from the FF-file
    Returns a specific key-value if requested
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
    key : str, optional
        Returns specific key in the parameters dictionary
        
    verbose : bool, optional
        Prints all parameters to console
        
    del_indices : bool, optional
        Deletes relative links within the H5Py and any quantity/units
    """
    
    if isinstance(h5_path, str):
        h5_path = px.io.HDFwriter(h5_path).file
    
    parameters =  px.hdf_utils.get_attributes(h5_path)
    
    # if this dataset does not have complete FFtrEFM parameters
    if 'trigger' not in parameters:
        parameters =  px.hdf_utils.get_attributes(h5_path.parent)

    # still not there? hard-select the main dataset
    if 'trigger' not in parameters:
        
        try:
            h5_file = px.io.HDFwriter(h5_path).file
            parameters = px.hdf_utils.get_attributes(h5_file['FF_Group'])
            
        except:
            raise TypeError('No proper parameters file found.')
    
    if any(key):
        return parameters[key]
    
    if verbose:
        print(parameters)
    
    del_keys = ['Position_Indices', 'Position_Values','Spectroscopic_Indices',
                'Spectroscopic_Values', 'quantity', 'units']
    
    for key in del_keys:
        if key in parameters:
            del parameters[key]
            
    return parameters

def change_params(h5_path, new_vals = {}, verbose=False):
    """
    Changes a parameter to a new value
    
    This is equivalent to h5_main.parent.attrs[key] = new_value
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
    key : dict, optional
        Returns specific keys in the parameters dictionary
    
    value : str, int, float
        The new value for the key. There is no error checking for this
    
    verbose : bool
        Prints all parameters to console
    
    """
    parameters =  px.hdf_utils.get_attributes(h5_path)
    
    if verbose:
        print('Old parameters:')
        for key in new_vals:
            print(key,':',parameters[key])
    
    for key in new_vals:
        h5_path.attrs[key] = new_vals[key]
        
    if verbose:
        print('\nNew parameters:')
        for key in new_vals:
            print(key,':',parameters[key])

    return parameters
    
def get_line(h5_path, line_num, pnts=1, 
             array_form=False, avg=False, transpose=False):    
    """
    Gets a line of data.
    
    If h5_path is a dataset it processes based on user-defined pnts_avg
    If h5_path is a group/file/string_path then it can be a Line class or array
    
    h5_path : str or h5py or Dataset
        Can pass either an h5_path to a file or a file already in use or a specific Dataset
    
    line_num : int
        Returns specific line in the dataset
    
    pnts : int, optional
        Number of points in a line. Same as parm_dict['pnts_per_line']
    
    array_form : bool, optional
        Returns the raw array contents rather than Line class
        
    avg : bool, optional
        Averages the pixels of the entire line 
        This will force output to be an array
        
    transpose : bool, optional
        For legacy FFtrEFM code, pixel is required in (n_points, n_signals) format
        
    Returns
    -------
    signal_line : numpy 2D array, iff array_form == True
        ndarray containing the line 
        
    line_inst : Line, iff array_form == False
        Line class containing the signal_array object and parameters
    """
    
    # If not a dataset, then find the associated Group
    if 'Dataset' not in str(type(h5_path)):
    
        parameters =  get_params(h5_path)
        h5_file = px.io.HDFwriter(h5_path).file

        d = px.hdf_utils.find_dataset(h5_file, 'FF_Raw')[0]
        c = parameters['num_cols']
        pnts = parameters['pnts_per_line']
     
    else: # if a Dataset, extract parameters from the shape. 
        
        d = h5_path
        parameters =  get_params(h5_path)
        
        c = parameters['num_cols']
        pnts = parameters['pnts_per_line']
    
    signal_line = d[line_num*pnts:(line_num+1)*pnts, :]
    
    if avg == True:
        signal_line = (signal_line.transpose() - signal_line.mean(axis=1)).transpose()
        signal_line = signal_line.mean(axis=0)
        
    if transpose == True:
        signal_line = signal_line.transpose()
    
    if array_form == True or avg == True:
        return signal_line
    
    line_inst = Line(signal_line, parameters, c)
    
    return line_inst
    

def get_pixel(h5_path, rc, pnts = 1, 
              array_form=False, avg=False, transpose=False):    
    """
    Gets a pixel of data, returns all the averages within that pixel
    Returns a specific key if requested
    Supplying a direct link to a Dataset is MUCH faster than just the file
    
    h5_path : str or h5py or Dataset
        Can pass either an h5_path to a file or a file already in use or specific Dataset
    
    rc : list [r, c]
        Pixel location in terms of ROW, COLUMN
        
    pnts : int
        Number of signals to average together. By default is extracted from parm_dict/attributes    
    
    array_form : bool, optional
        Returns the raw array contents rather than Pixel class
    
    avg : bool, optional
        Averages the pixels of n_pnts_per_pixel and then creates Pixel of that
        
    transpose : bool, optional
        For legacy FFtrEFM code, pixel is required in (n_points, n_signals) format
        
    Returns
    -------
    signal_pixel : 1D numpy array, iff array_form == True
        Ndarray of the (n_points) in specific pixel    
    
    pixel_inst : Pixel, iff array_form == False
        Line class containing the signal_array object and parameters
    """
    
    # If not a dataset, then find the associated Group
    if 'Dataset' not in str(type(h5_path)):
        p = get_params(h5_path)
        h5_file = px.io.HDFwriter(h5_path).file
    
        d = px.hdf_utils.find_dataset(h5_file, 'FF_Raw')[0]
        c = p['num_cols']
        pnts = int(p['pnts_per_pixel'])
        parameters =  px.hdf_utils.get_attributes(d.parent)
        
    else:
        
        d = h5_path
        parameters =  get_params(h5_path)

        c = parameters['num_cols']
        pnts = parameters['pnts_per_pixel']

    signal_pixel = d[rc[0]*c + rc[1]:rc[0]*c + rc[1]+pnts, :]    

    if avg == True:
        signal_pixel = signal_pixel.mean(axis=0)

    if transpose == True:   # this does nothing is avg==True
        signal_pixel = signal_pixel.transpose()
        
    if array_form == True:
        return signal_pixel
    
    if signal_pixel.shape[0] == 1:
        
        signal_pixel = np.reshape(signal_pixel, [signal_pixel.shape[1]])
        
    pixel_inst = Pixel(signal_pixel, parameters)
    
    return pixel_inst    