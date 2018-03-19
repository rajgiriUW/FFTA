# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:28 2018

@author: Raj
"""
import pycroscopy as px

from ffta.line import Line
from ffta.pixel import Pixel

import numpy as np

"""
Common HDF interfacing functions

_which_h5_group : Returns a group corresponding to main FFtrEFM file
get_params : Returns the common parameters list from config file
get_line : Gets a specific line, returns as either array or Line class
get_pixel : Gets a specific pixel, returns as either array or Pixel class
h5_list : Gets list of files corresponding to a key, for creating unique folders/Datasets
hdf_commands : Creates workspace-compatible commands for common HDF variable standards

"""

def _which_h5_group(h5_path):
    """
    Used internally in get_ functions to indentify type of H5_path parameter.
    H5_path can be passed as string (to h5 location), or as an existing
    variable in the workspace
    
    h5_path : str, HDF group, HDF file
    
    Returns: h5Py Group
        h5Py group corresponding to "FF_Group" typically at hdf.file['/FF_Group']
    """
    ftype = str(type(h5_path))

    # h5_path is a file path
    if 'str' in ftype:
        
        hdf = px.ioHDF5(h5_path)
        p = px.hdf_utils.findH5group(hdf.file, 'FF_Group')[0]
    
        return p

    # h5_path is an HDF Group
    if 'Group' in ftype:
        p = h5_path
    
    # h5_path is an HDF File
    elif 'File' in ftype:
        p = px.hdf_utils.findH5group(h5_path, 'FF_Group')[0]
        
    elif 'Dataset' in ftype:
        p = h5_path.parent
        
    return p

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
    
    gp = _which_h5_group(h5_path)
    
    parameters =  px.hdf_utils.get_attributes(gp)
    
    if 'trigger' not in parameters:
        parameters =  px.hdf_utils.get_attributes(h5_path.parent)
    
    if any(key):
        return parameters[key]
    
    return parameters
    
def get_line(h5_path, line_num, pnts=1, 
             array_form=False, avg=False, transpose=False):    
    """
    Gets a line of data.
    
    If h5_path is a dataset it processes based on user-defined pnts_avg
    If h5_path is a group/file/string_path then it can be a Line class or array
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
    line_num : int
        Returns specific line in the dataset
    
    pnts : int, optional
        Number of points in a line. Same as parm_dict['pnts_per_line']
    
    array_form : bool, optional
        Returns the raw array contents rather than Line class
        
    avg : bool, optional
        Averages the pixels of n_pnts_per_pixel and then creates Pixel of that
        
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
    
        p = _which_h5_group(h5_path)
        parameters =  px.hdf_utils.get_attributes(p)

        d = p['FF_Raw']
        c = p.attrs['num_cols']
        pnts = p.attrs['pnts_per_line']
     
    else: # if a Dataset, extract parameters from the shape. 
        
        d = h5_path
        c = d.shape[0]
        parameters =  px.hdf_utils.get_attributes(h5_path)
        
        if 'trigger' not in parameters:
            parameters =  px.hdf_utils.get_attributes(h5_path.parent)
    
    signal_line = d[line_num*pnts:(line_num+1)*pnts, :]
    
    if avg == True:
        signal_line = signal_line.mean(axis=0)
        
    if transpose == True:
        signal_line = signal_line.transpose()
    
    if array_form == True:
        return signal_line
    
    if avg == True and array_form == False:
        raise ValueError('Cannot use Line to return an averaged array')
    
    line_inst = Line(signal_line, parameters, c)
    
    return line_inst
    

def get_pixel(h5_path, rc, pnts = 1, 
              array_form=False, avg=False, transpose=False):    
    """
    Gets a pixel of data, returns all the averages within that pixel
    Returns a specific key if requested
    
    h5_path : str or h5py
        Can pass either an h5_path to a file or a file already in use
    
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
        p = _which_h5_group(h5_path)
    
        d = p['FF_Raw']
        c = p.attrs['num_cols']
        pnts = int(p.attrs['pnts_per_pixel'])
        parameters =  px.hdf_utils.get_attributes(p)
        
    else:
        
        d = h5_path
        c = h5_path.shape[0]
        parameters =  px.hdf_utils.get_attributes(h5_path)
        
        if 'trigger' not in parameters:
            parameters =  px.hdf_utils.get_attributes(h5_path.parent)

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

def h5_list(h5_file, key):
    '''
    Returns list of names matching a key in the h5 group passed.
    This is useful for creating unique keys in datasets
    
    e.g. this checks for -processed folder, increments the suffix
    >>    names = hdf_utils.h5_list(hdf.file['/FF_Group'], 'processed')
    >>    try:
    >>        suffix = names[-1][-4:]
    >>        suffix = str(int(suffix)+1).zfill(4)
            
    h5_file : h5py File
        hdf.file['/Measurement_000/Channel_000'] or similar
        
    key : str
        string to search for, e.g. 'processing'
    '''
    names = []
    
    for i in h5_file:
        if key in i:
            names.append(i)
            
    return names


def hdf_commands(h5_path):
    """
    Creates a bunch of typical workspace HDF5 variables for scripting use
    
    This prints the valid commands to the workspace. Then just highlight and 
        copy-paste to execute
    
    h5_path : str
        Path to hdf5 file on disk
    """
    
    commands = []

    try:
        hdf = px.ioHDF5(h5_path)
        commands.append("hdf = px.ioHDF5(h5_path)")
    except:
        pass
    
    try:
        h5_file = hdf.file
        commands.append("h5_file = hdf.file")
    except:
        pass
    
    try:
        h5_main = px.hdf_utils.getDataSet(hdf.file, 'FF_raw')[0]
        commands.append("px.hdf_utils.getDataSet(hdf.file, 'FF_raw')[0]")
    except:
        pass
    
    try:
        parameters = get_params(hdf.file)
        commands.append("parameters = hdf_utils.get_params(hdf.file)")
    except:
        pass
    try:
        h5_ll = get_line(h5_path, line_num=0)
        commands.append("h5_ll = hdf_utils.get_line(h5_path, line_num=0)")
    except:
        pass
    try:
        h5_px = get_pixel(h5_path, rc=[0,0])
        commands.append("h5_px = hdf_utils.get_pixel(h5_path, rc=[0,0])")
    except:
        pass
    try:
        h5_avg = px.hdf_utils.getDataSet(hdf.file, 'FF_Avg')[0]
        commands.append("h5_avg = px.hdf_utils.getDataSet(hdf.file, 'FF_Avg')[0]")
    except:
        pass
    
    for i in commands:
        print(i)
    
    return
