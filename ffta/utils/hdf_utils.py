# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:28 2018

@author: Raj
"""
import pycroscopy as px

from ffta.line import Line
from ffta.pixel import Pixel
import warnings
import numpy as np

from pycroscopy.io.write_utils import build_ind_val_dsets, Dimension

"""
Common HDF interfacing functions

_which_h5_group : Returns a group corresponding to main FFtrEFM file
get_params : Returns the common parameters list from config file
change_params : Changes specific parameters (replicates normal command line function)
get_line : Gets a specific line, returns as either array or Line class
get_pixel : Gets a specific pixel, returns as either array or Pixel class
h5_list : Gets list of files corresponding to a key, for creating unique folders/Datasets
hdf_commands : Creates workspace-compatible commands for common HDF variable standards
add_standard_sets : Adds the standard data sets needed for much processing

"""

def _which_h5_group(h5_path):
    """
    Used internally in get_ functions to indentify type of H5_path parameter.
    H5_path can be passed as string (to h5 location), or as an existing
    variable in the workspace
    
    If this is a Dataset, it will try and return the parent as that is 
        by default where all relevant attributs are
    
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
    
    # if this dataset does not have complete FFtrEFM parameters
    if 'trigger' not in parameters:
        parameters =  px.hdf_utils.get_attributes(h5_path.parent)

    # still not there? hard-select the main dataset
    if 'trigger' not in parameters:
        
        try:
            h5_file = px.ioHDF5(h5_path).file
            parameters = px.hdf_utils.get_attributes(h5_file['FF_Group'])
            
        except:
            warnings.warn('Improper parameters file. Try h5_file')
        
    
    if any(key):
        return parameters[key]
    
    if verbose:
        print(parameters)
    
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
    gp = _which_h5_group(h5_path)
    parameters =  px.hdf_utils.get_attributes(gp)
    
    if verbose:
        print('Old parameters:')
        for key in new_vals:
            print(key,':',parameters[key])
    
    for key in new_vals:
        gp.attrs[key] = new_vals[key]
        
    parameters =  px.hdf_utils.get_attributes(gp)
    
    if verbose:
        print('\nNew parameters:')
        for key in new_vals:
            print(key,':',parameters[key])

    parameters =  px.hdf_utils.get_attributes(gp)
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
    
        p = _which_h5_group(h5_path)
        parameters =  px.hdf_utils.get_attributes(p)

        d = p['FF_Raw']
        c = p.attrs['num_cols']
        pnts = p.attrs['pnts_per_line']
     
    else: # if a Dataset, extract parameters from the shape. 
        
        d = h5_path
        parameters =  px.hdf_utils.get_attributes(h5_path)
        
        if 'trigger' not in parameters:
            parameters =  px.hdf_utils.get_attributes(h5_path.parent)
        
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
        p = _which_h5_group(h5_path)
    
        d = p['FF_Raw']
        c = p.attrs['num_cols']
        pnts = int(p.attrs['pnts_per_pixel'])
        parameters =  px.hdf_utils.get_attributes(p)
        
    else:
        
        d = h5_path
        parameters =  px.hdf_utils.get_attributes(h5_path)
        
        if 'trigger' not in parameters:
            parameters =  px.hdf_utils.get_attributes(h5_path.parent)

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

def h5_list(h5_file, key):
    '''
    Returns list of names matching a key in the h5 group passed.
    This is useful for creating unique keys in datasets
    This ONLY works on the specific group and is not for searching the HDF5 file folder
    
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

def add_standard_sets(h5_path, group, fast_x=32e-6, slow_y=8e-6, 
                      parm_dict = {}, ds='FF_Raw', verbose=False):
    """
    Adds Position_Indices and Position_Value datasets to a folder within the h5_file
    
    Uses the values of fast_x and fast_y to determine the values
    
    h5_path : h5 File or str 
        Points to a path to process
    
    group : str or H5PY group 
        Location to process data to, either as str or H5PY
        
    parms_dict : dict, optional
        Parameters to be passed. By default this should be at the command line for FFtrEFM data
        
    ds : str, optional
        Dataset name to search for within this group and set as h5_main
        
    verbose : bool, optional
        Whether to write to the command line
    """
    
    hdf = px.io.HDFwriter(h5_path)
    
    if not any(parm_dict):
        parm_dict = get_params(_which_h5_group(h5_path))
    
    if 'FastScanSize' in parm_dict:
        fast_x = parm_dict['FastScanSize']
    
    if 'SlowScanSize' in parm_dict:
        slow_y = parm_dict['SlowScanSize']
    
    try:
        num_rows = parm_dict['num_rows']
        num_cols = parm_dict['num_cols']
        pnts_per_avg = parm_dict['pnts_per_avg']
        dt = 1/parm_dict['sampling_rate']
    except: # some defaults
        warnings.warn('Improper parameters specified.')
        num_rows = 64
        num_cols = 128
        pnts_per_avg = 1
        dt = 1
    
    try:
        grp = px.io.VirtualGroup(group)
    except:
        grp = px.io.VirtualGroup(group.name)
    
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]
    ds_pos_ind, ds_pos_val = build_ind_val_dsets(pos_desc, is_spectral=False, verbose=verbose)
    
    spec_desc = [Dimension('Time', 's',np.linspace(0, pnts_per_avg, dt))]
    ds_spec_inds, ds_spec_vals = build_ind_val_dsets(spec_desc, is_spectral=True, verbose=verbose)
    
    aux_ds_names = ['Position_Indices', 'Position_Values', 
                    'Spectroscopic_Indices', 'Spectroscopic_Values']
    
    grp.add_children([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals])
    
    h5_refs = hdf.write(grp, print_log=verbose)
    
    h5_main = hdf.file[grp.name]
    
    if any(ds):
        h5_main = px.hdf_utils.find_dataset(hdf.file[grp.name], ds)[0]
    
    try:
        px.hdf_utils.link_h5_objects_as_attrs(h5_main, px.hdf_utils.get_h5_obj_refs(aux_ds_names, h5_refs))
    except:
        px.hdf_utils.link_h5_objects_as_attrs(h5_main, px.hdf_utils.get_h5_obj_refs(aux_ds_names, h5_refs))

    
    return h5_main
