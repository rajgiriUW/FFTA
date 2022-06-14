# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:28:28 2018

@author: Raj
"""
import h5py
import pyUSID as usid

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
    variable in the workspace.

    This tries

    If this is a Dataset, it will try and return the parent as that is
        by default where all relevant attributs are

    :param h5_path:
    :type h5_path: str, HDF group, HDF file

    :returns: h5Py group corresponding to "FF_Group" typically at hdf.file['/FF_Group']
    :rtype: h5Py Group
        """
    ftype = str(type(h5_path))

    # h5_path is a file path
    if 'str' in ftype:
        hdf = h5py.File(h5_path)
        p = usid.hdf_utils.find_dataset(hdf.file, 'FF_Raw')[0]

        return p.parent

    # h5_path is an HDF Group
    if 'Group' in ftype:
        p = h5_path

    # h5_path is an HDF File
    elif 'File' in ftype:
        p = usid.hdf_utils.find_dataset(h5_path, 'FF_Raw')[0]

        p = p.parent

    elif 'Dataset' in ftype:
        p = h5_path.parent

    return p


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

    :param h5_file: hdf.file['/Measurement_000/Channel_000'] or similar
    :type h5_file: h5py File
        
    :param key: string to search for, e.g. 'processing'
    :type key: str
    
    :returns:
    :rtype: List of str
    '''
    names = []

    for i in h5_file:
        if key in i:
            names.append(i)

    return names
