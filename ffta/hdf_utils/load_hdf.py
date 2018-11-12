"""loadHDF5.py: Includes routines for loading into HDF5 files."""

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import numpy as np
import os

import pycroscopy as px
from pycroscopy.io.write_utils import build_ind_val_dsets, Dimension

from ffta.pixel_utils import load
from ffta.hdf_utils import gl_ibw
from ffta.hdf_utils import hdf_utils

import warnings

"""
To do: 
    Reconstruct as OOP format and create class with relevant variables
    
"""

"""
Common HDF Loading functions

loadHDF5_ibw : Loads a specific ibw and FFtrEFM folder into a new H5 file
loadHDF5_folder : Takes a folder of IBW files and creates an H5 file
*createHDF5_separate_lines : Creates a folder and saves each line as a separate file
*createHDF5_file : Creates a single H5 file for one IBW file. 
createHDF5_single_dataset : Creates FF_Raw which is the raw (r*c*averages, pnts_per_signal) Dataset
create_HDF_pixel_wise_averaged : Creates FF_Avg where each pixel's signal is averaged together
hdf_commands : Creates workspace-compatible commands for common HDF variable standards

*For debugging, not in active use

Example usage:
    >>from ffta.utils import loadHDF5
    >>h5_path, parameters = loadHDF5.loadHDF5_ibw(ibw_file_path='E:/Data/FF_image_file.ibw', 
                                                  ff_file_path=r'E:\Data\FF_Folder')
    >>loadHDF5.hdf_commands(h5_path) #prints out commands available
"""

def loadHDF5_ibw(ibw_file_path='', ff_file_path='', ftype='FF', verbose=False, 
                 subfolder='/', average=False):
    """
    Loads .ibw single file an HDF5 format. Then appends the FF data to that HDF5

    Parameters
    ----------
    ibw_file_path : string, optional
        Path to signal file IBW with images to add.

    ff_file_path : string, optional
        Path to folder containing the FF-trEFM files and config file. If empty prompts dialogue
           
    ftype : str, optional
        Delineates Ginger Lab imaging file type to be imported (not case-sensitive)
        'FF' : FF-trEFM
        'SKPM' : FM-SKPM
        'ringdown' : Ringdown
        'trEFM' : normal trEFM
        
    verbose : Boolean (Optional)
        Whether or not to show  print statements for debugging
    
    subfolder : str, optional
        Specifies folder under root (/) to save data in. Default is standard pycroscopy format
        
    average : bool, optional
        Whether to automatically call the create_pixel_averaged function to average data at each pixel
    
    Returns
    -------
    h5_path: str
        The filename path to the H5 file created
    
    parm_dict: dict
        Dictionary of relevant scan parameters
        
    """
    
    if not any(ibw_file_path):
        ibw_file_path = px.io_utils.uiGetFile(caption='Select IBW Image ',
                                                file_filter='IBW Files (*.ibw)')

    if not any(ff_file_path):
        ff_file_path = px.io_utils.uiGetFile(caption='Select FF config in folder',
                                                file_filter='Config File (*.cfg)')
        ff_file_path = '/'.join(ff_file_path.split('/')[:-1])
    
    tran = gl_ibw.GLIBWTranslator()
    h5_path = tran.translate(ibw_file_path, ftype=ftype, 
                             verbose=verbose, subfolder=subfolder)
    
    hdf = px.io.HDFwriter(h5_path)
    xy_scansize = [hdf.file.attrs['FastScanSize'],hdf.file.attrs['SlowScanSize']]
    
    if verbose:
        print("### Loading file folder ###")
    h5_path, _, parm_dict = loadHDF5_folder(folder_path=ff_file_path, 
                                            xy_scansize=xy_scansize, file_name=h5_path)
    
    if average:
        if verbose:
            print('### Creating averaged set ###')
        h5_avg = create_HDF_pixel_wise_averaged(h5_path)
        
        return h5_path, parm_dict, h5_avg
    
    return h5_path, parm_dict

def loadHDF5_folder(folder_path='', xy_scansize=[0,0], file_name='FF_H5', textload = False):
    """
    Loads .ibw folder into an HDF5 format.

    Parameters
    ----------
    folder_path : string
        Path to folder you want to process
    
    xy_sscansize : 2-float array
        Width by Height in meters (e.g. [8e-6, 4e-6])
        
    file_name : str
        Desired file name, otherwise is auto-generated

    textload : bool, optional
        If you have a folder of .txt instead of .ibw (older files, some synthetic data)

    Returns
    -------
    h5_path: str
        The filename path to the H5 file created
    
    data_files: list
        List of *.ibw files in the folder to be processed
        
    parm_dict: dict
        Dictionary of relevant scan parameters

    """
    
    if any(xy_scansize) and len(xy_scansize) != 2:
        raise Exception('XY Scan Size must be either empty (in .cfg) or length-2')
    
    if not any(folder_path):
        folder_path = px.io_utils.file_dialog(caption='Select Config File in FF-trEFM folder',
                                              file_filter='Config Files (*.cfg)')
        folder_path = '/'.join(folder_path.split('/')[:-1])
    
    print(folder_path, 'folder path')    
    filelist = os.listdir(folder_path)
    
    if textload == False:
    
        data_files = [os.path.join(folder_path, name)
                      for name in filelist if name[-3:] == 'ibw']
    
    else:
        
        data_files = [os.path.join(folder_path, name)
                      for name in filelist if name[-3:] == 'txt']
    
    if not data_files:
        raise OSError('No data files found! Are these text files?')
    
    config_file = [os.path.join(folder_path, name)
                   for name in filelist if name[-3:] == 'cfg'][0]
    
    n_pixels, parm_dict = load.configuration(config_file)
    parm_dict['num_rows'] = len(data_files)
    parm_dict['num_cols'] = n_pixels
    
    # Add dimensions if not in the config file
    if 'FastScanSize' not in parm_dict.keys():
        if not any(xy_scansize):
            raise Exception('Need XY Scan Size! Save "Width" and "Height" in Config or pass xy_scansize')
            
        [width, height] = xy_scansize
        if width > 1e-3:    # if entering as microns
            width = width*1e-6
            height = height*1e-6
            
        parm_dict['FastScanSize'] = width
        parm_dict['SlowScanSize'] = height
    
    # sometimes use width/height in config files
    if 'width' in parm_dict.keys():
        parm_dict['FastScanSize'] = width
        parm_dict['SlowScanSize'] = height
    
    # Check ratio is correct
    ratio = parm_dict['FastScanSize'] / parm_dict['SlowScanSize']
    if n_pixels/len(data_files) != ratio:
        raise Exception('X-Y Dimensions do not match filelist. Add manually to config file. Check n-pixels.')
    
    folder_path = folder_path.replace('/','\\')
    if os.path.exists(file_name) == False:
        h5_path = os.path.join(folder_path, file_name) + '.h5'
    else:
        h5_path = file_name
    
    create_HDF5_single_dataset(data_files, parm_dict, h5_path)

    return h5_path, data_files, parm_dict
    
    
def create_HDF5_single_dataset(data_files, parm_dict, h5_path, verbose=False):
    """
    Generates the HDF5 file given path to files_list and parameters dictionary
    
    Creates a Datagroup FFtrEFM_Group with a single dataset in chunks
    
    Parameters
    ----------
    data_files : list
        List of the *.ibw files to be invidually scanned
        
    parm_dict : dict
        Scan parameters to be saved as attributes

    h5_path : string
        Path to H5 file on disk

    verbose : bool, optional
        Display outputs of each function or not

    Returns
    -------
    h5_path: str
        The filename path to the H5 file created
    
    """
    
    # Uses first data set to determine parameters
    line_file = load.signal(data_files[0])
    
    # e.g. if a 16000 point signal with 2000 averages and 10 pixels
    # parm_dict['pnts_per_pixel'] = 200 (# signals at each pixel)
    #           ['pnts_per_avg'] = 16000 (# pnts per signal, called an "average")
    #           ['pnts_per_line'] = 2000 (# signals in each line)

    if 'pnts_per_pixel' not in parm_dict.keys():
        parm_dict['pnts_per_avg'] = int(line_file.shape[0])
        
        try:
        # for 1 average per pixel, this will fail
            parm_dict['pnts_per_pixel'] = int(line_file.shape[1] / parm_dict['num_cols'])
            parm_dict['pnts_per_line'] = int(line_file.shape[1])
        except:
            parm_dict['pnts_per_pixel'] = 1
            parm_dict['pnts_per_line'] = 1

    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_avg = parm_dict['pnts_per_avg']
    pnts_per_pixel = parm_dict['pnts_per_pixel']

    # Prepare data for writing to HDF
    dt = 1/parm_dict['sampling_rate']
    def_vec= np.arange(0, parm_dict['total_time'], dt)
    if def_vec.shape[0] != parm_dict['pnts_per_avg']:
        def_vec = def_vec[:-1]
        warnings.warn('Time-per-point calculation error')

    # This takes a very long time but creates a giant NxM matrix of the data
    # N = number of points in each spectra, M = number of rows * columns
    #data_size = [line_file.shape[0], line_file.shape[1]*parm_dict['num_rows'] ]
    data_size = [parm_dict['pnts_per_line']*parm_dict['num_rows'], 
                 line_file.shape[0] ]

    # To do: Fix the labels/atrtibutes on the relevant data sets
    hdf = px.io.HDFwriter(h5_path)
    ff_group = px.io.VirtualGroup('FF_Group', parent='/')
    root_group = px.io.VirtualGroup('/')
    
    # Set up the position vectors for the data
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols*pnts_per_pixel)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]
    ds_pos_ind, ds_pos_val = build_ind_val_dsets(pos_desc, is_spectral=False, verbose=verbose)
    
    spec_desc = [Dimension('Time', 's',np.linspace(0, parm_dict['total_time'], pnts_per_avg))]
    ds_spec_inds, ds_spec_vals = build_ind_val_dsets(spec_desc, is_spectral=True, verbose=verbose)
    
    ds_raw = px.io.VirtualDataset('FF_Raw', data=[[0],[0]], dtype=np.float32,
                                  parent=ff_group, maxshape=data_size, 
                                  chunking=(1, parm_dict['pnts_per_line']))

    # Standard list of auxiliary datasets that get linked with the raw dataset:
    aux_ds_names = ['Position_Indices', 'Position_Values', 
                    'Spectroscopic_Indices', 'Spectroscopic_Values']

    ff_group.add_children([ds_raw, ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals])

    # Get reference for writing the data
    ff_group.attrs = parm_dict
    h5_refs = hdf.write(ff_group, print_log=verbose)
    
    pnts_per_line = parm_dict['pnts_per_line']

    h5_raw = px.hdf_utils.find_dataset(hdf.file, 'FF_Raw')[0]
    h5_raw.attrs['quantity'] = 'Deflection'
    h5_raw.attrs['units'] = 'V'

    # Cycles through the remaining files. This takes a while (~few minutes)
    for k, num in zip(data_files, np.arange(0,len(data_files))):
        
        fname = k.replace('/','\\')
        print('####',fname.split('\\')[-1],'####')
        fname = str(num).rjust(4,'0')
        
        line_file = load.signal(k).transpose()

        f = hdf.file[h5_raw.name]
        f[pnts_per_line*num:pnts_per_line*(num+1), :] = line_file[:,:]
    
    h5_main = px.hdf_utils.get_h5_obj_refs(['FF_Raw'], h5_refs)[0] 
    px.hdf_utils.link_h5_objects_as_attrs(h5_main, px.hdf_utils.get_h5_obj_refs(aux_ds_names, h5_refs))
    
    if verbose == True:
        px.hdf_utils.print_tree(hdf.file, rel_paths=True)
    
    hdf.flush()
    
    return h5_path

def create_HDF_pixel_wise_averaged(h5_file, verbose=True):
    """
    Creates a new group FF_Avg where the FF_raw file is averaged together.
    
    This is more useful as pixel-wise averages are more relevant in FF-processing
    
    This Dataset is (n_pixels*n_rows, n_pnts_per_avg)
    
    h5_file : h5py File
        H5 File to be examined. File typically set as h5_file = hdf.file
        hdf = px.ioHDF5(h5_path), h5_path = path to disk
    
    verbose : bool, optional
        Display outputs of each function or not
    
    Returns
    -------
    h5_avg : Dataset
        The new averaged Dataset
        
    """
    
    hdf = px.io.HDFwriter(h5_file)
    h5_main = px.hdf_utils.find_dataset(hdf.file, 'FF_Raw' )[0]
    
    ff_avg_group = px.io.VirtualGroup('FF_Group_Avg', parent=h5_main.parent.name)
    root_group = px.io.VirtualGroup('/')
    parm_dict = px.hdf_utils.get_attributes(h5_main.parent)

    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_avg = parm_dict['pnts_per_avg']
    pnts_per_line = parm_dict['pnts_per_line']
    dt = 1/parm_dict['sampling_rate']
    
    # Set up the position vectors for the data
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]
    ds_pos_ind, ds_pos_val = build_ind_val_dsets(pos_desc, is_spectral=False, verbose=verbose)
    
    spec_desc = [Dimension('Time', 's',np.linspace(0, parm_dict['total_time'], pnts_per_avg))]
    ds_spec_inds, ds_spec_vals = build_ind_val_dsets(spec_desc, is_spectral=True, verbose=verbose)
    
    ds_raw = px.io.VirtualDataset('FF_Avg', data=[[0],[0]], dtype=np.float32,
                                  parent=ff_avg_group, maxshape=[num_cols*num_rows, pnts_per_avg], 
                                  chunking=(1, parm_dict['pnts_per_line']))

    # Standard list of auxiliary datasets that get linked with the raw dataset:
    aux_ds_names = ['Position_Indices', 'Position_Values', 
                    'Spectroscopic_Indices', 'Spectroscopic_Values']

    ff_avg_group.add_children([ds_raw, ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals])
    
    ff_avg_group.attrs = parm_dict
    ff_avg_group.attrs['pnts_per_line'] = num_cols # to change number of pnts in a line
    ff_avg_group.attrs['pnts_per_pixel'] = 1 # to change number of pnts in a pixel
    h5_refs = hdf.write(ff_avg_group, print_log=True)

    h5_avg = px.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[0]
    h5_avg.attrs['quantity'] = 'Deflection'
    h5_avg.attrs['units'] = 'V'
    
    # Uses get_line to extract line. Averages and returns to the Dataset FF_Avg
    # We can operate on the dataset array directly, get_line is used for future_proofing if
    #  we want to add additional operation (such as create an Image class)
    for i in range(num_rows):   
        
        if verbose == True:
            print('#### Row:',i,'####')
                  
        _ll = hdf_utils.get_line(h5_main, pnts=pnts_per_line, line_num=i, array_form=False, avg=False)  
        _ll = _ll.pixel_wise_avg()
        h5_avg[i*num_cols:(i+1)*num_cols,:] = _ll[:,:]
    
    h5_r = px.hdf_utils.get_h5_obj_refs(['FF_Avg'], h5_refs)[0] 
    px.hdf_utils.link_h5_objects_as_attrs(h5_r, px.hdf_utils.get_h5_obj_refs(aux_ds_names, h5_refs))
    
    if verbose == True:
        px.hdf_utils.print_tree(hdf.file, rel_paths=True)
        h5_avg = px.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[0]

        print('H5_avg of size:', h5_avg.shape)
    
    hdf.flush()

    return h5_avg


def createHDF5_separate_lines(data_files, parm_dict, h5_path):
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
    #data_size = [line_file.shape[1]*parm_dict['num_rows'], line_file.shape[0]]
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
        hdf.file[ff_group.name+'/'+fname] = line_file.transpose()
        
    px.hdf_utils.print_tree(hdf.file)
    
    hdf.flush()
    
    return h5_path
    

def createHDF5_file(signal, parm_dict, h5_path='', ds_name='FF_Raw'):
    """
    Generates the HDF5 file given path to a specific file and a parameters dictionary

    Parameters
    ----------
    h5_path : string
        Path to desired h5 file.
    
    signal : str, ndarray
        Path to the data file to be converted or a workspace array
        
    parm_dict : dict
        Scan parameters

    Returns
    -------
    h5_path: str
        The filename path to the H5 file create
    
    """
    
    sg = signal
    
    if 'str' in str(type(signal)):
        sg = load.signal(signal)
    
    if not any(h5_path): # if not passed, auto-generate name
        fname = signal.replace('/','\\')    
        h5_path = fname[:-4] + '.h5'
    else:
        fname = h5_path
        
    hdf = px.ioHDF5(h5_path)
    px.hdf_utils.print_tree(hdf.file)
    
    ff_group = px.MicroDataGroup('FF_Group', parent='/')
    root_group = px.MicroDataGroup('/')
    
#    fname = fname.split('\\')[-1][:-4]
    sg = px.MicroDataset(ds_name, data=sg, dtype=np.float32,parent=ff_group)
    
    if 'pnts_per_pixel' not in parm_dict.keys():
        parm_dict['pnts_per_avg'] = signal.shape[1]
        parm_dict['pnts_per_pixel'] = 1
        parm_dict['pnts_per_line'] = parm_dict['num_cols']
        
    ff_group.addChildren([sg])
    ff_group.attrs = parm_dict
    
    # Get reference for writing the data
    h5_refs = hdf.writeData(ff_group, print_log=True)
    
    hdf.flush()
    


def hdf_commands(h5_path, ds='FF_Raw'):
    """
    Creates a bunch of typical workspace HDF5 variables for scripting use
    
    h5_path : str
        String path to H5PY file
    
    ds : str, optional
        The dataset to search for and set as h5_main. 
    
    This prints the valid commands to the workspace. Then just highlight and 
        copy-paste to execute
    
    h5_path : str
        Path to hdf5 file on disk
    """
    
    commands = ['from ffta.utils import hdf_utils']

    try:
        hdf = px.io.HDFwriter(h5_path)
        commands.append("hdf = px.io.HDFwriter(h5_path)")
    except:
        pass
    
    try:
        h5_file = hdf.file
        commands.append("h5_file = hdf.file")
    except:
        pass
    
    try:
        h5_main = px.hdf_utils.find_dataset(hdf.file, ds)[0]
        commands.append("h5_main = px.hdf_utils.find_dataset(hdf.file, '"+ds+"')[0]")
    except:
        pass
    
    try:
        parameters = hdf_utils.get_params(hdf.file)
        commands.append("parameters = hdf_utils.get_params(hdf.file)")
    except:
        pass
    
    try:
        h5_ll = hdf_utils.get_line(h5_path, line_num=0)
        commands.append("h5_ll = hdf_utils.get_line(h5_path, line_num=0)")
    except:
        pass
    
    try:
        h5_px = hdf_utils.get_pixel(h5_path, rc=[0,0])
        commands.append("h5_px = hdf_utils.get_pixel(h5_path, rc=[0,0])")
    except:
        pass
    
    try:
        h5_if = px.hdf_utils.find_dataset(hdf.file, 'inst_freq')[-1]
        commands.append("h5_if = px.hdf_utils.find_dataset(hdf.file, 'inst_freq')[-1]")
    except:
        pass
    
    try:
        h5_avg = px.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[-1]
        commands.append("h5_avg = px.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[-1]")
    except:
        pass
    
    try:
        h5_filt = px.hdf_utils.find_dataset(hdf.file, 'Filtered_Data')[-1]     
        commands.append("h5_filt = px.hdf_utils.find_dataset(hdf.file, 'Filtered_Data')[-1]")
    except:
        pass
    
    try:
        h5_rb = px.hdf_utils.find_dataset(hdf.file, 'Rebuilt_Data')[-1]     
        commands.append("h5_rb = px.hdf_utils.find_dataset(hdf.file, 'Rebuilt_Data')[-1]")
    except:
        pass
    
    for i in commands:
        print(i)
    
    return