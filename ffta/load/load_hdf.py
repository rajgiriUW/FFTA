"""loadHDF5.py: Includes routines for loading into HDF5 files."""

from __future__ import division, print_function, absolute_import, unicode_literals

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2018, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import os

import h5py
import numpy as np
import pyUSID as usid
import sidpy
from pyUSID import Dimension
from sidpy.hdf.hdf_utils import get_attributes

from ffta import line
from ffta.load import get_utils
from ffta.load import gl_ibw
from ffta.pixel_utils import load

"""
Common HDF Loading functions

load_wrapper: Loads a specific ibw and FFtrEFM folder into a new H5 file
load_folder : Takes a folder of IBW files and creates an H5 file
load_FF : Creates FF_Raw which is the raw (r*c*averages, pnts_per_signal) Dataset
load_pixel_averaged_FF : Creates FF_Avg where each pixel's signal is averaged together (r*c, pnts_per_signal) Dataset
load_pixel_averaged_from_raw : Creates FF_Avg where each pixel's signal is averaged together using the Raw data file
createHDF5_file : Creates a single H5 file for one IBW file.

Example usage:
    If you want to load an IBW (image) + associated data
    >>from ffta.hdf_utils import load_hdf
    >>h5_path, parameters, h5_avg = load_hdf.load_wrapper(ibw_file_path='E:/Data/FF_image_file.ibw',
                                                          ff_file_path=r'E:\Data\FF_Folder')
    
    If you have data and just want to load (no associated image file)
    >> file_name = 'name_you_want.h5'
    >> h5_path, data_files, parm_dict = loadHDF5_folder(folder_path=ff_file_path, verbose=True, file_name=file_name)
    >> h5_avg = load_pixel_averaged_FF(data_files, parm_dict, h5_path, mirror=True)
"""


def load_wrapper(ibw_file_path='', ff_file_path='', ftype='FF', verbose=False,
                 subfolder='/', loadverbose=True, average=True, mirror=True):
    """
    Wrapper function for processing a .ibw file and associated FF data
    
    Average just uses the pixel-wise averaged data
    Raw_Avg processes the raw deflection data, then averages those together
    
    Loads .ibw single file an HDF5 format. Then appends the FF data to that HDF5

    :param ibw_file_path: Path to signal file IBW with images to add.
    :type ibw_file_path: string, optional
    
    :param ff_file_path: Path to folder containing the FF-trEFM files and config file. If empty prompts dialogue
    :type ff_file_path: string, optional
        
    :param ftype: Delineates Ginger Lab imaging file type to be imported (not case-sensitive)
        'FF' : FF-trEFM
        'SKPM' : FM-SKPM
        'ringdown' : Ringdown
        'trEFM' : normal trEFM
    :type ftype: str, optional
        
    :param verbose:
    :type verbose: bool, optional
        Whether or not to show  print statements for debugging. Passed to Pycroscopy functions
    
    :param subfolder: Specifies folder under root (/) to save data in. Default is standard pycroscopy format
    :type subfolder: string
    
    :param loadverbose: Whether to print any simple "loading Line X" statements for feedback
    :type loadverbose: bool, optional
        
    :param average: Whether to automatically call the load_pixel_averaged_FF function to average data at each pixel
    :type average: bool, optional
        
    :param mirror: Whether to reverse the data on each line read (since data are usually saved during a RETRACE scan)
    :type mirror: bool, optional
        

    :returns: tuple (h5_path, parm_dict, h5_ff)
        WHERE
        string h5_path is the filename path to the H5 file created
        dict parm_dict Dictionary of relevant scan parameters
        [type] h5_ff is...

    """

    if not any(ibw_file_path):
        ibw_file_path = sidpy.io.interface_utils.openfile_dialog(caption='Select IBW Image ',
                                                                 file_types='IBW Files (*.ibw)')

    if not any(ff_file_path):
        ff_file_path = sidpy.io.interface_utils.openfile_dialog(caption='Select FF config in folder',
                                                                file_types='Config File (*.cfg)')
        ff_file_path = '/'.join(ff_file_path.split('/')[:-1])

    # Igor image file translation into an H5 file

    if ibw_file_path:

        if verbose:
            print('Creating Topo Directory')
        tran = gl_ibw.GLIBWTranslator()
        h5_path = tran.translate(ibw_file_path, ftype=ftype,
                                 verbose=verbose, subfolder=subfolder)
        if verbose:
            print('Created file', h5_path)
    else:
        h5_path = 'FF_H5'

    # Set up FF data
    h5_path, data_files, parm_dict = load_folder(folder_path=ff_file_path,
                                                 verbose=verbose,
                                                 file_name=h5_path)

    # Processes the data
    h5_ff = load_FF(data_files, parm_dict, h5_path, average=average,
                    verbose=verbose, loadverbose=loadverbose, mirror=mirror)

    if loadverbose:
        print('*** Copy-Paste below code for IDE access to h5***')
        print('import h5py')
        print("h5_file = h5py.File(r'" + h5_path + "')")
        print("h5_avg = h5_file['FF_Group/FF_Avg']")

    return h5_path, parm_dict, h5_ff


def load_folder(folder_path='', xy_scansize=[0, 0], file_name='FF_H5',
                textload=False, verbose=False):
    """
    Sets up loading the HDF5 files. Parses the data file list and creates the .H5 file path

    :param folder_path: Path to folder you want to process
    :type folder_path: string
        
    :param xy_scansize: Width by Height in meters (e.g. [8e-6, 4e-6]), if not in parameters file
    :type xy_scansize: 2-float array
        
    :param file_name: Desired file name, otherwise is auto-generated
    :type file_name: str
        
    :param textload:  If you have a folder of .txt instead of .ibw (older files, some synthetic data)
    :type textload: bool, optional
       
    :param verbose: Whether to output the datasets being processed
    :type verbose: bool, optional
        

    :returns: tuple (h5_path, data_files, parm_dict)
        WHERE
        str h5_path is the filename path to the H5 file created
        List data_files is the list of \*.ibw files in the folder to be processed
        dict parm_dict is the dictionary of relevant scan parameters
    """

    if any(xy_scansize) and len(xy_scansize) != 2:
        raise Exception('XY Scan Size must be either empty (in .cfg) or length-2')

    if not any(folder_path):
        folder_path = sidpy.io.interface_utils.openfile_dialog(caption='Select Config File in FF-trEFM folder',
                                                               file_types='Config Files (*.cfg)')
        folder_path = '/'.join(folder_path.split('/')[:-1])

    print(folder_path, 'folder path')
    filelist = sorted(os.listdir(folder_path))

    if textload == False:

        data_files = [os.path.join(folder_path, name)
                      for name in filelist if (name[-3:] == 'ibw' and 'FFtrEFM' in name)]

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
        if width > 1e-3:  # if entering as microns
            width = width * 1e-6
            height = height * 1e-6

        parm_dict['FastScanSize'] = width
        parm_dict['SlowScanSize'] = height

    # sometimes use width/height in config files
    if 'width' in parm_dict.keys():
        parm_dict['FastScanSize'] = width
        parm_dict['SlowScanSize'] = height

    # Check ratio is correct
    ratio = np.round(parm_dict['FastScanSize'] * 1e6, 4) / np.round(parm_dict['SlowScanSize'] * 1e6, 4)
    if n_pixels / len(data_files) != ratio:
        print(ratio)
        print(parm_dict['FastScanSize'], parm_dict['SlowScanSize'],
              n_pixels / len(data_files),
              len(data_files))
        raise Exception('X-Y Dimensions do not match filelist. Add manually to config file. Check n-pixels.')

    # add associated dimension info
    #
    # e.g. if a 16000 point signal with 2000 averages and 10 pixels 
    #   (10MHz sampling of a 1.6 ms long signal=16000, 200 averages per pixel)
    # parm_dict['pnts_per_pixel'] = 200 (# signals at each pixel)
    #           ['pnts_per_avg'] = 16000 (# pnts per signal, called an "average")
    #           ['pnts_per_line'] = 2000 (# signals in each line)

    if 'pnts_per_pixel' not in parm_dict.keys():
        print('Loading first signal')
        # Uses first data set to determine parameters
        line_file = load.signal(data_files[0])
        parm_dict['pnts_per_avg'] = int(line_file.shape[0])

        try:
            # for 1 average per pixel, this will fail
            parm_dict['pnts_per_pixel'] = int(line_file.shape[1] / parm_dict['num_cols'])
            parm_dict['pnts_per_line'] = int(line_file.shape[1])
        except:
            parm_dict['pnts_per_pixel'] = 1
            parm_dict['pnts_per_line'] = 1

    folder_path = folder_path.replace('/', '\\')
    if os.path.exists(file_name) == False:
        h5_path = os.path.join(folder_path, file_name) + '.h5'
    else:
        h5_path = file_name

    return h5_path, data_files, parm_dict


def load_FF(data_files, parm_dict, h5_path, verbose=False, loadverbose=True,
            average=True, mirror=True):
    """
    Generates the HDF5 file given path to data_files and parameters dictionary

    Creates a Datagroup FFtrEFM_Group with a single dataset in chunks

    :param data_files: List of the \*.ibw files to be invidually scanned. This is generated
        by load_folder above
    :type data_files: list
        
    :param parm_dict: Scan parameters to be saved as attributes. This is generated
        by load_folder above, or you can pass this explicitly.
    :type parm_dict: dict
        
    :param h5_path:
    :type h5_path : string
        
    :param verbose: Display outputs of each function or not
    :type verbose: bool, optional
        
    :param loadverbose: Whether to print any simple "loading Line X" statements for feedback
    :type loadverbose: bool, optional
        
    :param average: Whether to average each pixel before saving to H5. This saves both time and space
    :type average: bool, optional
        
    :param mirror: Mirrors the data when saving. This parameter is to match the FFtrEFM data
        with the associate topography as FFtrEFM is acquired during a retrace while
        topo is saved during a forward trace
    :type mirror: bool, optional
        
    :returns: The filename path to the H5 file created
    :rtype: str
        
    """

    # Prepare data for writing to HDF
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_avg = parm_dict['pnts_per_avg']
    name = 'FF_Raw'

    if average:
        parm_dict['pnts_per_pixel'] = 1
        parm_dict['pnts_per_line'] = num_cols
        name = 'FF_Avg'

    pnts_per_pixel = parm_dict['pnts_per_pixel']
    pnts_per_line = parm_dict['pnts_per_line']

    dt = 1 / parm_dict['sampling_rate']
    def_vec = np.arange(0, parm_dict['total_time'], dt)
    if def_vec.shape[0] != parm_dict['pnts_per_avg']:
        def_vec = def_vec[:-1]
        # warnings.warn('Time-per-point calculation error')

    # To do: Fix the labels/attributes on the relevant data sets
    try:
        hdf = h5py.File(h5_path, 'r+')
    except:
        print('Creating HDF5 file...')
        hdf = h5py.File(h5_path, 'w')

    try:
        ff_group = hdf.create_group('FF_Group')
    except:
        print('Group already exists, creating new one')
        ff_group = usid.hdf_utils.create_indexed_group(hdf['/'], 'FF_Group')

    # Set up the position vectors for the data
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols * pnts_per_pixel)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

    spec_desc = [Dimension('Time', 's', np.linspace(0, parm_dict['total_time'], pnts_per_avg))]

    for p in parm_dict:
        ff_group.attrs[p] = parm_dict[p]
    ff_group.attrs['pnts_per_line'] = num_cols

    h5_ff = usid.hdf_utils.write_main_dataset(ff_group,  # parent HDF5 group
                                              (num_rows * num_cols * pnts_per_pixel, pnts_per_avg),
                                              # shape of Main dataset
                                              name,  # Name of main dataset
                                              'Deflection',  # Physical quantity contained in Main dataset
                                              'V',  # Units for the physical quantity
                                              pos_desc,  # Position dimensions
                                              spec_desc,  # Spectroscopic dimensions
                                              dtype=np.float32,  # data type / precision
                                              compression='gzip',
                                              main_dset_attrs=parm_dict)

    pnts_per_line = parm_dict['pnts_per_line']

    # Cycles through the remaining files. This takes a while (~few minutes)
    for k, num in zip(data_files, np.arange(0, len(data_files))):

        if loadverbose:
            fname = k.replace('/', '\\')
            print('####', fname.split('\\')[-1], '####')
            fname = str(num).rjust(4, '0')

        line_file = load.signal(k)

        if average:
            _ll = line.Line(line_file, parm_dict, n_pixels=num_cols, pycroscopy=False)
            _ll = _ll.pixel_wise_avg().T
        else:
            _ll = line_file.transpose()

        f = hdf.file[h5_ff.name]

        if mirror:
            f[pnts_per_line * num:pnts_per_line * (num + 1), :] = np.flipud(_ll[:, :])
        else:
            f[pnts_per_line * num:pnts_per_line * (num + 1), :] = _ll[:, :]

    if verbose == True:
        usid.hdf_utils.print_tree(hdf.file, rel_paths=True)

    hdf.flush()

    return h5_ff


def load_pixel_averaged_from_raw(h5_file, verbose=True, loadverbose=True):
    """
    Creates a new group FF_Avg where the FF_raw file is averaged together.

    This is more useful as pixel-wise averages are more relevant in FF-processing

    This Dataset is (n_pixels*n_rows, n_pnts_per_avg)

    :param h5_file: H5 File to be examined. File typically set as h5_file = hdf.file
        hdf = px.ioHDF5(h5_path), h5_path = path to disk
    :type h5_file: h5py File
        
    :param verbose: Display outputs of each function or not
    :type verbose: bool, optional
        
    :param loadverbose: Whether to print any simple "loading Line X" statements for feedback
    :type loadverbose: bool, optional
        
    :returns: The new averaged Dataset
    :rtype: Dataset
        
    """

    hdf = h5py.File(h5_file)
    h5_main = usid.hdf_utils.find_dataset(hdf.file, 'FF_Raw')[0]

    try:
        ff_avg_group = h5_main.parent.create_group('FF_Avg')
    except:
        ff_avg_group = usid.hdf_utils.create_indexed_group(h5_main.parent, 'FF_Avg')

    parm_dict = get_attributes(h5_main.parent)

    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_avg = parm_dict['pnts_per_avg']
    pnts_per_line = parm_dict['pnts_per_line']
    pnts_per_pixel = parm_dict['pnts_per_pixel']
    parm_dict['pnts_per_pixel'] = 1  # only 1 average per pixel now
    parm_dict['pnts_per_line'] = num_cols  # equivalent now with averaged data
    n_pix = int(pnts_per_line / pnts_per_pixel)
    dt = 1 / parm_dict['sampling_rate']

    # Set up the position vectors for the data
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

    spec_desc = [Dimension('Time', 's', np.linspace(0, parm_dict['total_time'], pnts_per_avg))]

    for p in parm_dict:
        ff_avg_group.attrs[p] = parm_dict[p]
    ff_avg_group.attrs['pnts_per_line'] = num_cols  # to change number of pnts in a line
    ff_avg_group.attrs['pnts_per_pixel'] = 1  # to change number of pnts in a pixel

    h5_avg = usid.hdf_utils.write_main_dataset(ff_avg_group,  # parent HDF5 group
                                               (num_rows * num_cols, pnts_per_avg),  # shape of Main dataset
                                               'FF_Avg',  # Name of main dataset
                                               'Deflection',  # Physical quantity contained in Main dataset
                                               'V',  # Units for the physical quantity
                                               pos_desc,  # Position dimensions
                                               spec_desc,  # Spectroscopic dimensions
                                               dtype=np.float32,  # data type / precision
                                               compression='gzip',
                                               main_dset_attrs=parm_dict)

    # Uses get_line to extract line. Averages and returns to the Dataset FF_Avg
    # We can operate on the dataset array directly, get_line is used for future_proofing if
    #  we want to add additional operation (such as create an Image class)
    for i in range(num_rows):

        if loadverbose == True:
            print('#### Row:', i, '####')

        _ll = get_utils.get_line(h5_main, pnts=pnts_per_line, line_num=i, array_form=False, avg=False)
        _ll = _ll.pixel_wise_avg()

        h5_avg[i * num_cols:(i + 1) * num_cols, :] = _ll[:, :]

    if verbose == True:
        usid.hdf_utils.print_tree(hdf.file, rel_paths=True)
        h5_avg = usid.hdf_utils.find_dataset(hdf.file, 'FF_Avg')[0]

        print('H5_avg of size:', h5_avg.shape)

    hdf.flush()

    return h5_avg
