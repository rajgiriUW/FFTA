# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:16:36 2020

@author: Raj
"""


import numpy as np
import h5py
from scipy.optimize import fmin_tnc
import os

import pyUSID as usid
from pycroscopy.io.write_utils import build_ind_val_dsets, Dimension

from ffta.hdf_utils.load_hdf import load_folder
from ffta.hdf_utils import gl_ibw
from ffta.pixel_utils import badpixels

from igor.binarywave import load as loadibw

'''
Loads Ringdown data from raw .ibw and with the associated *.ibw Image file.

Usage
h5_rd = load_ringdown.wrapper() # will prompt for folders

or

h5_rd = load_ringdown.wrapper(ibw_file_path='ringdown.ibw', rd_folder='ringdown_folder_path')

By default, this will average the ringdown data together per-pixel and mirror to match the topography 
'''



def wrapper(ibw_file_path='', rd_folder='', verbose=False, subfolder='/', 
            loadverbose = True, mirror = True, average=True):
    """
    Wrapper function for processing a .ibw file and associated FF data
    
    Average just uses the pixel-wise averaged data
    Raw_Avg processes the raw deflection data, then averages those together
    
    Loads .ibw single file an HDF5 format. Then appends the FF data to that HDF5

    Parameters
    ----------
    ibw_file_path : string, optional
        Path to signal file IBW with images to add.

    rd_folder : string, optional
        Path to folder containing the Ringdown files and config file. If empty prompts dialogue

    verbose : Boolean (Optional)
        Whether or not to show  print statements for debugging. Passed to Pycroscopy functions
        
    loadverbose : Boolean (optional)
        Whether to print any simple "loading Line X" statements for feedback

    subfolder : str, optional
        Specifies folder under root (/) to save data in. Default is standard pycroscopy format

    average : bool, optional
        Whether to automatically call the load_pixel_averaged_FF function to average data at each pixel
        
    mirror : bool, optional
        Whether to reverse the data on each line read (since data are usually saved during a RETRACE scan)

    Returns
    -------
    h5_rd: USID Dataset 
        USIDataset of ringdown

    """
    if not any(ibw_file_path):
        ibw_file_path = usid.io_utils.file_dialog(caption='Select IBW Image ',
                                                file_filter='IBW Files (*.ibw)')
        
    if not any(rd_folder):
        rd_folder = usid.io_utils.file_dialog(caption='Select Ringdown config in folder',
                                                file_filter='Config File (*.cfg)')
        rd_folder = '/'.join(rd_folder.split('/')[:-1])
        
    tran = gl_ibw.GLIBWTranslator()
    h5_path = tran.translate(ibw_file_path, ftype='ringdown',
                             verbose=verbose, subfolder=subfolder)
    
    h5_path, data_files, parm_dict = load_folder(folder_path=rd_folder, 
                                                 verbose=verbose,
                                                 file_name=h5_path)
    
    h5_rd = load_ringdown(data_files, parm_dict, h5_path, 
                          verbose=verbose, loadverbose=loadverbose, 
                          average=average, mirror=mirror)
    
    return h5_rd

def load_ringdown(data_files, parm_dict, h5_path, 
                  verbose=False, loadverbose=True, average=True, mirror=False):
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

    loadverbose : Boolean (optional)
        Whether to print any simple "loading Line X" statements for feedback

    mirror : bool, optional
        Flips the ibw signal if acquired during a retrace, so data match the topography pixel-to-pixel

    Returns
    -------
    h5_path: str
        The filename path to the H5 file created

    """
    # e.g. if a 16000 point signal with 2000 averages and 10 pixels 
    #   (10MHz sampling of a 1.6 ms long signal=16000, 200 averages per pixel)
    # parm_dict['pnts_per_pixel'] = 200 (# signals at each pixel)
    #           ['pnts_per_avg'] = 16000 (# pnts per signal, called an "average")
    #           ['pnts_per_line'] = 2000 (# signals in each line)
    
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']

    # The signals are hard-coded in the AFM software as 800 points long
    # Therefore, we can calculate pnts_per_pixel etc from the first file
    signal = loadibw(data_files[0])['wave']['wData']  # Load data.
    parm_dict['pnts_per_pixel'] = int(signal.shape[0]/(800 * num_cols))
    parm_dict['pnts_per_avg'] = 800 #hard-coded in our AFM software
    parm_dict['total_time'] = 16e-3 #hard-coded in our AFM software

    pnts_per_avg = parm_dict['pnts_per_avg']
    orig_pnts_per_pixel = parm_dict['pnts_per_pixel']
    if average:
        parm_dict['pnts_per_pixel'] = 1
        parm_dict['pnts_per_line'] = num_cols
    pnts_per_pixel = parm_dict['pnts_per_pixel']
    pnts_per_line = parm_dict['pnts_per_line']
        
    hdf = h5py.File(h5_path)
    
    try:
        rd_group = hdf.file.create_group('RD_Group')
    except:
        rd_group = usid.hdf_utils.create_indexed_group(hdf.file['/'], 'RD_Group')
        
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols * pnts_per_pixel)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]
    ds_pos_ind, ds_pos_val = build_ind_val_dsets(pos_desc, is_spectral=False, verbose=verbose)

    spec_desc = [Dimension('Time', 's',np.linspace(0, parm_dict['total_time'], pnts_per_avg))]
    ds_spec_inds, ds_spec_vals = build_ind_val_dsets(spec_desc, is_spectral=True)
    
    for p in parm_dict:
        rd_group.attrs[p] = parm_dict[p]
    rd_group.attrs['pnts_per_line'] = num_cols # to change number of pnts in a line

    h5_rd = usid.hdf_utils.write_main_dataset(rd_group,  # parent HDF5 group
                                              (num_rows * num_cols * pnts_per_pixel, pnts_per_avg),  # shape of Main dataset
                                              'Ringdown',  # Name of main dataset
                                              'Amplitude',  # Physical quantity contained in Main dataset
                                              'nm',  # Units for the physical quantity
                                              pos_desc,  # Position dimensions
                                              spec_desc,  # Spectroscopic dimensions
                                              dtype=np.float32,  # data type / precision
                                              compression='gzip',
                                              main_dset_attrs=parm_dict)

    # Cycles through the remaining files. This takes a while (~few minutes)
    for k, num in zip(data_files, np.arange(0,len(data_files))):

        if loadverbose:
            fname = k.replace('/','\\')
            print('####',fname.split('\\')[-1],'####')
            fname = str(num).rjust(4,'0')

        signal = loadibw(k)['wave']['wData'] 
        signal = np.reshape(signal.T, [num_cols * orig_pnts_per_pixel, pnts_per_avg])
        
        if average:
            pixels = np.split(signal, num_cols, axis=0)
            signal = np.vstack([np.mean(p, axis=0) for p in pixels])
        
        if mirror:
            h5_rd[num_cols*pnts_per_pixel*num : num_cols*pnts_per_pixel*(num+1), :] = np.flipud(signal[:,:])
        else:
            h5_rd[num_cols*pnts_per_pixel*num : num_cols*pnts_per_pixel*(num+1), :] = signal[:,:]
        
    if verbose == True:
        usid.hdf_utils.print_tree(hdf.file, rel_paths=True)

    return h5_rd

def reprocess_ringdown(h5_rd, fit_time=[1, 5]):
    '''
    Reprocess ringdown data using an exponential fit around the timescales indicated.
    
    h5_rd : USIDataset
        Ringdown dataset
    
    fit_time : list
        The times (in milliseconds) to fit between. This function uses a single exponential fit
    '''
    h5_gp = h5_rd.parent
    drive_freq = h5_rd.attrs['drive_freq']
    
    Q = np.zeros([h5_rd[()].shape[0]])
    A = np.zeros([h5_rd[()].shape[0]])
    tx = np.arange(0, h5_rd.attrs['total_time'], h5_rd.attrs['total_time']/h5_rd.attrs['pnts_per_avg'])
    [start, stop] = [np.searchsorted(tx, fit_time[0]*1e-3), np.searchsorted(tx, fit_time[1]*1e-3)]
    for pxl, n in zip(h5_rd[()], np.arange(h5_rd[()].shape[0])):
        popt = fit_exp(tx[start:stop], pxl[start:stop])
        Q[n] = popt[3] * np.pi * drive_freq
        A[n] = popt[1]
    
    Q = np.reshape(Q, [ h5_rd.attrs['num_rows'], h5_rd.attrs['num_cols']])
    A = np.reshape(A, [ h5_rd.attrs['num_rows'], h5_rd.attrs['num_cols']])
    
    h5_Q = h5_gp.create_dataset('Q', data=Q, dtype=np.float32)
    h5_A = h5_gp.create_dataset('Amplitude', data=Q, dtype=np.float32)
    
    return h5_Q

def save_CSV_from_file(h5_file, h5_path='/', append='', mirror=False):
    """
    Saves the Q, Amp, as CSV files
    
    h5_file : H5Py file
        Reminder you can always type: h5_svd.file or h5_avg.file for this
    
    h5_path : str, optional
        specific folder path to search for the tfp data. Usually not needed.
    
    append : str, optional
        text to append to file name (e.g. RD01 or something related to the file)
    """

    Q = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Q')[-1][()]
    A = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Amplitude')[-1][()]
    Q_fixed, _ = badpixels.fix_array(Q, threshold=2)

    print(usid.hdf_utils.find_dataset(h5_file[h5_path], 'Q')[-1].parent.name)

    path = h5_file.file.filename.replace('\\', '/')
    path = '/'.join(path.split('/')[:-1]) + '/'
    os.chdir(path)

    if mirror:
        np.savetxt('Q-' + append + '.csv', np.fliplr(Q).T, delimiter=',')
        np.savetxt('Qfixed-' + append + '.csv', np.fliplr(Q_fixed).T, delimiter=',')
        np.savetxt('Amp-' + append + '.csv', np.fliplr(A).T, delimiter=',')
    else:
        np.savetxt('Q' + append + '.csv', Q.T, delimiter=',')
        np.savetxt('Qfixed-' + append + '.csv', Q_fixed.T, delimiter=',')
        np.savetxt('Amp-' + append + '.csv', A.T, delimiter=',')

    return


def exp(t, xoff, A1, y0, tau):
        '''Uses a single exponential for the case of no drive'''
        return y0 + A1 * np.exp(-(t-xoff)/tau)

def fit_exp(t, cut):
           
        # Cost function to minimize.
        cost = lambda p: np.sum((exp(t, *p) - cut) ** 2)
        
        pinit = [cut.min(), t[0], cut.min(), 1e-4]
        
        popt, n_eval, rcode = fmin_tnc(cost, pinit, approx_grad=True, disp=0,
                                       bounds=[(t[0], t[0]),
                                               (1e-5, 1000),
                                               (-100, 100),
                                               (1e-5, 0.1)])
        
        return popt