# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:16:36 2020

@author: Raj
"""

import os

import h5py
import numpy as np
import pyUSID as usid
from igor2.binarywave import load as loadibw
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sidpy import Dimension

from ffta.load import gl_ibw
from ffta.load.load_hdf import load_folder
from ffta.pixel_utils import badpixels

'''
Loads Ringdown data from raw .ibw and with the associated *.ibw Image file.

Usage:
>> h5_rd = load_ringdown.wrapper() # will prompt for folders

or

>> h5_rd = load_ringdown.wrapper(ibw_file_path='ringdown.ibw', rd_folder='ringdown_folder_path')

>> ffta.hdf_utils.load_ringdown.test_fitting(h5_rd, pixel=0, fit_time=[1.1, 5]) # tests fitting
>> h5_Q = ffta.hdf_utils.load_ringdown.reprocess_ringdown(h5_rd, fit_time=[1.1, 5]) #reprocesses the data
>> ffta.hdf_utils.load_ringdown.plot_ringdown(h5_rd.file, h5_path=h5_rd.parent.name)  # plots
>> ffta.hdf_utils.load_ringdown.save_CSV_from_file(h5_rd.file, h5_path=h5_rd.parent.name) # saves CSV

By default, this will average the ringdown data together per-pixel and mirror to match the topography 
'''


def wrapper(ibw_file_path='', rd_folder='', verbose=False, subfolder='/',
            loadverbose=True, mirror=True, average=True, AMPINVOLS=100e-9):
    """
    Wrapper function for processing a .ibw file and associated ringdown data

    Average just uses the pixel-wise averaged data
    Raw_Avg processes the raw deflection data, then averages those together

    Loads .ibw single file an HDF5 format. Then appends the FF data to that HDF5

    :param ibw_file_path: Path to signal file IBW with images to add.
    :type ibw_file_path: string, optional

    :param rd_folder: Path to folder containing the Ringdown files and config file. If empty prompts dialogue
    :type rd_folder: string, optional

    :param verbose: Whether or not to show  print statements for debugging. Passed to Pycroscopy functions
    :type verbose: bool, optional

    :param subfolder: Specifies folder under root (/) to save data in. Default is standard pycroscopy format
    :type subfolder: str, optional

    :param loadverbose: Whether to print any simple "loading Line X" statements for feedback
    :type loadverbose: bool, optional

    :param mirror:
    :type mirror: bool, optional

    :param average: Whether to automatically call the load_pixel_averaged_FF function to average data at each pixel
    :type average: bool, optional

    :param AMPINVOLS: inverted optical level sensitivity (scaling factor for amplitude).
        if not provided, it will search for one in the attributes of h5_rd or use default
    :type AMPINVOLS: float

    :returns:
    :rtype: USIDataset of ringdown

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

    if 'AMPINVOLS' not in parm_dict:
        parm_dict.update({'AMPINVOLS': AMPINVOLS})

    h5_rd = load_ringdown(data_files, parm_dict, h5_path,
                          verbose=verbose, loadverbose=loadverbose,
                          average=average, mirror=mirror)

    return h5_rd


def load_ringdown(data_files, parm_dict, h5_path,
                  verbose=False, loadverbose=True, average=True, mirror=False):
    """
    Generates the HDF5 file given path to files_list and parameters dictionary

    Creates a Datagroup FFtrEFM_Group with a single dataset in chunks

    :param data_files: List of the \*.ibw files to be invidually scanned
    :type data_files: list

    :param parm_dict: Scan parameters to be saved as attributes
    :type parm_dict: dict

    :param h5_path: Path to H5 file on disk
    :type h5_path: string

    :param verbose: Display outputs of each function or not
    :type verbose: bool, optional

    :param loadverbose: Whether to print any simple "loading Line X" statements for feedback
    :type loadverbose: bool, optional

    :param average: Whether to reverse the data on each line read (since data are usually saved during a RETRACE scan)
    :type average: bool, optional

    :param mirror: Flips the ibw signal if acquired during a retrace, so data match the topography pixel-to-pixel
    :type mirror: bool, optional

    :returns: The filename path to the H5 file created
    :rtype: str

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
    parm_dict['pnts_per_pixel'] = int(signal.shape[0] / (800 * num_cols))
    parm_dict['pnts_per_avg'] = 800  # hard-coded in our AFM software
    parm_dict['total_time'] = 16e-3  # hard-coded in our AFM software

    if 'AMPINVOLS' not in parm_dict:
        parm_dict.update({'AMPINVOLS': 100e-9})

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

    spec_desc = [Dimension('Time', 's', np.linspace(0, parm_dict['total_time'], pnts_per_avg))]

    for p in parm_dict:
        rd_group.attrs[p] = parm_dict[p]
    rd_group.attrs['pnts_per_line'] = num_cols  # to change number of pnts in a line

    h5_rd = usid.hdf_utils.write_main_dataset(rd_group,  # parent HDF5 group
                                              (num_rows * num_cols * pnts_per_pixel, pnts_per_avg),
                                              # shape of Main dataset
                                              'Ringdown',  # Name of main dataset
                                              'Amplitude',  # Physical quantity contained in Main dataset
                                              'nm',  # Units for the physical quantity
                                              pos_desc,  # Position dimensions
                                              spec_desc,  # Spectroscopic dimensions
                                              dtype=np.float32,  # data type / precision
                                              compression='gzip',
                                              main_dset_attrs=parm_dict)

    # Cycles through the remaining files. This takes a while (~few minutes)
    for k, num in zip(data_files, np.arange(0, len(data_files))):

        if loadverbose:
            fname = k.replace('/', '\\')
            print('####', fname.split('\\')[-1], '####')
            fname = str(num).rjust(4, '0')

        signal = loadibw(k)['wave']['wData']
        signal = np.reshape(signal.T, [num_cols * orig_pnts_per_pixel, pnts_per_avg])

        if average:
            pixels = np.split(signal, num_cols, axis=0)
            signal = np.vstack([np.mean(p, axis=0) for p in pixels])

        signal *= parm_dict['AMPINVOLS']

        if mirror:
            h5_rd[num_cols * pnts_per_pixel * num: num_cols * pnts_per_pixel * (num + 1), :] = np.flipud(signal[:, :])
        else:
            h5_rd[num_cols * pnts_per_pixel * num: num_cols * pnts_per_pixel * (num + 1), :] = signal[:, :]

    if verbose == True:
        usid.hdf_utils.print_tree(hdf.file, rel_paths=True)

    return h5_rd


def reprocess_ringdown(h5_rd, fit_time=[1, 5]):
    '''
    Reprocess ringdown data using an exponential fit around the timescales indicated.

    :param h5_rd: Ringdown dataset
    :type h5_rd : USIDataset

    :param fit_time: The times (in milliseconds) to fit between. This function uses a single exponential fit
    :type fit_time: list

    '''
    h5_gp = h5_rd.parent
    drive_freq = h5_rd.attrs['drive_freq']

    Q = np.zeros([h5_rd[()].shape[0]])
    A = np.zeros([h5_rd[()].shape[0]])
    tx = np.arange(0, h5_rd.attrs['total_time'], h5_rd.attrs['total_time'] / h5_rd.attrs['pnts_per_avg'])
    [start, stop] = [np.searchsorted(tx, fit_time[0] * 1e-3), np.searchsorted(tx, fit_time[1] * 1e-3)]

    for n, pxl in enumerate(h5_rd[()]):
        popt = fit_exp(tx[start:stop], pxl[start:stop] * 1e9)
        popt[0] *= 1e-9
        popt[1] *= 1e-9
        Q[n] = popt[2] * np.pi * drive_freq
        A[n] = popt[1]

    Q = np.reshape(Q, [h5_rd.attrs['num_rows'], h5_rd.attrs['num_cols']])
    A = np.reshape(A, [h5_rd.attrs['num_rows'], h5_rd.attrs['num_cols']])

    h5_Q_gp = usid.hdf_utils.create_indexed_group(h5_gp, 'Reprocess')  # creates a new group
    h5_Q = h5_Q_gp.create_dataset('Q', data=Q, dtype=np.float32)
    h5_A = h5_Q_gp.create_dataset('Amplitude', data=A, dtype=np.float32)
    h5_Q_gp.attrs['fit_times'] = [a * 1e-3 for a in fit_time]

    return h5_Q


def test_fitting(h5_rd, pixel=0, fit_time=[1, 5], plot=True):
    '''
    Tests curve fitting on a particular pixel, then plots the result

    :param h5_rd: Ringdown dataset
    :type h5_rd: USIDataset

    :param pixel: Which pixel to fit to.
    :type pixel: int

    :param fit_time: The times (in milliseconds) to fit between. This function uses a single exponential fit
    :type fit_time: list

    :param plot:
    :type plot: bool, optional

    :returns:
    :rtype:
    '''
    drive_freq = h5_rd.attrs['drive_freq']

    tx = np.arange(0, h5_rd.attrs['total_time'], h5_rd.attrs['total_time'] / h5_rd.attrs['pnts_per_avg'])
    [start, stop] = [np.searchsorted(tx, fit_time[0] * 1e-3), np.searchsorted(tx, fit_time[1] * 1e-3)]

    cut = h5_rd[()][pixel]
    popt = fit_exp(tx[start:stop], cut[start:stop] * 1e9)  # 1e9 for amplitude for better curve-fitting

    popt[0] *= 1e-9
    popt[1] *= 1e-9

    if plot:
        print('Fit params:', popt, ' and Q=', popt[2] * drive_freq * np.pi)

        fig, a = plt.subplots()
        a.plot(tx, cut, 'k')
        a.plot(tx[start:stop], exp(tx[start:stop] - tx[start], *popt), 'g--')
        a.set_xlabel('Time (s)')
        a.set_ylabel('Amplitude (nm)')

    return popt


def save_CSV_from_file(h5_file, h5_path='/', append='', mirror=False):
    """
    Saves the Q, Amp, as CSV files

    :param h5_file: Reminder you can always type: h5_svd.file or h5_avg.file for this
    :type h5_file: H5Py file

    :param h5_path: specific folder path to search for the tfp data. Usually not needed.
    :type h5_path: str, optional

    :param append: text to append to file name (e.g. RD01 or something related to the file)
    :type append: str, optional

    :param mirror: Flips the ibw signal if acquired during a retrace, so data match the topography pixel-to-pixel
    :type mirror: bool, optional
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


def exp(t, A1, y0, tau):
    '''
    Uses a single exponential for the case of no drive

    :param t:
    :type t:

    :param A1:
    :type A1:

    :param y0:
    :type y0:

    :param tau:
    :type tau:

    :returns:
    :rtype:
    '''
    return y0 + A1 * np.exp(-t / tau)


def fit_exp(t, cut):
    """
    :param t:
    :type t:

    :param cut:
    :type cut:

    :returns:
    :rtype:
    """
    # Cost function to minimize. Faster than normal scipy optimize or lmfit
    cost = lambda p: np.sum((exp(t - t[0], *p) - cut) ** 2)

    pinit = [cut.max() - cut.min(), cut.min(), 1e-4]

    bounds = [(0, 5 * (cut.max() - cut.min())), (0, cut.min()), (1e-8, 1)]
    popt = minimize(cost, pinit, method='TNC', bounds=bounds)

    return popt.x


def plot_ringdown(h5_file, h5_path='/', append='', savefig=True, stdevs=2):
    """
    Plots the relevant tfp, inst_freq, and shift values as separate image files

    :param h5_file:
    :type h5_file: h5Py File

    :param h5_path: Location of the relevant datasets to be saved/plotted. e.g. h5_rb.name
    :type h5_path: str, optional

    :param append: A string to include in the saved figure filename
    :type append: str, optional

    :param savefig: Whether or not to save the image
    :type savefig: bool, optional

    :param stdevs: Number of standard deviations to display
    :type stdevs: int, optional

    :returns: tuple (fig, a)
        WHERE
        fig is figure object
        a is axes object
    """

    # h5_rd = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Ringdown')[0]

    if 'Dataset' in str(type(h5_file[h5_path])):
        h5_path = h5_file[h5_path].parent.name

    Q = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Q')[0][()]
    A = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Amplitude')[0][()]

    Q_fixed, _ = badpixels.fix_array(Q, threshold=2)
    A_fixed, _ = badpixels.fix_array(A, threshold=2)

    parm_dict = usid.hdf_utils.get_attributes(h5_file[h5_path])

    if 'FastScanSize' not in parm_dict:
        parm_dict = usid.hdf_utils.get_attributes(h5_file[h5_path].parent)

    xs = parm_dict['FastScanSize']
    ys = parm_dict['SlowScanSize']
    asp = ys / xs
    if asp != 1:
        asp = asp * 2

    fig, a = plt.subplots(nrows=2, figsize=(8, 9))

    _, cbar_t = usid.viz.plot_utils.plot_map(a[0], Q_fixed, x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)
    _, cbar_s = usid.viz.plot_utils.plot_map(a[1], A_fixed * 1e9, x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)

    cbar_t.set_label('Q (a.u.)', rotation=270, labelpad=16)
    a[0].set_title('Q', fontsize=12)

    cbar_s.set_label('Amplitude (nm)', rotation=270, labelpad=16)
    a[1].set_title('Amplitude', fontsize=12)

    fig.tight_layout()

    if savefig:
        path = h5_file.file.filename.replace('\\', '/')
        path = '/'.join(path.split('/')[:-1]) + '/'
        os.chdir(path)
        fig.savefig('Q_Amp_' + append + '_.tif', format='tiff')

    return fig, a
