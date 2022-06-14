# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:16:05 2018

@author: Raj
"""

import os

import h5py
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt
from pyUSID.io import Dimension

from ffta.hdf_utils import hdf_utils
from ffta.load import get_utils
from ffta.pixel_utils import badpixels

"""
Analyzes an HDF_5 format trEFM data set and writes the result into that file
"""


def process(h5_file, ds='FF_Raw', ref='', clear_filter=False,
            verbose=True, liveplots=True, **kwargs):
    """
    Processes FF_Raw dataset in the HDF5 file

    This then saves within the h5 file in FF_Group-processed

    This uses the dataset in this priority:
        \*A relative path specific by ref, e.g. '/FF_Group/FF_Avg/FF_Avg'
        \*A dataset specified by ds, returning the last found, e.g. 'FF_Raw'
        \*FF_Group/FF_Raw found via searching from hdf_utils folder

    Typical usage:
    >> import pycroscopy as px
    >> h5_file = px.io.HDFwriter('path_to_h5_file.h5').file
    >> from ffta import analyze_h5
    >> tfp, shift, inst_freq = analyze_h5.process(h5_file, ref = '/FF_Group/FF_Avg/FF_Avg')

    :param h5_file: Path to a specific h5 file on the disk or an hdf.file
    :type h5_file: h5Py file or str

    :param ds: The Dataset to search for in the file
    :type ds: str, optional

    :param ref: A path to a specific dataset in the file.
        e.g. h5_file['/FF_Group/FF_Avg/FF_Avg']
    :typr ref: str, optional

    :param clear_filter: For data already filtered, calls Line's clear_filter function to
            skip FIR/windowing steps
    :type clear_filter: bool, optional

    :param verbose: Whether to write data to the command line
    :type verbose: bool, optional,

    :param liveplots: Displaying can sometimes cause the window to pop in front of other active windows
        in Matplotlib. This disables it, with an obvious drawback of no feedback.
    :type liveplots : bool

    :returns: tuple (tfp, shift, inst_freq, h5_if)
        WHERE
        ndarray tfp is time-to-first-peak image array
        ndarray shift is frequency shift image array
        2D ndarray inst_freq is instantaneous frequency array, an N x p array of N=rows\*cols points
            and where p = points_per_signal (e.g. 16000 for 1.6 ms @10 MHz sampling)
        USIDataset h5_if is instantaneous frequency

    """
    #    logging.basicConfig(filename='error.log', level=logging.INFO)
    ftype = str(type(h5_file))

    if ('str' in ftype) or ('File' in ftype) or ('Dataset' in ftype):

        h5_file = h5py.File(h5_file)

    else:

        raise TypeError('Must be string path, e.g. E:\Test.h5')

    # Looks for a ref first before searching for ds, h5_ds is group to process
    if any(ref):
        h5_ds = h5_file[ref]
        parameters = get_utils.get_params(h5_ds)

    elif ds != 'FF_Raw':
        h5_ds = usid.hdf_utils.find_dataset(h5_file, ds)[-1]
        parameters = get_utils.get_params(h5_ds)

    else:
        h5_ds, parameters = find_FF(h5_file)

    if isinstance(h5_ds, h5py.Dataset):
        h5_gp = h5_ds.parent

    # Initialize file and read parameters
    num_cols = parameters['num_cols']
    num_rows = parameters['num_rows']
    pnts_per_pixel = parameters['pnts_per_pixel']
    pnts_per_avg = parameters['pnts_per_avg']

    for key, value in kwargs.items():
        _temp = parameters[key]
        parameters[key] = value
        if verbose:
            print('Changing', key, 'from', _temp, 'to', value)

    if verbose:
        print('Recombination: ', parameters['recombination'])
        print('ROI: ', parameters['roi'])

    # Initialize arrays.
    tfp = np.zeros([num_rows, num_cols])
    shift = np.zeros([num_rows, num_cols])
    inst_freq = np.zeros([num_rows * num_cols, pnts_per_avg])

    # Initialize plotting.

    plt.ion()
    fig, a = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))

    tfp_ax = a[0][1]
    shift_ax = a[1][1]

    img_length = parameters['FastScanSize']
    img_height = parameters['SlowScanSize']
    kwargs = {'origin': 'lower', 'x_vec': img_length * 1e6,
              'y_vec': img_height * 1e6, 'num_ticks': 5, 'stdevs': 3}

    try:
        ht = h5_file['/height/Raw_Data'][:, 0]
        ht = np.reshape(ht, [num_cols, num_rows]).transpose()
        ht_ax = a[0][0]
        ht_image, cbar = usid.viz.plot_utils.plot_map(ht_ax, ht * 1e9, cmap='gray', **kwargs)
        cbar.set_label('Height (nm)', rotation=270, labelpad=16)
    except:
        pass

    tfp_ax.set_title('tFP Image')
    shift_ax.set_title('Shift Image')

    tfp_image, cbar_tfp = usid.viz.plot_utils.plot_map(tfp_ax, tfp * 1e6,
                                                       cmap='inferno', show_cbar=False, **kwargs)
    shift_image, cbar_sh = usid.viz.plot_utils.plot_map(shift_ax, shift,
                                                        cmap='inferno', show_cbar=False, **kwargs)
    text = tfp_ax.text(num_cols / 2, num_rows + 3, '')
    plt.show()

    print('Analyzing with roi of', parameters['roi'])

    # Load every file in the file list one by one.
    for i in range(num_rows):

        line_inst = get_utils.get_line(h5_ds, i, params=parameters)

        if clear_filter:
            line_inst.clear_filter_flags()

        _tfp, _shf, _if = line_inst.analyze()
        tfp[i, :] = _tfp.T
        shift[i, :] = _shf.T
        inst_freq[i * num_cols:(i + 1) * num_cols, :] = _if.T

        if liveplots:
            tfp_image, _ = usid.viz.plot_utils.plot_map(tfp_ax, tfp * 1e6,
                                                        cmap='inferno', show_cbar=False, **kwargs)
            shift_image, _ = usid.viz.plot_utils.plot_map(shift_ax, shift,
                                                          cmap='inferno', show_cbar=False, **kwargs)

            tfp_sc = tfp[tfp.nonzero()] * 1e6
            tfp_image.set_clim(vmin=tfp_sc.min(), vmax=tfp_sc.max())

            shift_sc = shift[shift.nonzero()]
            shift_image.set_clim(vmin=shift_sc.min(), vmax=shift_sc.max())

            tfpmean = 1e6 * tfp[i, :].mean()
            tfpstd = 1e6 * tfp[i, :].std()

            if verbose:
                string = ("Line {0:.0f}, average tFP (us) ="
                          " {1:.2f} +/- {2:.2f}".format(i + 1, tfpmean, tfpstd))
                print(string)

                text.remove()
                text = tfp_ax.text((num_cols - len(string)) / 2, num_rows + 4, string)

            # plt.draw()
            plt.pause(0.0001)

        del line_inst  # Delete the instance to open up memory.

    tfp_image, cbar_tfp = usid.viz.plot_utils.plot_map(tfp_ax, tfp * 1e6, cmap='inferno', **kwargs)
    cbar_tfp.set_label('Time (us)', rotation=270, labelpad=16)
    shift_image, cbar_sh = usid.viz.plot_utils.plot_map(shift_ax, shift, cmap='inferno', **kwargs)
    cbar_sh.set_label('Frequency Shift (Hz)', rotation=270, labelpad=16)
    text = tfp_ax.text(num_cols / 2, num_rows + 3, '')

    plt.show()

    h5_if = save_IF(h5_ds.parent, inst_freq, parameters)
    _, _, tfp_fixed = save_ht_outs(h5_if.parent, tfp, shift)

    # save_CSV(h5_path, tfp, shift, tfp_fixed, append=ds)

    if verbose:
        print('Please remember to close the H5 file explicitly when you are done to retain these data',
              'e.g.:',
              'h5_if.file.close()',
              '...and then reopen the file as needed.')

    return tfp, shift, inst_freq, h5_if


def save_IF(h5_gp, inst_freq, parm_dict):
    """
    Adds Instantaneous Frequency as a main dataset

    :param h5_gp:
    :type h5_gp:

    :param inst_freq:
    :type inst_freq:

    :param parm_dict:
    :type parm_dict: dict

    :returns:
    :rtype:
    """
    # Error check
    if isinstance(h5_gp, h5py.Dataset):
        raise ValueError('Must pass an h5Py group')

    # Get relevant parameters
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']
    pnts_per_avg = parm_dict['pnts_per_avg']

    h5_meas_group = usid.hdf_utils.create_indexed_group(h5_gp, 'processed')

    # Create dimensions
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

    # ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
    spec_desc = [Dimension('Time', 's', np.linspace(0, parm_dict['total_time'], pnts_per_avg))]
    # ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)

    # Writes main dataset
    h5_if = usid.hdf_utils.write_main_dataset(h5_meas_group,
                                              inst_freq,
                                              'inst_freq',  # Name of main dataset
                                              'Frequency',  # Physical quantity contained in Main dataset
                                              'Hz',  # Units for the physical quantity
                                              pos_desc,  # Position dimensions
                                              spec_desc,  # Spectroscopic dimensions
                                              dtype=np.float32,  # data type / precision
                                              main_dset_attrs=parm_dict)

    usid.hdf_utils.copy_attributes(h5_if, h5_gp)

    h5_if.file.flush()

    return h5_if


def save_ht_outs(h5_gp, tfp, shift):
    """
    Save processed Hilbert Transform outputs

    :param h5_gp:
    :type h5_gp:

    :param tfp:
    :type tfp:

    :param shift:
    :type shift:

    :returns: tuple (tfp_px, shift_px, tfp_fixed_px)
        WHERE
        [type] tfp_px is...
        [type] shift_px is...
        [type] tfp_fixed_px is...
    """
    # Error check
    if isinstance(h5_gp, h5py.Dataset):
        raise ValueError('Must pass an h5Py group')

    # Filter bad pixels
    tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
    tfp_fixed = np.array(tfp_fixed)

    #    grp_tr = h5_file.create_group(h5_gp.name)
    tfp_px = h5_gp.create_dataset('tfp', data=tfp, dtype=np.float32)
    shift_px = h5_gp.create_dataset('shift', data=shift, dtype=np.float32)
    tfp_fixed_px = h5_gp.create_dataset('tfp_fixed', data=tfp_fixed, dtype=np.float32)
    # 	h5_gp.attrs['timestamp'] = write_book_keeping_attrs()

    return tfp_px, shift_px, tfp_fixed_px


def save_CSV_from_file(h5_file, h5_path='/', append='', mirror=False):
    """
    Saves the tfp, shift, and fixed_tfp as CSV files

    :param h5_file: Reminder you can always type: h5_svd.file or h5_avg.file for this
    :type h5_file: H5Py file

    :param h5_path: specific folder path to search for the tfp data. Usually not needed.
    :type h5_path: str, optional

    :param append: text to append to file name
    :type append: str, optional

    :param mirror:
    :type mirror: bool, optional

    """

    tfp = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp')[0][()]
    tfp_fixed = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp_fixed')[0][()]
    shift = usid.hdf_utils.find_dataset(h5_file[h5_path], 'shift')[0][()]

    print(usid.hdf_utils.find_dataset(h5_file[h5_path], 'shift')[0].parent.name)

    path = h5_file.file.filename.replace('\\', '/')
    path = '/'.join(path.split('/')[:-1]) + '/'
    os.chdir(path)

    if mirror:
        np.savetxt('tfp-' + append + '.csv', np.fliplr(tfp).T, delimiter=',')
        np.savetxt('shift-' + append + '.csv', np.fliplr(shift).T, delimiter=',')
        np.savetxt('tfp_fixed-' + append + '.csv', np.fliplr(tfp_fixed).T, delimiter=',')
    else:
        np.savetxt('tfp-' + append + '.csv', tfp.T, delimiter=',')
        np.savetxt('shift-' + append + '.csv', shift.T, delimiter=',')
        np.savetxt('tfp_fixed-' + append + '.csv', tfp_fixed.T, delimiter=',')
    return


def plot_tfps(h5_file, h5_path='/', append='', savefig=True, stdevs=2):
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

    """

    h5_file = h5py.File(h5_file)

    try:
        h5_if = usid.hdf_utils.find_dataset(h5_file[h5_path], 'Inst_Freq')[0]
    except:
        h5_if = usid.hdf_utils.find_dataset(h5_file[h5_path], 'inst_freq')[0]
    parm_dict = get_utils.get_params(h5_if)
    #  parm_dict = usid.hdf_utils.get_attributes(h5_file[h5_path])

    if 'Dataset' in str(type(h5_file[h5_path])):
        h5_path = h5_file[h5_path].parent.name

    tfp = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp')[0][()]
    shift = usid.hdf_utils.find_dataset(h5_file[h5_path], 'shift')[0][()]

    if tfp.shape[1] == 1:
        # Forgot to reconvert and/or needs to be converted
        [ysz, xsz] = h5_if.pos_dim_sizes
        tfp = np.reshape(tfp, [ysz, xsz])
        shift = np.reshape(shift, [ysz, xsz])

    try:
        tfp_fixed = usid.hdf_utils.find_dataset(h5_file[h5_path], 'tfp_fixed')[0][()]
    except:
        tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
        tfp_fixed = np.array(tfp_fixed)

    xs = parm_dict['FastScanSize']
    ys = parm_dict['SlowScanSize']
    asp = ys / xs
    if asp != 1:
        asp = asp * 2

    fig, a = plt.subplots(nrows=3, figsize=(8, 9))

    [vmint, vmaxt] = np.mean(tfp) - 2 * np.std(tfp), np.mean(tfp) - 2 * np.std(tfp)
    [vmins, vmaxs] = np.mean(shift) - 2 * np.std(shift), np.mean(shift) - 2 * np.std(shift)

    _, cbar_t = usid.viz.plot_utils.plot_map(a[0], tfp_fixed * 1e6, x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)
    _, cbar_r = usid.viz.plot_utils.plot_map(a[1], 1 / (1e3 * tfp_fixed), x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)
    _, cbar_s = usid.viz.plot_utils.plot_map(a[2], shift, x_vec=xs * 1e6, y_vec=ys * 1e6,
                                             aspect=asp, cmap='inferno', stdevs=stdevs)

    cbar_t.set_label('tfp (us)', rotation=270, labelpad=16)
    a[0].set_title('tfp', fontsize=12)

    cbar_r.set_label('Rate (kHz)', rotation=270, labelpad=16)
    a[1].set_title('1/tfp', fontsize=12)

    cbar_s.set_label('shift (Hz)', rotation=270, labelpad=16)
    a[2].set_title('shift', fontsize=12)

    fig.tight_layout()

    if savefig:
        path = h5_file.file.filename.replace('\\', '/')
        path = '/'.join(path.split('/')[:-1]) + '/'
        os.chdir(path)
        fig.savefig('tfp_shift_' + append + '_.tif', format='tiff')

    return


def find_FF(h5_path):
    """

    :param h5_path: Location of the relevant datasets to be saved/plotted. e.g. h5_rb.name
    :type h5_path: str

    :returns: tuple (h5_gp, parameters)
        WHERE
        [type] h5_gp is...
        [type] parameters is...
    """
    parameters = get_utils.get_params(h5_path)
    h5_gp = hdf_utils._which_h5_group(h5_path)

    return h5_gp, parameters
