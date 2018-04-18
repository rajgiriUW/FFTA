# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:16:05 2018

@author: Raj
"""

import os
import numpy as np
import badpixels

# Plotting imports
from matplotlib import pyplot as plt

from ffta.utils import hdf_utils
import pycroscopy as px
from pycroscopy.io.io_utils import getTimeStamp

"""
Analyzes an HDF_5 format trEFM data set and writes the result into that file
"""

def find_FF(h5_path):
    
    h5_gp = hdf_utils._which_h5_group(h5_path)
    parameters = hdf_utils.get_params(h5_gp)
    
    return h5_gp, parameters

def process(h5_file, ds = 'FF_Raw', ref='', clear_filter = False):
    """
    Processes FF_Raw dataset in the HDF5 file
    
    This then saves within the h5 file in FF_Group-processed
    
    h5_file : h5Py file or str
        Path to a specific h5 file on the disk or an hdf.file
        
    ds : str, optional
        The Dataset to search for in the file
        
    ref : str, optional
        A path to a specific dataset in the file.
        e.g. h5_file['/FF_Group/FF_Avg/FF_Avg']
        
    clear_filter : bool, optional
        For data already filtered, calls Line's clear_filter function to 
            skip FIR/windowing steps
        
    Returns
    -------
    tfp : ndarray
        time-to-first-peak image array
    shift : ndarray
        frequency shift image array
    """
#    logging.basicConfig(filename='error.log', level=logging.INFO)
    ftype = str(type(h5_file))
    
    if ('str' in ftype) or ('File' in ftype) or ('Dataset' in ftype):
        
        h5_file = px.ioHDF5(h5_file).file
    
    else:

        raise TypeError('Must be string path, e.g. E:\Test.h5')
    
    h5_gp, parameters = find_FF(h5_file)
    
    if any(ref):
        h5_gp = h5_file[ref]
        parameters = hdf_utils.get_params(h5_gp)
    
    elif ds != 'FF_Raw':
        h5_gp = px.hdf_utils.getDataSet(h5_file, ds)[0]
        parameters = hdf_utils.get_params(h5_gp)

    # Initialize file and read parameters
    num_cols = parameters['num_cols']
    num_rows = parameters['num_rows']
    pnts_per_pixel = parameters['pnts_per_pixel']
    pnts_per_avg = parameters['pnts_per_avg']

    print('Recombination: ', parameters['recombination'])
    print( 'ROI: ', parameters['roi'])

    # Initialize arrays.
    tfp = np.zeros([num_rows, num_cols])
    shift = np.zeros([num_rows, num_cols])
    inst_freq = np.zeros([num_rows*num_cols, pnts_per_avg])

    # Initialize plotting.
    plt.ion()

    fig, a = plt.subplots(nrows=2, ncols=2,figsize=(13, 6))

    tfp_ax = a[0][1]
    shift_ax = a[1][1]
    
    img_length = parameters['FastScanSize']
    img_height = parameters['SlowScanSize']
    kwargs = {'origin': 'lower',  'x_size':img_length*1e6,
          'y_size':img_height*1e6, 'num_ticks': 5, 'stdevs': 3}
    
    try:
        ht = h5_file['/height/Raw_Data'][:,0]
        ht = np.reshape(ht, [num_cols, num_rows]).transpose()
        ht_ax = a[0][0]
        ht_image, cbar = px.plot_utils.plot_map(ht_ax, np.fliplr(ht)*1e9, cmap='gray', **kwargs)
        cbar.set_label('Height (nm)', rotation=270, labelpad=16)
    except:
        pass
    
    tfp_ax.set_title('tFP Image')
    shift_ax.set_title('Shift Image')

    tfp_image, cbar_tfp = px.plot_utils.plot_map(tfp_ax, tfp * 1e6, 
                                                 cmap='inferno', show_cbar=False, **kwargs)
    shift_image, cbar_sh = px.plot_utils.plot_map(shift_ax, shift, 
                                                  cmap='inferno', show_cbar=False, **kwargs)
    text = tfp_ax.text(num_cols/2,num_rows+3, '')
    plt.show()

    # Load every file in the file list one by one.
    for i in range(num_rows):

        line_inst = hdf_utils.get_line(h5_gp, i)
        
        if clear_filter:
            line_inst.clear_filter_flags()
        
        tfp[i, :], shift[i, :], inst_freq[i*num_cols:(i+1)*num_cols,:] = line_inst.analyze()

        tfp_image, _ = px.plot_utils.plot_map(tfp_ax, tfp * 1e6, 
                                              cmap='inferno', show_cbar=False, **kwargs)
        shift_image, _ = px.plot_utils.plot_map(shift_ax, shift, 
                                                      cmap='inferno', show_cbar=False, **kwargs)

        tfp_sc = tfp[tfp.nonzero()] * 1e6
        tfp_image.set_clim(vmin=tfp_sc.min(), vmax=tfp_sc.max())

        shift_sc = shift[shift.nonzero()]
        shift_image.set_clim(vmin=shift_sc.min(), vmax=shift_sc.max())

        tfpmean = 1e6 * tfp[i, :].mean()
        tfpstd = 1e6 * tfp[i, :].std()

        string = ("Line {0:.0f}, average tFP (us) ="
                  " {1:.2f} +/- {2:.2f}".format(i + 1, tfpmean, tfpstd))
        print(string)

        text.remove()
        text = tfp_ax.text((num_cols-len(string))/2,num_rows+4, string)

        #plt.draw()
        plt.pause(0.0001)

        del line_inst  # Delete the instance to open up memory.

    tfp_image, cbar_tfp = px.plot_utils.plot_map(tfp_ax, tfp * 1e6, cmap='inferno', **kwargs)
    cbar_tfp.set_label('Time (us)', rotation=270, labelpad=16)
    shift_image, cbar_sh = px.plot_utils.plot_map(shift_ax, shift, cmap='inferno', **kwargs)
    cbar_sh.set_label('Frequency Shift (Hz)', rotation=270, labelpad=16)
    text = tfp_ax.text(num_cols/2,num_rows+3, '')
    
    plt.show()

    _,_, tfp_fixed = save_process(h5_file, h5_gp, tfp, shift, inst_freq)
    
    #save_CSV(h5_path, tfp, shift, tfp_fixed, append=ds)

    return tfp, shift, inst_freq

def save_process(h5_file, h5_gp, tfp, shift, inst_freq):

    # set correct folder path
    ftype = str(type(h5_gp))
    if 'Dataset' in ftype:
        grp = h5_gp.parent.name
    else: # if not a dataset, we know it's a group to create a folder in
        grp = h5_gp.name
        
    # Filter bad pixels
    tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
    tfp_fixed = np.array(tfp_fixed)
    
    # create correct directory structure
    names = hdf_utils.h5_list(h5_file[grp], 'processed')
    
    # create unique suffix
    suffix = 0
    try:
        suffix = names[-1][-4:]
        suffix = str(int(suffix)+1).zfill(4)
    except:
        suffix = suffix = str(int(suffix)).zfill(4)
    
    # write data
    grp_name = grp.split('/')[-1] + '-processed-' + suffix
    grp_tr = px.MicroDataGroup(grp_name, parent = h5_file[grp].name)

    tfp_px = px.MicroDataset('tfp', tfp, parent = h5_file[grp].name)
    shift_px = px.MicroDataset('shift', shift, parent = h5_file[grp].name)
    tfp_fixed_px = px.MicroDataset('tfp_fixed', tfp_fixed, parent = h5_file[grp].name)
    inst_freq = px.MicroDataset('inst_freq', inst_freq, parent = h5_file[grp].name)
    grp_tr.attrs['timestamp'] = getTimeStamp()

    grp_tr.addChildren([tfp_px])
    grp_tr.addChildren([shift_px])
    grp_tr.addChildren([tfp_fixed_px])
    grp_tr.addChildren([inst_freq])
    
    hdf = px.ioHDF5(h5_file)
    hdf.writeData(grp_tr, print_log=True)
    
    # create standard datasets
    h5_main = hdf_utils.add_standard_sets(h5_file, group=h5_gp.name)
    
    hdf.flush()

    return tfp, shift, tfp_fixed

def save_CSV_from_file(h5_file, append=''):
    
    h5_file = px.ioHDF5(h5_file).file
    tfp = px.hdf_utils.getDataSet(h5_file, 'tfp')[0].value
    tfp_fixed = px.hdf_utils.getDataSet(h5_file, 'tfp_fixed')[0].value
    shift = px.hdf_utils.getDataSet(h5_file, 'shift')[0].value
    
    path = h5_file.file.filename.replace('\\','/')
    path = '/'.join(path.split('/')[:-1])+'/'
    os.chdir(path)
    np.savetxt('tfp-'+append+'.csv', np.fliplr(tfp).T, delimiter=',')
    np.savetxt('shift-'+append+'.csv', np.fliplr(shift).T, delimiter=',')
    np.savetxt('tfp_fixed-'+append+'.csv', np.fliplr(tfp_fixed).T, delimiter=',')
    
    return


def save_CSV(h5_path, tfp, shift, tfp_fixed, append):
    
    if type(h5_path) is not str:
        h5_path = h5_path.name
    path = '/'.join(h5_path.split('/')[:-1])+'/'
    os.chdir(path)
    np.savetxt('tfp-'+append+'.csv', np.fliplr(tfp).T, delimiter=',')
    np.savetxt('shift-'+append+'.csv', np.fliplr(shift).T, delimiter=',')
    np.savetxt('tfp_fixed-'+append+'.csv', np.fliplr(tfp_fixed).T, delimiter=',')
    
    return