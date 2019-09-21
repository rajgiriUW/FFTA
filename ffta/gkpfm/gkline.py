import numpy as np
from scipy import optimize as spo
from scipy.optimize import fmin_tnc

import warnings

import ffta
import time
from pyUSID.io.write_utils import  Dimension

def cpd_total(ds, params, verbose=False):
    t0 = time.time()
    gk = ffta.gkpfm.gkpixel.GKPixel(ds[0,:], params)
    cpd_mat = np.zeros([ds.shape[0], gk.num_ncycles])
    cpd, _, _ = gk.analyze(fast=True)
    cpd_mat[0, :] = cpd

    for i in np.arange(1, cpd_mat.shape[0]):
        if verbose:
            if i % 100 == 0:
                print('Line ', i)
        gk = ffta.gkpfm.gkpixel.GKPixel(ds[i,:], params)
        cpd, _, _ = gk.analyze(fast=True)
        cpd_mat[i,:] = cpd

    t1 = time.time()

    print('Time:', t1-t0)

    return cpd_mat


def save_cpd(h5_main, cpd_mat):
    parm_dict = usid.hdf_utils.get_attributes(h5_main)

    # Get relevant parameters
    num_rows = parm_dict['num_rows']
    num_cols = parm_dict['num_cols']

    h5_gp = h5_main.parent
    h5_meas_group = usid.hdf_utils.create_indexed_group(h5_gp, 'CPD')

    # Create dimensions
    pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

    # ds_pos_ind, ds_pos_val = build_ind_val_matrices(pos_desc, is_spectral=False)
    spec_desc = [Dimension('Time', 's', np.linspace(0, parm_dict['total_time'], cpd_mat.shape[1]))]
    # ds_spec_inds, ds_spec_vals = build_ind_val_matrices(spec_desc, is_spectral=True)

    # Writes main dataset
    h5_cpd = usid.hdf_utils.write_main_dataset(h5_meas_group,
                                             cpd_mat,
                                             'cpd',  # Name of main dataset
                                             'Contact Potential',  # Physical quantity contained in Main dataset
                                             'V',  # Units for the physical quantity
                                             pos_desc,  # Position dimensions
                                             spec_desc,  # Spectroscopic dimensions
                                             dtype=np.float32,  # data type / precision
                                             main_dset_attrs=parm_dict)

    return h5_cpd

def cpd_single(ds, params):

    gk = ffta.gkpfm.gkpixel.GKPixel(ds, params)
    cpd, _, _ = gk.analyze(fft=True)

    return cpd