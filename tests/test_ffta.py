import os
import sys
import pytest
import numpy as np
import h5py

sys.path.insert(0, '..')

import ffta
import pyUSID as usid
import pytest

# Testing of standard process flow
class TestFFTA:

    self.ff_folder = 'tests/testdata'
    self.ff_file = 'tests/testdata/FF_H5.h5'
    try:
        os.remove(ff_file)

        os.remove(ff_folder + '/tfp_fixed.csv')
        os.remove(ff_folder + '/shift.csv')
        os.remove(ff_folder + '/tfp_test.csv')

    except:
        print('aaa')

    def test_load_data_files(self):
        h5_path, data_files, parm_dict = ffta.load.load_hdf.load_folder(folder_path=self.ff_folder)
        assert (len(data_files) == 8)
        assert (len(parm_dict.items()) == 22)
        assert (type(h5_path) == str)

    def test_load_FF(self):
        h5_path, data_files, parm_dict = ffta.load.load_hdf.load_folder(folder_path=self.ff_folder)
        h5_avg = ffta.load.load_hdf.load_FF(data_files, parm_dict, h5_path)

        assert(h5_avg.shape == (8192, 20000) )
        usid.USIDataset(h5_avg)

        h5_svd = ffta.analysis.svd.test_svd(h5_avg, show_plots=False)
        h5_rb = ffta.analysis.svd.svd_filter(h5_avg, clean_components = [0, 1, 2, 3, 4])
        assert(h5_rb.shape == (8192, 20000) )
        assert(h5_rb.name == '/FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data') # in right spot
        
        ff = ffta.hdf_utils.process.FFtrEFM(h5_rb, override=True)
        ff.update_parm(roi=0.0007, n_taps=499, fit=True, filter_amplitude= True)
        ff.compute()
        ff.reshape()

        tfp = h5_rb.file['FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data-Fast_Free_000/tfp']
        assert(tfp.shape == (64, 128))

        shift = h5_rb.file['FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data-Fast_Free_000/shift']
        assert(shift.shape == (64, 128))

        inst_freq = hdf['FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data-Fast_Free_001/Inst_Freq']
        assert(inst_freq.shape == (8192, 20000))

        h5_avg.file.close()

        ffta.hdf_utils.process.save_CSV_from_file(ff)