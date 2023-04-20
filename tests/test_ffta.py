import os
import sys

import numpy as np

sys.path.insert(0, '..')

import ffta
import pyUSID as usid

from ffta.simulation.mechanical_drive import MechanicalDrive
from ffta.simulation.utils.load import simulation_configuration
from ffta.simulation.utils import excitation

# Testing of standard process flow
class TestFFTA:
    ff_folder = 'tests/testdata'
    ff_file = 'tests/testdata/FF_H5.h5'

    def delete_old_h5(self):

        if 'testdata' in os.getcwd():
            os.chdir('..')
            os.chdir('..')
        try:
            os.remove(self.ff_file)
            os.remove(self.ff_folder + '/tfp_fixed.csv')
            os.remove(self.ff_folder + '/shift.csv')
            os.remove(self.ff_folder + '/tfp_test.csv')
            print('###### Deleted old .h5 file')
        except:
            print('###### Error with old .h5 file')

    def test_load_data_files(self):
        self.delete_old_h5()
        h5_path, data_files, parm_dict = ffta.load.load_hdf.load_folder(folder_path=self.ff_folder)
        assert (len(data_files) == 8)
        assert (len(parm_dict.items()) == 23)
        assert (type(h5_path) == str)

    def test_load_FF(self):
        self.delete_old_h5()
        h5_path, data_files, parm_dict = ffta.load.load_hdf.load_folder(folder_path=self.ff_folder)
        h5_path = h5_path.replace('\\', '/')  # travis
        h5_avg = ffta.load.load_hdf.load_FF(data_files, parm_dict, h5_path)

        assert (h5_avg.shape == (1024, 16000))
        usid.USIDataset(h5_avg)

        h5_svd = ffta.analysis.svd.test_svd(h5_avg, show_plots=False)
        h5_rb = ffta.analysis.svd.svd_filter(h5_avg, clean_components=[0, 1, 2, 3, 4])
        assert (h5_rb.shape == (1024, 16000))
        assert (h5_rb.name == '/FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data')  # in right spot

        ff = ffta.hdf_utils.process.FFtrEFM(h5_rb, override=True)
        ff.update_parm(roi=0.0007, n_taps=499, fit=True, filter_amplitude=True)
        ff.compute()
        ff.reshape()
        ffta.hdf_utils.process.save_CSV_from_file(ff)

        tfp = h5_rb.file['FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data-Fast_Free_000/tfp']
        assert (tfp.shape == (8, 128))

        shift = h5_rb.file['FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data-Fast_Free_000/shift']
        assert (shift.shape == (8, 128))

        inst_freq = h5_rb.file['FF_Group/FF_Avg-SVD_000/Rebuilt_Data_000/Rebuilt_Data-Fast_Free_000/Inst_Freq']
        assert (inst_freq.shape == (1024, 16000))

        h5_avg.file.close()

        self.delete_old_h5()

        return


# Test individual signal processing
class TestSignal:

    # Load a simulated dataset with 100 us time constant
    def load_deflection(self):
        if 'testdata' in os.getcwd():
            os.chdir('..')
            os.chdir('..')

        self.Z = np.load('tests/signaldata/Deflection.npy')

        return

    def test_pixel_default(self):
        # Checks if pixel with default (hilbert) runs
        self.load_deflection()
        pix = ffta.pixel.Pixel(self.Z, trigger=5e-4, total_time=5e-3, roi=1e-3)
        pix.analyze()

        assert (pix.tfp > 1e-6 and pix.tfp < 500e-4)

    def test_pixel_stft(self):
        # Checks if pixel with STFT runs
        self.load_deflection()
        pix = ffta.pixel.Pixel(self.Z, trigger=5e-4, total_time=5e-3, roi=1e-3, method='stft')
        pix.analyze()

        assert (pix.tfp > 1e-6 and pix.tfp < 500e-4)

    def test_pixel_wavelet(self):
        # Checks if pixel with CWT runs
        self.load_deflection()
        pix = ffta.pixel.Pixel(self.Z, trigger=5e-4, total_time=5e-3, roi=1e-3, method='wavelet')
        pix.scales = np.arange(100, 10, -5)
        pix.analyze()

        assert (pix.tfp > 1e-6 and pix.tfp < 500e-4)

    # def test_pixel_nfmd(self):
    #     self.load_deflection()
    #     pix = ffta.pixel.Pixel(self.Z, trigger=5e-4, total_time=5e-3, roi=1e-3, method='nfmd')
    #     pix.analyze()
    #
    #     assert (pix.tfp > 1e-6 and pix.tfp < 500e-4)

class TestSimulation:

    if 'testdata' in os.getcwd():
        os.chdir('..')
        os.chdir('..')
    path = r'tests/example_sim_params.cfg'

    def test_read_sim_config(self):
        can_params, force_params, sim_params = simulation_configuration(self.path)
        assert (force_params['es_force'] == 3e-09 and force_params['delta_freq'] == -277.53 
                and force_params['tau'] ==1e-05)
        return

    def test_mech_drive(self):
        can_params, force_params, sim_params = simulation_configuration(self.path)
        cant = MechanicalDrive(can_params, force_params, sim_params)
        assert (cant.tau == 1e-5)

        Z, info = cant.simulate()
        assert(len(Z.shape) > 0)

        return

