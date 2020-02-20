Standards
---------
h5_path = 'string_to_h5_file'
hdf = px.io.HDFwriter(h5_path)

# this also works, if the above is finally rmeoved
import h5py
hdf = h5py.File(h5_path)

h5_file = hdf.file ; hdf File
h5_main = px.hdf_utils.getDataSet(hdf.file, 'FF_Raw')[0] the main dataset
h5_ll = hdf_utils.get_line(h5_path, line_num=5) ; gets a line (here, line 5) returns as Line class
parameters = hdf_utils.get_params(hdf.file) ; Parameters file
h5_px = hdf_utils.get_pixel(h5_path, rc=[0,0]) ; gets a pixel (here at 0,0) returns as Pixel class
h5_avg = px.hdf_utils.getDataSet(hdf.file, 'FF_Avg')[0]


File contains Group and Dataset

h5_file = hfPy File
h5_main = Dataset (see below)
h5_ll = Line (see below)
h5_px = Pixel (see below)
parameters = Dict
h5_avg = h5_main averaged pixel wise ; Dataset

h5_ll.pixel_wise_avg() is an Array of the pixel-averages (i.e. 128 pixels with 60 averages is now just 128 points long instead of 128 * 60 points long)
h5_px = hdf_utils.get_pixel

Quick example:
-------------
import pycroscopy as px
from ffta import hdf_utils
h5_path = 'E:/Data/20180216_BAPI_IDep/FF1_x10_test9.h5'
hdf = px.io.HDFwriter(h5_path)
h5_file = hdf.file
h5_main = px.hdf_utils.find_dataset(hdf.file, 'FF_Raw')[0]
parameters = px.hdf_utils.get_attributes(h5_main.parent)

h5_ll =  hdf_utils.get_line(h5_path, line_num=0)
filt_sig, freq_filts, _,_ = filtering.FFT_testfilter(h5_ll.pixel_wise_avg(), parameters, narrowband=True, noise_tolerance=1e-10)