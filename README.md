[![PyPI version](https://badge.fury.io/py/FFTA.svg)](https://badge.fury.io/py/FFTA)
[![Documentation Status](https://readthedocs.org/projects/ffta/badge/?version=latest)](https://ffta.readthedocs.io/en/latest/?badge=latest)


# Fast-Free Transient Analysis
A package for processing time-dependent frequency response information in atomic force microscopy data. This package is primarily geared towards electrostatic force microscopy, with the resulting output being the amplitude, phase, and instantaneous frequency as a function of time and position.

### For further information and discussion, primary point-of-contact:
```
Dr. Raj Giridharagopal, Ph.D.
Senior Research Scientist
University of Washington
Department of Chemistry
Box 351700
Seattle, Washington, USA 98195
Office: +1-206-221-2095
Lab: +1-206-221-4191
E-mail: rgiri@uw.edu
```

### Overview
FFTA extracts Instantaneous frequency from  Time-Resolved AFM data, such as Fast Free Time-Resolved Electrostatic Force Microscopy (FF-trEFM) data[1-4]. This package also returns time-to-first frequency peak (tFP) and phase from atomic force microscopy. The use case is if your experiment records AFM deflection data and you wish to reconstruct the instantaneous frequency vs time.

This includes a few types of frequency analysis:
* Hilbert Transform (primary use case) 
* Wavelet Transform (Morlet wavelet)
* Short-time Fourier Transform
* General Mode KPFM (G-KPFM)

### References
1. Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time resolution atomic force microscopy for probing nanoscale dynamics. *Nano Lett.* **12,** 893-8 (2012). [DOI: 10.1021/nl203956q](http://dx.doi.org/10.1021/nl203956q)
2. Karatay DU, Harrison JA, et al. Fast time-resolved electrostatic force microscopy: Achieving sub-cycle time resolution. *Rev Sci Inst.* **87,** 053702 (2016). [DOI: 10.1063/1.4948396](http://dx.doi.org/10.1063/1.4948396)
3. Giridharagopal R, Precht JA, Jariwala S, Collins L, et al. Time-Resolved Electrical Scanning Probe Microscopy of Layered Perovskites Reveals Spatial Variations in Photoinduced Ionic and Electronic Carrier Motion. *ACS Nano* **13,** 2812-21 (2019). [DOI: 10/1021/acsnano.8b08390](http://dx.doi.org/10.1021/acsnano.8b08390)
4. Ginger DS, Giridharagopal R, Moore DT, Rayermann GE, Reid OG. Sub-microsecond-resolution probe microscopy. US Patent [US8686358B2](https://patents.google.com/patent/US8686358)

## Getting Started 
The Jupyter notebook in this repository "FFtrEFM Basic Processing" is good at walking through the basic commands.

#### Setup the repository
Currently, there's no automated way to do this because it's not packaged. So download the source, then unzip somewhere. Then, in whichever command prompt you like:
```
python setup.py
```
or, if you wish to update locally and push changes (i.e. Ginger Lab members)
```
python setup.py develop
```
This packaged does require a few packages not installed by default in Anaconda:
* [Pycroscopy](https://pycroscopy.github.io/pyUSID/about.html) 
* [PyUSID](https://pycroscopy.github.io/pyUSID/about.html) 

#### Load the data via interactive dialog
For this package to work, you need a parameters.cfg file. There is an example in main FFTA folder
```
import ffta

h5_path, parm_dict, h5_avg = ffta.hdf_utils.load_hdf.load_wrapper()
```

(or you can explicitly load the data without the dialog)
```
ibw_file = r'E:/Data/20190107 - BAPI trEFM/FF2_128_7V.ibw'
ff_folder = r'E:/Data/20190107 - BAPI trEFM/FF02_128x64_455nm_7V_400mA'
h5_path, parm_dict, h5_avg = ffta.hdf_utils.load_hdf.load_wrapper(ibw_file_path=ibw_file, ff_file_path=ff_folder, verbose=False, average=True)
```

#### SVD filter
```
h5_svd = ffta.analysis.svd.FF_SVD(h5_avg)
```
#### Rebuild after filtering out components
```
clean_components = [0,1,2,3,4] # would only recontruct with the first 4 components
h5_rb = ffta.analysis.svd.FF_SVD_filter(h5_avg, clean_components)
```

#### The actual FF-trEFM processing.
##### Use the Pycroscopy Process class
```
data = ffta.hdf_utils.process.FFtrEFM(h5_rb)
```

##### Test the current setting (plots the data as well)
```
data.test()
```

##### If you want to change any parameters, use update_parm

Here, we will filter the amplitude, set the ROI for determining tFP to 12 us, set the trigger to be 685 us, and change the instantaneous frequency analysis method to "FFT" (a "sliding FFT" which is essentially a STFT). Then, re-run test() to see the changes.
```
data.update_parm(**{'filter_amp': True, 'roi':0.00012, 'trigger': 0.000685, 'method':'fft'})
data.test()
```

##### If everything looks good, run Compute() to process the entire dataset. Takes a few minutes to run.
```
data.compute()
```

##### Reshape to matrix form (pyUSID saves everything as an array, see their documentation), plot, then save the data
```
data.reshape()
ffta.hdf_utils.process.plot_tfp(data)
ffta.hdf_utils.process.save_CSV_from_file(data, append='_string_to_identify_data')
```

#### Lastly, it's good practice to close your data
```
h5_rb.file.close()
```

### Loading an old H5 file in Spyder or other IDE
```
h5_path  = 'path_to_H5file_on_disk.h5'
import ffta
ffta.hdf_utils.load_commands.hdf_commands(h5_path)
```
That will print out commands that you can copy-paste into the console. The commands listed will depend on what processing has occurred. 

You can also directly load the file:
```
import h5py
h5_file = h5py.File(h5_path)
```

#### Print out the tree

Assuming you copy-pasted the commands above, there will be a variable h5_file as above.
```
import pyUSID as usid
usid.hdf_utils.print_tree(h5_file, rel_paths=True)
```
There's in introductory [cookbook](https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_ten_mins_pyusid.html) to see what pyUSID format has to offer (written by me!). 
