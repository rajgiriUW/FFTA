Fast Free Transient Analysis (FFTA)
===================================
**Methods for analyzing non-stationary microscopy information**


.. toctree::
   :hidden:
   :glob:
   :maxdepth: 2

   About <about.rst>
   Github <https://github.com/rajgiriUW/ffta>
   Analyzing a Signal <about.rst>
   Analyzing an Image <image.rst>
   Simulating a Cantilever <notebooks/simulating.ipynb>
   Source <source.ffta.rst>

| Rajiv Giridharagopal, Ph.D.
| University of Washington
| rgiri@uw.edu



About
-----
FFTA extracts instantaneous frequency from digitized sine wave data, with the resulting output being the amplitude, phase, and instantaneous frequency as a function of time and position. FFTA is geared towards dynamic scanning probe microscopy data, such as that acquired in Time-Resolved Electrostatic Force Microscopy data[1-4]. FFTA generates time-to-first frequency peak (tFP) and phase from atomic force microscopy. The typical use case is if your experiment records AFM deflection data and you wish to reconstruct the instantaneous frequency vs time.

This includes a few types of spectral analysis:

* Hilbert Transform (primary use case)
* Wavelet Transform (Morlet wavelet)
* Short-time Fourier Transform
* General Mode KPFM (G-KPFM)
* Non-stationary Fourier Mode Decomposition by `Daniel Shea <https://github.com/sheadan/NFMD-ExtractionInstantaneous/>`_

Also, this package includes simulation code for computing damped driven harmonic oscillator (DDHO) response to an excitation source. 

FFTA builds heavily upon the `Universal Spectroscopic and Imaging Data (USID) <https://pycroscopy.github.io/pyUSID/index.html>`_ from Oak Ridge National Lab, as well as utilities from `Spectroscopic and Imaging Data (sidPy) <https://pycroscopy.github.io/sidpy/index.html>`_. The result is a HDF5 of the data, making transfer and re-analysis straightforward.

The general method can be applied to any reasonably narrowband signal, with the primary requirement being a triggered event occurring not at the first point (time=0). For example, in our data the signals are typically 2 ms long, sampled at 10 MHz, with a trigger at 0.4 ms. More information is on the associated page (forthcoming, "Applying FFTA to your data").

References
----------
1. Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time resolution atomic force microscopy for probing nanoscale dynamics. *Nano Lett.* 12, 893-8 (2012). DOI: `10.1021/nl203956q <http://dx.doi.org/10.1021/nl203956q>`_
2. Karatay DU, Harrison JA, et al. Fast time-resolved electrostatic force microscopy: Achieving sub-cycle time resolution. *Rev Sci Inst.* 87, 053702 (2016). DOI: `10.1063/1.4948396 <http://dx.doi.org/10.1063/1.4948396>`_
3. Giridharagopal R, Precht JA, Jariwala S, Collins L, et al. Time-Resolved Electrical Scanning Probe Microscopy of Layered Perovskites Reveals Spatial Variations in Photoinduced Ionic and Electronic Carrier Motion. *ACS Nano* 13, 2812-21 (2019). DOI: `10/1021/acsnano.8b08390 <http://dx.doi.org/10.1021/acsnano.8b08390>`_
4. Ginger DS, Giridharagopal R, Moore DT, Rayermann GE, Reid OG. Sub-microsecond-resolution probe microscopy. US Patent `US8686358B2 <https://patents.google.com/patent/US8686358>`_

Installation
------------

1) Via pip

.. code:: bash

    pip install -U ffta

2) Clone and install from source to enable debugging locally. 

.. code:: bash

    python setup.py develop



Typical Workflow
----------------

The typical workflow involves:

1. Loading the data
2. Using principal component analysis to filter the data
3. Generating the instantaneous frequency/amplitude/phase and time-to-first peak (tFP)
4. (Optional) Assuming you have a calibrated tip, you can then apply a calibration curve to the data
5. Saving the tFP data to a CSV for use in your favorite image software like Igor. You can also of course save an image in Python directly. You might also consider `seaborn-image <https://seaborn-image.readthedocs.io/en/latest/>`_.
6. Close the file.

Quick processing
~~~~~~~~~~~~~~~~
This assumes all your data are saved as Igor IBW Files, and you have an appropriate .cfg file. Details on the example will be in the `Analyzing an Image <image.rst>`_ page.

.. code:: bash

   import ffta
   ff_folder = r'D:\Raj FFtrEFM continued\Old_Decent_OPV_Sets\MDMO-PPV\FF04_outputs\IBW_Files'
   h5_path, data_files, parm_dict = ffta.load.load_hdf.load_folder(ff_folder)
   h5_avg = ffta.load.load_hdf.load_FF(data_files, parm_dict, h5_path)
   h5_svd = ffta.analysis.svd.test_svd(h5_avg) # PCA filter, choose first X non-noisy components
   clean_components = [0,1,2,3,4] 
   h5_rb = ffta.analysis.svd.svd_filter(h5_avg, clean_components) # say the first 7 are okay
   data = ffta.hdf_utils.process.FFtrEFM(h5_rb)
   data.compute()

More details on each step are below.

Load the data
~~~~~~~~~~~~~
This command assumes your data are saved as Igor IBW files.

.. code:: bash

   import ffta
   h5_path, parm_dict, h5_avg = ffta.hdf_utils.load_hdf.load_wrapper()

h5_avg will the dataset that is extracted. You can think print out the contents:

.. code:: bash

   import pyUSID as usid
   usid.hdf_utils.print_tree(h5_avg.file, rel_paths=True)

PCA filter the data
~~~~~~~~~~~~~~~~~~~
.. code:: bash

   h5_svd = ffta.analysis.svd.test_svd(h5_avg)

This command will generate several images: the abundance maps, the eigenvectors, and a scree plot. It will also print out "You need X components to capture 95%" which describes how many components capture 95% of the variance. Use your best judgment for how many components are required to reconstruct the data. Say you only want the first 5 components.

.. code:: bash

   clean_components = [0,1,2,3,4] 
   h5_rb = ffta.analysis.svd.svd_filter(h5_avg, clean_components)

Generating instantaneous frequency data and tFP via the FFtrEFM Process class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Process class is from pyUSID, and allows easy parallel processing and auto-generation of USID-compatible datasets. First, we will test our processing parameters prior to computing on the entire data set.

.. code:: bash

   data = ffta.hdf_utils.process.FFtrEFM(h5_rb)
   data.test()

That "test" will generate an image. Make sure the fit in the top graph looks sensible. If your data are reversed, you can use the argument ```recombination=True``` to fix it. Here, we will filter the amplitude, set the ROI for determining tFP to 12 us, set the trigger to be 685 us, and change the instantaneous frequency analysis method to "FFT" (a "sliding FFT" which is essentially a STFT). Then, re-run test() to see the changes.

.. code:: bash

   data.update_parm(filter_amp= True, roi=0.00012, trigger=0.000685, method='fft')
   data.test()

If everything looks good, run Compute() to process the entire dataset. **This process generally takes a few minutes to run.**

.. code:: bash

   data.compute()

Once that is done, we will need to reshape our data to be a matrix form. Then, we can plot the data

.. code:: bash

   from ffta.hdf_utils.process import plot_tfp, save_CSV_from_file
   data.reshape()
   plot_tfp(data)

You then can manually save the image. You can use any matplotlib keyword arguments (e.g. ```plot_tfp(data, vmin=10, vmax=100)```)

Save the tFP and then close the file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code::bash

   save_CSV_from_file(data, append='_string_to_identify_the_data')

It is good practice in HDF5 files to explicitly close them when not in use.

.. code:: bash

   h5_avg.file.close()