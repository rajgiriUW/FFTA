[![PyPI version](https://badge.fury.io/py/FFTA.svg)](https://badge.fury.io/py/FFTA)
[![Documentation Status](https://readthedocs.org/projects/ffta/badge/?version=latest)](https://ffta.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/rajgiriUW/ffta/badge.svg?branch=master)](https://coveralls.io/github/rajgiriUW/ffta?branch=master)


# Fast-Free Transient Analysis
A package for processing time-dependent frequency response information in atomic force microscopy data. This package is primarily geared towards electrostatic force microscopy, with the resulting output being the amplitude, phase, and instantaneous frequency as a function of time and position. This includes non-stationary Fourier mode decomposition by Daniel Shea of University of Washington.

### For further information and discussion, primary point-of-contact:
```
Dr. Raj Giridharagopal, Ph.D.
Senior Research Scientist
University of Washington
Department of Chemistry
Seattle, Washington, USA
E-mail: rgiri@uw.edu
```

### Overview
FFTA extracts Instantaneous frequency from  Time-Resolved AFM data, such as Fast Free Time-Resolved Electrostatic Force Microscopy (FF-trEFM) data[1-4]. This package also returns time-to-first frequency peak (tFP) and phase from atomic force microscopy. The use case is if your experiment records AFM deflection data and you wish to reconstruct the instantaneous frequency vs time.

This includes a few types of frequency analysis:
* Hilbert Transform  
* Wavelet Transform (CWT, via Morlet wavelet)
* Short-time Fourier Transform (STFT)
* General Mode KPFM (G-KPFM)
* Nonstationary Fourier Mode Decomposition (NFMD)

### References
1. Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time resolution atomic force microscopy for probing nanoscale dynamics. *Nano Lett.* **12,** 893-8 (2012). [DOI: 10.1021/nl203956q](http://dx.doi.org/10.1021/nl203956q)
2. Karatay DU, Harrison JA, et al. Fast time-resolved electrostatic force microscopy: Achieving sub-cycle time resolution. *Rev Sci Inst.* **87,** 053702 (2016). [DOI: 10.1063/1.4948396](http://dx.doi.org/10.1063/1.4948396)
3. Giridharagopal R, Precht JA, Jariwala S, Collins L, et al. Time-Resolved Electrical Scanning Probe Microscopy of Layered Perovskites Reveals Spatial Variations in Photoinduced Ionic and Electronic Carrier Motion. *ACS Nano* **13,** 2812-21 (2019). [DOI: 10/1021/acsnano.8b08390](http://dx.doi.org/10.1021/acsnano.8b08390)
4. Ginger DS, Giridharagopal R, Moore DT, Rayermann GE, Reid OG. Sub-microsecond-resolution probe microscopy. US Patent [US8686358B2](https://patents.google.com/patent/US8686358)
5. Shea DE, Giridharagopal R, Ginger DS, Brunton SL, Kutz JN. Extraction of instantaneous frequencies and amplitudes in nonstationary time-series data. *IEEE Access.* In press (as of 6/14/2021). [DOI: 10.1109/ACCESS.2021.3087595](https://ieeexplore.ieee.org/document/9448199)

### Getting started
To get started and to see some examples, check out the (very much in progress) [documention](https://ffta.readthedocs.io/en/latest/).

### Installation

```
pip install ffta
```


