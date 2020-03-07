#Fast-Free Time-Resolved Electrostatic Force Microscopy (FF-trEFM)
A package for processing time-dependent frequency response information in electostatic force microscopy.

This package is currently focused on the Pycroscopy branch. Using this branch requires installation of Pycroscopy and pyUSID, available at https://pycroscopy.github.io/pycroscopy/about.html and https://pycroscopy.github.io/USID/index.html, respectively. Updates will likely continue operating on data in this pycroscopy model, but this is backwards compatible with existing data.

###For further information and discussion, primary point-of-contact:
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

###Overview
Extracts Time-to-First-Peak (tFP) from digitized Fast-Free Time-Resolved Electrostatic Force Microscopy (FF-trEFM) signals [1-4].

This includes a few types of frequency analysis:
* Hilbert Transform (primary use case) 
* Wavelet Transform
* Hilbert-Huang Transform (EMD)

###References
1. Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time resolution atomic force microscopy for probing nanoscale dynamics. *Nano Lett.* 2012;12(2):893-8. [DOI: 10.1021/nl203956q](http://dx.doi.org/10.1021/nl203956q)
2. Karatay DU, Harrison JA, et al. Fast time-resolved electrostatic force microscopy: Achieving sub-cycle time resolution. Rev Sci Inst. 2016;87(5):053702. [DOI: 10.1063/1.4948396](http://dx.doi.org/10.1063/1.4948396)
3. Giridharagopal R, Precht JA, Jariwala S, Collins L, et al. Time-Resolved Electrical Scanning Probe Microscopy of Layered Perovskites Reveals Spatial Variations in Photoinduced Ionic and Electronic Carrier Motion. ACS Nano. In press. [DOI: 10/1021/acsnano.8b08390](http://dx.doi.org/10.1021/acsnano.8b08390)
4. Ginger DS, Giridharagopal R, Moore DT, Rayermann GE, Reid OG. Sub-microsecond-resolution probe microscopy. US Patent [US8686358B2](https://patents.google.com/patent/US8686358)