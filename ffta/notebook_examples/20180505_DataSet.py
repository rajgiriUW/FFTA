# -*- coding: utf-8 -*-
"""
# FF-trEFM Notebook
# Rajiv Giridharagopal, University of Washington


"""
#%%
# Checks Python Version
import sys
if sys.version_info < (3, 5):
    print('''This notebook was optimized to work on Python 3.5.
    While it may also run on other Python versions,
    functionality and performance are not guaranteed
    Please consider upgrading your python version.''')

# ## Configure Notebook

'''Import necessary libraries'''
# Visualization:
import matplotlib.pyplot as plt

# General utilities:
import os
import sys

import h5py

# Finally, pycroscopy itself
import pycroscopy as px

# Define Layouts for Widgets
lbl_layout=dict(
    width='15%'
)
widget_layout=dict(
    width='15%',margin='0px 0px 5px 12px'
)
button_layout=dict(
    width='15%',margin='0px 0px 0px 5px'
)
