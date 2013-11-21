#!/usr/bin/env python

"""load.py: Includes routines for loading data and configuration files."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2013, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Production"


import ibw
import ConfigParser


def igor(fname):
    """
    Opens a .ibw file and returns it as a numpy.ndarray.

    Parameters
    ----------
    fname : file or strplot()

            File, filename, or generator to read.

    Returns
    -------
    signal : array_like

            Real signals.

    """

    data = ibw.loadibw(fname)  # Load data.
    signal = data[0]  # Crop off the data and only keep the numeric array.
    signal.flags.writeable = True

    return signal


def configuration(filename):
    """
    Reads a .cfg file with relevant parameters for processing.

    Parameters
    ----------
    fname : file or str

            File, filename, or generator to read.

    Returns
    -------
    parameters : dict

            Parameters for processing.

    """

    # Create a parser for configuration file and parses it.
    config = ConfigParser.RawConfigParser()
    config.read(filename)
    parameters = {}
    parameters['Processing'] = {}

    # Assign parameters from file. These are the keys for parameters.
    param_keys = ['Trigger', 'Time', 'SampleRate', 'DriveFreq', 'NPix']
    procf_keys = ['Bandwidth', 'SmoothTime', 'WindowSize', 'NRLevel']
    procb_keys = ['BPF', 'Window', 'Smooth']

    for key in param_keys:
        if config.has_option('Parameters', key):
            parameters[key] = config.getfloat('Parameters',
                                              key)

    for key in procf_keys:
        if config.has_option('Processing', key):
            parameters['Processing'][key] = config.getfloat('Processing',
                                                            key)

    for key in procb_keys:
        if config.has_option('Processing', key):
            parameters['Processing'][key] = config.getboolean('Processing',
                                                              key)

    if not ('NPix' in parameters):

        ans = raw_input('Number of pixels is not given. '
                        'Enter number of pixels:')

        parameters['NPix'] = int(ans)

    if not ('NRLevel' in parameters['Processing']):

        parameters['Processing']['NRLevel'] = 0

    if not ('Bandwidth' in parameters['Processing']):

        parameters['Processing']['Bandwidth'] = 10e3

    return parameters
