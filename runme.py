#!/usr/bin/env python
# encoding: utf-8
"""
runme.py

See "LICENSE" and "COPYRIGHT" for the full license governing this code.

"""

import multiprocessing as mp
import numpy as np
import pixel
import time
import sys
from calc import load
from functools import partial


def mainParallel(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    print argv

    # Loads parameters from file.
    parameters = {'Trigger': 0.5e-3, 'SampleRate': 10e6}
    npix = 128

    data = load.igor(argv[0])
    splitdata = np.split(data, npix, axis=1)

    fcn = partial(doStuff, parameters=parameters)

    start = time.clock()

    pool = mp.Pool(mp.cpu_count())
    out = zip(*pool.map(fcn, splitdata))

    end = time.clock()

    return (end - start)


def mainSerial(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    print argv

    # Loads parameters from file.
    parameters = {'Trigger': 0.5e-3, 'SampleRate': 10e6}
    npix = 128

    data = load.igor(argv[0])
    splitdata = np.split(data, npix, axis=1)

    fcn = partial(doStuff, parameters=parameters)

    start = time.clock()

    out = []

    for element in splitdata:

        out.append(fcn(element))

    end = time.clock()

    return (end-start)


def doStuff(element, parameters):

        pz = pixel.Pixel(element)
        pz.setParams(parameters)
        didx = pz.getResult()

        return didx

if __name__ == "__main__":
    sys.exit(mainParallel(sys.argv[1:]))
