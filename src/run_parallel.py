"""run_parallel.py: Runs the FF-trEFM calculation for a set of given files."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import os
import sys
import multiprocessing
import pixel
import time
import argparse as ap
import numpy as np
from utils import load


def process_line(args):

    signal_file, n_pixels, parameters = args

    signal_array = load.signal(signal_file)

    num = int(signal_array.shape[1] / n_pixels)

    tfp = np.zeros(n_pixels)
    shift = np.zeros(n_pixels)

    # For every pixel in the file, runs the pixel.getresult.
    for i in range(n_pixels):

        temp = signal_array[:, (num * i):(num * (i + 1))]
        p = pixel.Pixel(temp, parameters)
        tfp[i], shift[i] = p.get_tfp()

    return tfp, shift


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    # Parse arguments from the command line, and print out help.
    p = ap.ArgumentParser(description='Data analysis software for FF-trEFM')
    p.add_argument('input', nargs=1,
                   help='data file (for image enter a directory)')
    p.add_argument('config', nargs=1, help='configuration file for parameters')
    p.add_argument('-v', action='version', version='FFtr-EFM 2.0b')
    args = p.parse_args(argv)

    # Load parameters from .cfg file.
    n_pixels, parameters = load.configuration(args.config[0])
    path = args.input[0]

    # Scan the path for .ibw files.
    filelist = os.listdir(path)
    filelist = filter(lambda name: name[-3:] == 'ibw', filelist)
    filelist = [os.path.join(path, name) for name in filelist]

    # Start timing.
    start_time = time.time()

    # Create a pool of workers.
    pool = multiprocessing.Pool(processes=None)

    n_files = len(filelist)
    # Create the iterable and map onto the function.
    iterable = zip(filelist, [n_pixels] * n_files, [parameters] * n_files)
    result = pool.map(process_line, iterable)

    pool.close()  # Do not forget to close spawned processes.
    pool.join()

    # Unzip the result.
    tfp_list, shift_list = zip(*result)

    # Initialize arrays.
    tfp = np.zeros((n_files, n_pixels))
    shift = np.zeros((n_files, n_pixels))

    # Convert list of arrays to 2D array.
    for i in range(n_files):

        tfp[i, :] = tfp_list[i]
        shift[i, :] = shift_list[i]

    # Save csv files.
    np.savetxt('tfp.csv', np.fliplr(tfp).T, delimiter=',')
    np.savetxt('shift.csv', np.fliplr(shift).T, delimiter=',')

    elapsed_time = time.time() - start_time

    print 'Time: ', elapsed_time

    return 0

if __name__ == '__main__':

    sys.exit(main(sys.argv[1:]))
