"""analyze.py: Runs the FF-trEFM Analysis for a set of given files."""

__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2019, Ginger Lab"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"
__status__ = "Development"

import argparse as ap
import logging
import multiprocessing
import os
import sys
import time

import numpy as np
from matplotlib import gridspec as gs
# Plotting imports
# mpl.use('WxAgg')
from matplotlib import pyplot as plt

import ffta.line as line
from .pixel_utils import badpixels
from .pixel_utils import load


def process_line(args):
    """
    Wrapper function for line class, used in parallel processing.
    
    :param args:
    :type args:
    
    :returns: tuple (tfp, shift)
        WHERE
        [type] tfp is...
        [type] shift is...
    """

    signal_file, params, n_pixels = args
    signal_array = load.signal(signal_file)

    line_inst = line.Line(signal_array, params, n_pixels)
    tfp, shift, _ = line_inst.analyze()

    return tfp, shift


def main(argv=None):
    """
    Main function of the executable file.
    
    :param argv:
    :type argv:
    """
    logging.basicConfig(filename='error.log', level=logging.INFO)

    # Get the CPU count to display in help.
    cpu_count = multiprocessing.cpu_count()

    if argv is None:
        argv = sys.argv[1:]

    # Parse arguments from the command line, and print out help.
    parser = ap.ArgumentParser(description='Analysis software for FF-trEFM')
    parser.add_argument('path', nargs='?', default=os.getcwd(),
                        help='path to directory')
    parser.add_argument('-p', help='parallel computing option should be'
                                   'followed by the number of CPUs.', type=int,
                        choices=range(2, cpu_count + 1))
    parser.add_argument('-v', action='version',
                        version='FFtr-EFM 2.0 Release Candidate')
    args = parser.parse_args(argv)

    # Scan the path for .ibw and .cfg files.
    path = args.path
    filelist = os.listdir(path)

    data_files = [os.path.join(path, name)
                  for name in filelist if name[-3:] == 'ibw']

    config_file = [os.path.join(path, name)
                   for name in filelist if name[-3:] == 'cfg'][0]

    # Load parameters from .cfg file.
    n_pixels, parameters = load.configuration(config_file)

    print('Recombination: ', parameters['recombination'])
    if 'phase_fitting' in parameters:
        print('Phase fitting: ', parameters['phase_fitting'])

    print('ROI: ', parameters['roi'])

    if not args.p:

        # Initialize arrays.
        tfp = np.zeros((len(data_files), n_pixels))
        shift = np.zeros((len(data_files), n_pixels))

        # Initialize plotting.
        plt.ion()

        fig = plt.figure(figsize=(12, 6), tight_layout=True)
        grid = gs.GridSpec(1, 2)
        tfp_ax = plt.subplot(grid[0, 0])
        shift_ax = plt.subplot(grid[0, 1])

        plt.setp(tfp_ax.get_xticklabels(), visible=False)
        plt.setp(tfp_ax.get_yticklabels(), visible=False)
        plt.setp(shift_ax.get_xticklabels(), visible=False)
        plt.setp(shift_ax.get_yticklabels(), visible=False)

        tfp_ax.set_title('tFP Image')
        shift_ax.set_title('Shift Image')

        kwargs = {'origin': 'lower', 'aspect': 'equal'}

        tfp_image = tfp_ax.imshow(tfp * 1e6, cmap='afmhot', **kwargs)
        shift_image = shift_ax.imshow(shift, cmap='cubehelix', **kwargs)
        text = plt.figtext(0.4, 0.1, '')
        plt.show()

        # Load every file in the file list one by one.
        for i, data_file in enumerate(data_files):
            signal_array = load.signal(data_file)
            line_inst = line.Line(signal_array, parameters, n_pixels)
            tfp[i, :], shift[i, :], _ = line_inst.analyze()
            #            line_inst = line.Line(signal_array, parameters, n_pixels,fitphase=True)
            #            tfpphase[i, :], _, _ = line_inst.analyze()

            tfp_image = tfp_ax.imshow(tfp * 1e6, cmap='inferno', **kwargs)
            shift_image = shift_ax.imshow(shift, cmap='cubehelix', **kwargs)

            tfp_sc = tfp[tfp.nonzero()] * 1e6
            tfp_image.set_clim(vmin=tfp_sc.min(), vmax=tfp_sc.max())

            shift_sc = shift[shift.nonzero()]
            shift_image.set_clim(vmin=shift_sc.min(), vmax=shift_sc.max())

            tfpmean = 1e6 * tfp[i, :].mean()
            tfpstd = 1e6 * tfp[i, :].std()

            string = ("Line {0:.0f}, average tFP (us) ="
                      " {1:.2f} +/- {2:.2f}".format(i + 1, tfpmean, tfpstd))

            text.remove()
            text = plt.figtext(0.35, 0.1, string)

            plt.draw()
            plt.pause(0.0001)

            del line_inst  # Delete the instance to open up memory.

    elif args.p:

        print('Starting parallel processing, using {0:1d} \
               CPUs.'.format(args.p))
        start_time = time.time()  # Keep when it's started.

        # Create a pool of workers.
        pool = multiprocessing.Pool(processes=args.p)

        # Create the iterable and map onto the function.
        n_files = len(data_files)
        iterable = zip(data_files, [parameters] * n_files,
                       [n_pixels] * n_files)
        result = pool.map(process_line, iterable)

        # Do not forget to close spawned processes.
        pool.close()
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

        elapsed_time = time.time() - start_time

        print('It took {0:.1f} seconds.'.format(elapsed_time))

    # Filter bad pixels
    tfp_fixed, _ = badpixels.fix_array(tfp, threshold=2)
    tfp_fixed = np.array(tfp_fixed)

    # Save csv files.
    os.chdir(path)
    np.savetxt('tfp.csv', np.fliplr(tfp).T, delimiter=',')
    np.savetxt('shift.csv', np.fliplr(shift).T, delimiter=',')
    np.savetxt('tfp_fixed.csv', np.fliplr(tfp_fixed).T, delimiter=',')

    return


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
