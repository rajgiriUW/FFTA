"""run.py: Runs the FF-trEFM calculation for a set of given files."""

__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import sys
import os
import pixel
import argparse as ap
import numpy as np
from utils import load
from progressbar import ProgressBar, ETA, Percentage


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

    # Set the progress bar.
    widgets = [Percentage(), ' | ', ETA()]

    # Check if given argument is a path to directory or a file.
    if os.path.isdir(args.input[0]):

        os.chdir(args.input[0])
        filelist = []

        # List files ending with .ibw or.txt in the given directory.
        for files in os.listdir('.'):

            if files.endswith('.ibw') or files.endswith('.txt'):

                filelist.append(files)

        filelist.sort()  # Sort the list alphabetically.

        # If there is nothing, raise error and exit.
        if not filelist:

            raise SystemExit('There is no file in the given directory!')

        # Initialize arrays.
        tfp = np.zeros((len(filelist), n_pixels))
        shift = np.zeros((len(filelist), n_pixels))

        # Create a progress bar with defined widgets.
        pbar = ProgressBar(widgets=widgets, maxval=len(filelist)).start()

        # Load every file in the file list one by one.
        for n in range(len(filelist)):

            signal_array = load.signal(filelist[n])
            num = int(signal_array.shape[1] / n_pixels)

            # For every pixel in the file, runs the pixel.getresult.
            for m in range(n_pixels):

                temp = signal_array[:, (num * m):(num * (m + 1))]
                p = pixel.Pixel(temp, parameters)
                tfp[n, m], shift[n, m] = p.get_tfp()

            tfpmean = 1e6 * tfp[n, :].mean()
            tfpstd = 1e6 * tfp[n, :].std()

            print ("For line {0:.0f}, average tFP (us) ="
                   " {1:.2f} +/- {2:.2f}".format(n + 1, tfpmean, tfpstd))
            pbar.update(n + 1)

        # Save csv files.
        np.savetxt('tfp.csv', np.fliplr(tfp).T, delimiter=',')
        np.savetxt('shift.csv', np.fliplr(shift).T, delimiter=',')

        # Finish the progress bar.
        pbar.finish()

    # If given argument is a file, runs pixel.get_tfp for single pixel.
    elif os.path.isfile(args.input[0]) and args.input[0].endswith('.ibw'):

        signal_array = load.signal(args.input[0])
        p = pixel.Pixel(signal_array, parameters)
        tfp, shift = p.get_tfp()

        print "tFP (us) = {0:.3f}".format(tfp * 1e6)
        print "Frequency Shift = {0:.3f}".format(shift)

    # If no condition holds, exits the process.
    else:

        raise SystemExit("Unrecoverable error! Exiting the program!!!")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
