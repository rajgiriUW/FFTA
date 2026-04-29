"""calibration_curve.py: Generates a tFP calibration curve from simulated cantilever responses."""
__author__ = "Rajiv Giridharagopal"
__copyright__ = "Copyright 2012-2026, Rajiv Giridharagopal"
__maintainer__ = "Rajiv Giridharagopal"
__email__ = "rgiri@uw.edu"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import medfilt

from ffta.pixel_utils.load import configuration
from .mechanical_drive import MechanicalDrive
from .utils.load import params_from_experiment as load_parm
from .utils.load import simulation_configuration as load_sim_config


def cal_curve(can_path, param_cfg, taus_range=[-7, -3], plot=True, **kwargs):
    """
    Generate a tFP calibration curve by sweeping tau over a simulated cantilever.

    Ideally you would have a tip parameters file as well. Note that taus, tfps,
    and the returned spline are all in log-space.

    Parameters
    ----------
    can_path : str or tuple
        Path to cantilever parameters .txt file, or a tuple of simulation
        config paths accepted by load_sim_config.
    param_cfg : str
        Path to parameters.cfg file from the FFtrEFM experiment data folder.
    taus_range : list of float, optional
        Two-element [low, high] range for the tau sweep. Negative values are
        interpreted as log10 exponents (e.g. [-7, -3] → 1e-7 to 1e-3 s);
        positive values are converted to log10 automatically. Default [-7, -3].
    plot : bool, optional
        Plot the last simulated pixel and the calibration curve. Default True.
    **kwargs
        Additional keyword arguments passed to cant.analyze() (e.g. roi, n_taps).

    Returns
    -------
    taus : ndarray
        Log-space single-exponential time constants that were simulated.
    tfps : ndarray
        Log-space measured time-to-first-peak values.
    spl : InterpolatedUnivariateSpline or None
        Spline fit of the calibration curve (tfps → taus, both in log-space).
        None if spline fitting failed.

    Examples
    --------
    >>> taus, tfps, spl = cal_curve('cantilever.txt', 'parameters.cfg')
    >>> plt.plot(np.exp(tfps), np.exp(taus), 'bX-')

    >>> # Change fit parameters per tau
    >>> taus, tfps, spl = cal_curve('cantilever.txt', 'parameters.cfg', roi=0.001, n_taps=199)
    """
    if isinstance(can_path, str):
        can_params, force_params, sim_params, _, parms = load_parm(can_path, param_cfg)
    elif isinstance(can_path, tuple):
        can_params, force_params, sim_params = load_sim_config(can_path)
        _, parms = configuration(param_cfg)
        can_params['drive_freq'] = parms['drive_freq']
        can_params['res_freq'] = parms['drive_freq']
        sim_params['trigger'] = parms['trigger']
        sim_params['total_time'] = parms['total_time']
        sim_params['sampling_rate'] = parms['sampling_rate']

    if len(taus_range) != 2 or (taus_range[1] <= taus_range[0]):
        raise ValueError('Range must be ascending and 2-items')

    # Check if given as log or actual values
    if taus_range[0] < 0 or taus_range[1] < 0:
        _rlo = taus_range[0]
        _rhi = taus_range[1]

    else:
        _rlo = np.log10(taus_range[0])
        _rhi = np.log10(taus_range[1])

    taus = np.logspace(_rlo, _rhi, 50)
    tfps = []

    for t in taus:
        force_params['tau'] = t
        cant = MechanicalDrive(can_params, force_params, sim_params)
        Z, _ = cant.simulate()
        try:
            pix = cant.analyze(plot=False, **kwargs)
            tfps.append(pix.tfp)
        except:
            print('Error', t)

    # sort the arrays
    taus = taus[np.argsort(tfps)]
    tfps = np.sort(tfps)

    # Splines work better on shorter lengthscales
    taus = np.log(taus)
    tfps = np.log(tfps)

    # Error corrections
    # negative x-values (must be monotonic for spline)
    dtfp = np.diff(tfps)
    tfps = np.array(tfps)
    taus = np.array(taus)
    tfps = np.delete(tfps, np.where(dtfp < 0)[0])
    taus = np.delete(taus, np.where(dtfp < 0)[0])

    # "hot" pixels in the cal-curve
    hotpixels = np.abs(taus - medfilt(taus))
    taus = np.delete(taus, np.where(hotpixels > 0))
    tfps = np.delete(tfps, np.where(hotpixels > 0))

    # Negative slopes
    neg_slope = np.diff(taus) / np.diff(tfps)
    while np.any(neg_slope < 0):
        tfps = np.delete(tfps, np.where(neg_slope < 0)[0])
        taus = np.delete(taus, np.where(neg_slope < 0)[0])
        neg_slope = np.diff(taus) / np.diff(tfps)

    # Infinite slopes (tfp saturation at long taus)
    while np.any(np.isinf(neg_slope)):
        tfps = np.delete(tfps, np.where(np.isinf(neg_slope))[0])
        taus = np.delete(taus, np.where(np.isinf(neg_slope))[0])
        neg_slope = np.diff(taus) / np.diff(tfps)

    try:
        spl = ius(tfps, taus, k=4)
    except:
        print('=== Error generating cal-curve. Check manually ===')
        spl = None
        print(taus)
        print(tfps)

    if plot:
        pix.plot()
        fig, ax = plt.subplots(facecolor='white')
        ax.loglog(np.exp(tfps), np.exp(taus), 'bX-')
        try:
            ax.loglog(np.exp(tfps), np.exp(spl(tfps)), 'r--')
        except:
            pass
        ax.set_xlabel('$t_{fp}$ (s)')
        ax.set_ylabel(r'$\tau$ (s)')
        ax.set_title('Calibration curve')

    # Save Calibration Curve
    df = pd.DataFrame(index=taus, data=tfps)
    df = df.rename(columns={0: 'tfps'})
    df.index.name = 'taus'
    df.to_csv('Calibration_Curve.csv')

    print('Do not forget that the spline is in log-space')

    return taus, tfps, spl


def reconstruct_cal_curve(csv_file):
    """
    Reconstruct a calibration curve spline from a previously saved CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file written by cal_curve.

    Returns
    -------
    taus : Series
        Log-space time constants read from the CSV index.
    tfps : Series
        Log-space tFP values read from the CSV.
    spl : InterpolatedUnivariateSpline
        Spline fit of the calibration curve (tfps → taus, both in log-space).
    """

    df = pd.read_csv(csv_file)

    taus = df['taus']
    tfps = df['tfps']
    spl = ius(tfps, taus, k=4)

    print('Note that taus, tfps, and cal_curve are in log-space')

    return taus, tfps, spl
