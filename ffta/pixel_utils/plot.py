"""plot.py: Visualization helpers for Pixel objects."""

import numpy as np
from matplotlib import pyplot as plt


def plot_pixel(px, newplot=True, fit=True):
    """
    Quick visualization of instantaneous frequency, amplitude, and phase.

    Parameters
    ----------
    px : Pixel
        A processed Pixel instance (analyze() must have been called).
    newplot : bool, optional
        Generate a new figure. If False, plot on the current axes. Default True.
    fit : bool, optional
        Overlay the best-fit curve on the frequency panel. Default True.
    """
    if newplot:
        fig, a = plt.subplots(nrows=3, figsize=(6, 9), facecolor='white')

    dt = 1 / px.sampling_rate
    ridx = int(px.roi * px.sampling_rate)
    fidx = int(px.tidx)

    cut = [fidx, fidx + ridx]
    tx = np.arange(cut[0], cut[1]) * dt

    a[0].plot(tx * 1e3, px.inst_freq[cut[0]:cut[1]], 'r-')

    if fit:
        if px.fit_form == 'ringdown':
            a[1].plot(tx * 1e3, px.best_fit, 'g--')
        elif px.fit_form == 'exp' and px.method == 'nfmd':
            a[2].plot(tx * 1e3, px.best_fit + px.phase[px.tidx], 'g--')
        else:
            a[0].plot(tx * 1e3, px.best_fit, 'g--')

    a[1].plot(tx * 1e3, px.amplitude[cut[0]:cut[1]], 'b')
    if px.fit_form == 'exp' and px.method == 'nfmd':
        a[2].plot(tx * 1e3, px.phase[cut[0]:cut[1]], 'm')
    else:
        a[2].plot(tx * 1e3, px.phase[cut[0]:cut[1]] * 180 / np.pi, 'm')

    a[0].set_title('Instantaneous Frequency')
    a[0].set_ylabel('Frequency Shift (Hz)')
    a[1].set_ylabel('Amplitude (nm)')
    a[2].set_ylabel('Phase (deg)')
    a[2].set_xlabel('Time (ms)')

    plt.tight_layout()


def plot_dc_removal(px, sig_dc, signal_corrected):
    """
    Visualize the DC component removed and the before/after signals.

    Parameters
    ----------
    px : Pixel
        The Pixel instance, used for sampling_rate and total_time.
    sig_dc : array_like
        The extracted DC component (1-D, n_points).
    signal_corrected : array_like
        The DC-corrected signal (1-D, n_points).
    """
    t = np.arange(0, px.total_time, 1 / px.sampling_rate)

    fig, ax = plt.subplots(nrows=2, figsize=(6, 10))
    ax[0].plot(t, sig_dc, 'b')
    ax[0].set_title('DC Offset')
    ax[1].plot(t, px.signal_array, 'b', label='original')
    ax[1].plot(t, signal_corrected, 'r', label='corrected')
    ax[1].legend()

    plt.tight_layout()
