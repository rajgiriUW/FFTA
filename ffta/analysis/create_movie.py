import matplotlib.animation as animation
import numpy as np
import pyUSID as usid
import sidpy
from matplotlib import pyplot as plt
from scipy import signal as sps

'''
Note: A 'str is not callable' bug is often due to not running set_mpeg

'''


def set_mpeg(path=None):
    '''

    :param path:
    :type path: str
    '''
    if not path:
        plt.rcParams[
            'animation.ffmpeg_path'] = r'C:/Users/Raj/Downloads/ffmpeg/ffmpeg/bin/ffmpeg.exe'

    else:
        plt.rcParams['animation.ffmpeg_path'] = path
    return


def setup_movie(h5_ds, size=(10, 6), vscale=[None, None], cmap='inferno'):
    '''

    :param h5_ds: The instantaneous frequency data. NOT deflection, this is for post-processed data
    :type h5_ds: USID dataset

    :param size:
    :type size: tuple

    :param vscale:
    :type vscale: list

    :param cmap:
    :type cmap: str

    :returns: tuple (fig, ax, cbar, vmin, vmax)
        WHERE
        [type] fig is...
        [type] ax is...
        [type] cbar is...
        [type] vmin is...
        [type vmax is...
    '''
    fig, ax = plt.subplots(nrows=1, figsize=size, facecolor='white')

    if 'USID' not in str(type(h5_ds)):
        h5_ds = usid.USIDataset(h5_ds)

    params = sidpy.hdf_utils.get_attributes(h5_ds)
    if 'trigger' not in params:
        params = sidpy.hdf_utils.get_attributes(h5_ds.parent)

    ds = h5_ds.get_n_dim_form()[:, :, 0]

    # set scale based on the first line, pre-trigger to post-trigger
    tdx = params['trigger'] * params['sampling_rate'] / params['pnts_per_avg']
    tdx = int(tdx * len(h5_ds[0, :]))
    if any(vscale):
        [vmin, vmax] = vscale
    else:
        vmin = np.min(h5_ds[0][int(tdx * 0.7): int(tdx * 1.3)])
        vmax = np.max(h5_ds[0][int(tdx * 0.7): int(tdx * 1.3)])

    length = h5_ds.get_pos_values('X')
    height = h5_ds.get_pos_values('Y')

    im0 = ax.imshow(ds, cmap=cmap, origin='lower',
                    extent=[0, length[-1] * 1e6, 0, height[-1] * 1e6],
                    vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im0, ax=ax, orientation='vertical',
                        fraction=0.023, pad=0.03, use_gridspec=True)

    return fig, ax, cbar, vmin, vmax


def create_freq_movie(h5_ds,
                      filename='inst_freq',
                      time_step=50,
                      idx_start=500,
                      idx_stop=100,
                      smooth=None,
                      size=(10, 6),
                      vscale=[None, None],
                      cmap='inferno',
                      interval=60,
                      repeat_delay=100,
                      crop=None):
    '''
    Creates an animation that goes through all the instantaneous frequency data.

    :param h5_ds: The instantaneous frequency data. NOT deflection, this is for post-processed data
    :type h5_ds: USID dataset

    :param filename:
    :type filename: str

    :param time_step:
        10 @ 10 MHz = 1 us
        50 @ 10 MHz = 5 us
    :type time_step: int, optional

    :param idx_start: What index to start at. Typically to avoid the Hilbert Transform edge artifacts, you start a little ahead
    :type idx_start: int

    :param idx_stop: Same as the above,in terms of how many points BEFORE the end to stop
    :type idx_stop: int

    :param smooth: Whether to apply a simple boxcar smoothing kernel to the data
    :type smooth: int, optional

    :param size: figure size
    :type size: tuple, optional

    :param vscale: To hard-code the color scale, otherwise these are automatically generated
    :type vscale: list [float, float], optional

    :param cmap:
    :type cmap: str, optional

    :param interval:
    :type interval:

    :param repeat_delay: Used when saving to set the delay for when the mp4 repeats from the start
    :type repeat_delay: int

    :param crop: Crops the image to a certain line, in case part of the scan is bad
    :type crop: int

    '''

    if not isinstance(h5_ds, usid.USIDataset):
        h5_ds = usid.USIDataset(h5_ds)

    if any(vscale):
        fig, ax, cbar, _, _ = setup_movie(h5_ds, size, vscale, cmap=cmap)
        [vmin, vmax] = vscale
    else:
        fig, ax, cbar, vmin, vmax = setup_movie(h5_ds, size, cmap=cmap)

    _orig = np.copy(h5_ds[()])
    length = h5_ds.get_pos_values('X')
    height = h5_ds.get_pos_values('Y')
    if isinstance(crop, int):
        height = height * crop / h5_ds.get_n_dim_form()[:, :, 0].shape[0]

    params = sidpy.hdf_utils.get_attributes(h5_ds)
    if 'trigger' not in params:
        params = sidpy.hdf_utils.get_attributes(h5_ds.parent)

    if isinstance(smooth, int):
        kernel = np.ones(smooth) / smooth
        for i in np.arange(h5_ds.shape[0]):
            h5_ds[i, :] = sps.fftconvolve(h5_ds[i, :], kernel, mode='same')

    cbar.set_label('Frequency (Hz)', rotation=270, labelpad=20, fontsize=16)

    tx = h5_ds.get_spec_values('Time')

    # Loop through time segments
    ims = []
    for k, t in zip(np.arange(idx_start, len(tx) - idx_stop, time_step),
                    tx[idx_start:-idx_stop:time_step]):

        _if = h5_ds.get_n_dim_form()[:, :, k]
        if isinstance(crop, int):
            if crop < 0:
                _if = _if[crop:, :]
            else:
                _if = _if[:crop, :]

        htitle = 'at ' + '{0:.4f}'.format(t * 1e3) + ' ms'
        im0 = ax.imshow(_if, cmap=cmap, origin='lower', animated=True,
                        extent=[0, length[-1] * 1e6, 0, height[-1] * 1e6],
                        vmin=vmin, vmax=vmax)

        if t > params['trigger']:
            tl0 = ax.text(length[-1] * 1e6 / 2 - .35, height[-1] * 1e6 + .01,
                          htitle + ' ms, TRIGGER', color='blue', weight='bold', fontsize=16)
        else:
            tl0 = ax.text(length[-1] * 1e6 / 2 - .35, height[-1] * 1e6 + .01,
                          htitle + ' ms, PRE-TRIGGER', color='black', weight='regular', fontsize=14)

        ims.append([im0, tl0])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=repeat_delay)

    try:
        ani.save(filename + '.mp4')
    except TypeError as e:
        print(e)
        print('A "str is not callable" message is often due to not running set_mpeg function')

    # restore data
    for i in np.arange(h5_ds.shape[0]):
        h5_ds[i, :] = _orig[i, :]

    return


def create_cpd_movie(h5_ds, filename='cpd', size=(10, 6),
                     vscale=[None, None], cmap='inferno', smooth=None, interval=60, repeat_delay=100):
    '''

    :param h5_ds: The instantaneous frequency data. NOT deflection, this is for post-processed data
    :type h5_ds: USID dataset

    :param filename:
    :type filename: str

    :param size: figure size
    :type size: tuple, optional

    :param vscale: To hard-code the color scale, otherwise these are automatically generated
    :type vscale: list [float, float], optional

    :param interval:
    :type interval:

    :param repeat_delay: Used when saving to set the delay for when the mp4 repeats from the start
    :type repeat_delay: int

    '''

    if not isinstance(h5_ds, usid.USIDataset):
        h5_ds = usid.USIDataset(h5_ds)

    if any(vscale):
        fig, ax, cbar, _, _ = setup_movie(h5_ds, size, vscale, cmap=cmap)
        [vmin, vmax] = vscale
    else:
        fig, ax, cbar, vmin, vmax = setup_movie(h5_ds, size, cmap=cmap)

    cbar.set_label('Potential (V)', rotation=270, labelpad=20, fontsize=16)

    _orig = np.copy(h5_ds[()])
    length = h5_ds.get_pos_values('X')
    height = h5_ds.get_pos_values('Y')
    params = sidpy.hdf_utils.get_attributes(h5_ds)
    if 'trigger' not in params:
        params = sidpy.hdf_utils.get_attributes(h5_ds.parent)

    if isinstance(smooth, int):
        kernel = np.ones(smooth) / smooth
        for i in np.arange(h5_ds.shape[0]):
            h5_ds[i, :] = sps.fftconvolve(h5_ds[i, :], kernel, mode='same')

    tx = h5_ds.get_spec_values('Time')

    # Loop through time segments
    ims = []
    for k, t in enumerate(tx):

        _if = h5_ds.get_n_dim_form()[:, :, k]
        htitle = 'at ' + '{0:.4f}'.format(t * 1e3) + ' ms'
        im0 = ax.imshow(_if, cmap=cmap, origin='lower', animated=True,
                        extent=[0, length[-1] * 1e6, 0, height[-1] * 1e6],
                        vmin=vmin, vmax=vmax)

        if t > params['trigger']:
            tl0 = ax.text(length[-1] * 1e6 / 2 - .35, height[-1] * 1e6 + .01,
                          htitle + ' ms, TRIGGER', color='blue', weight='bold', fontsize=16)
        else:
            tl0 = ax.text(length[-1] * 1e6 / 2 - .35, height[-1] * 1e6 + .01,
                          htitle + ' ms, PRE-TRIGGER', color='black', weight='regular', fontsize=14)

        ims.append([im0, tl0])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=repeat_delay)

    try:
        ani.save(filename + '.mp4')
    except TypeError:
        print('A "str is not callable" message is often due to not running set_mpeg function')

    # restore data
    for i in np.arange(h5_ds.shape[0]):
        h5_ds[i, :] = _orig[i, :]

    return
