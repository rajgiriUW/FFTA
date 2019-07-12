
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import pyUSID as usid
import ffta
import numpy as np

def set_mpeg(path=None):

    if not any(path):
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Raj\Downloads\ffmpeg-20180124-1948b76-win64-static\bin\ffmpeg.exe'
        
    return

def create_movie(h5_ds, rotate=True, filename='inst_freq', time_step=50):
    '''
    Creates an animation that goes through all the instantaneous frequency data.
    
    h5_ds : USID Dataset 
        The instantaneous frequency data. NOT deflection, this is for post-processed data
        
    rotate : bool, optional
        The data are saved in ndim_form in pycroscopy rotated 90 degrees
        
    time_step : int, optional
        10 @ 10 MHz = 1 us
        50 @ 10 MHz = 5 us
    
    '''
    fig, ax = plt.subplots(nrows=1, figsize=(10, 6), facecolor='white')
    
    if 'USID' not in str(type(h5_ds)):
        h5_ds = usid.USIDataset(h5_ds)
    
    tx = h5_ds.get_spec_values('Time')
    params = ffta.hdf_utils.get_utils.get_params(h5_ds)
    
    ds = h5_ds.get_n_dim_form()[:,:,0].T

    tdx = int(params['trigger'] * params['sampling_rate'])
    vmin = np.min(h5_ds[1000][int(tdx * 0.7): int(tdx * 1.3)])
    vmax = np.max(h5_ds[1000][int(tdx * 0.7): int(tdx * 1.3)])
        
    length = h5_ds.get_pos_values('X')
    height = h5_ds.get_pos_values('Y')
    
    im0 = ax.imshow(ds, cmap='inferno', origin='lower',
                    extent=[0, length[-1]*1e6, 0, height[-1]*1e6],
                    vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im0, ax=ax, orientation='vertical',
                    fraction=0.023, pad=0.03, use_gridspec=True)
    cbar.set_label('Frequency (Hz)', rotation=270, labelpad = 20, fontsize=16)
    # Loop through time segments
    ims = []
    for k, t in zip(np.arange(500, len(tx)-100, time_step), tx[500:-100:time_step]):
        
        _if = h5_ds.get_n_dim_form()[:,:,k].T
        htitle = 'at '+ '{0:.4f}'.format(t*1e3)+ ' ms'
        im0 = ax.imshow(_if, cmap='inferno', origin='lower',animated=True,
                        extent=[0, length[-1]*1e6, 0, height[-1]*1e6],
                        vmin=vmin, vmax=vmax)
    
        if t > params['trigger']:
            tl0 = ax.text(length[-1]*1e6/2 - .35, height[-1]*1e6+.01, 
                            htitle + ' ms, TRIGGER', color='blue', weight='bold', fontsize=14)
        else:
            tl0 = ax.text(length[-1]*1e6/2 - .35, height[-1]*1e6+.01, 
                             htitle + ' ms, PRE-TRIGGER', color='black', weight='regular', fontsize=14)
    
        ims.append([im0, tl0])
    
    ani = animation.ArtistAnimation(fig, ims, interval=60,repeat_delay=100)

    ani.save(filename + '.mp4')
    
    return