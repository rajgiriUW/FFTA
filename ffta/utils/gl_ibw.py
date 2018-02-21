# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:06:12 2018

@author: Raj Giridharagopal, Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from os import path, remove  # File Path formatting
import numpy as np  # For array operations

from igor import binarywave as bw

#from .translator import Translator  # Because this class extends the abstract Translator class
#from .utils import generate_dummy_main_parms, build_ind_val_dsets
#from ..hdf_utils import getH5DsetRefs, linkRefs
#from ..io_hdf5 import ioHDF5  # Now the translator is responsible for writing the data.
#from ..microdata import MicroDataGroup, \
#    MicroDataset  # The building blocks for defining hierarchical storage in the H5 file

import pycroscopy as px

from pycroscopy.io.io_hdf5 import ioHDF5
from pycroscopy.io.translators.translator import Translator
from pycroscopy.io.translators.utils import generate_dummy_main_parms, build_ind_val_dsets
from pycroscopy.io.io_hdf5 import ioHDF5
from pycroscopy.io.hdf_utils import getH5DsetRefs, linkRefs
from pycroscopy.io.microdata import MicroDataGroup, MicroDataset

from ffta.utils import load

class GLIBWTranslator(Translator):
    """
    Translates Ginger Lab Igor Binary Wave (.ibw) files containing images or force curves to .h5
    """

    def translate(self, file_path, verbose=False, parm_encoding='utf-8', ftype='FF', subfolder='Measurement_0000'):
        """
        Translates the provided file to .h5
        Adapted heavily from pycroscopy IBW file, modified to work with Ginger format

        Parameters
        ----------
        file_path : String / unicode
            Absolute path of the .ibw file
        verbose : Boolean (Optional)
            Whether or not to show  print statements for debugging
        parm_encoding : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'
        ftype : str, optional
            Delineates Ginger Lab imaging file type to be imported (not case-sensitive)
            'FF' : FF-trEFM
            'SKPM' : FM-SKPM
            'ringdown' : Ringdown
            'trEFM' : normal trEFM
        subfolder : str, optional
            Specifies folder under root (/) to save data in. Default is standard pycroscopy format

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the .h5 file
        """

        # If subfolder improperly formatted
        if subfolder == '':
            subfolder = '/'

        # Load the ibw file first
        ibw_obj = bw.load(file_path)
        ibw_wave = ibw_obj.get('wave')
        parm_dict = self._read_parms(ibw_wave, parm_encoding)
        chan_labels, chan_units = self._get_chan_labels(ibw_wave, parm_encoding)
        if verbose:
            print('Channels and units found:')
            print(chan_labels)
            print(chan_units)

        chan_labels = self._get_image_type(ibw_wave, ftype)
        if verbose:
            print('Processing image type', ftype, 'with channels', chan_labels)

        # Get the data to figure out if this is an image or a force curve
        images = ibw_wave.get('wData')
        
        type_suffix = 'Image'
        
        num_rows = ibw_wave['wave_header']['nDim'][1] #lines
        num_cols = ibw_wave['wave_header']['nDim'][0] # points
        num_imgs = ibw_wave['wave_header']['nDim'][2] # layers
        unit_scale = self._get_unit_factor(''.join([str(s)[-2] for s in ibw_wave['wave_header']['dimUnits'][0][0:2]]))
        data_scale = self._get_unit_factor(str(ibw_wave['wave_header']['dataUnits'][0])[-2])
        
        parm_dict['FastScanSize'] = unit_scale * num_cols * ibw_wave['wave_header']['sfA'][0]
        parm_dict['SlowScanSize'] = unit_scale * num_rows * ibw_wave['wave_header']['sfA'][1]
        
        images = images.transpose(2, 0, 1)  # now ordered as [chan, Y, X] image
        images = np.reshape(images, (images.shape[0], -1, 1))  # 3D [chan, Y*X points,1]

        ds_pos_ind, ds_pos_val = build_ind_val_dsets([num_cols, num_rows], is_spectral=False,
                                                     steps=[1.0 * parm_dict['FastScanSize'] / num_cols,
                                                            1.0 * parm_dict['SlowScanSize'] / num_rows],
                                                     labels=['X', 'Y'], units=['m', 'm'], verbose=verbose)

        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([1], is_spectral=True, steps=[1],
                                                         labels=['arb'], units=['a.u.'], verbose=verbose)

        # Prepare the list of raw_data datasets
        chan_raw_dsets = list()
        for chan_data, chan_name, chan_unit in zip(images, chan_labels, chan_units):
            ds_raw_data = MicroDataset('Raw_Data', data=np.atleast_2d(chan_data), dtype=np.float32, compression='gzip')
            ds_raw_data.attrs['quantity'] = chan_name
            ds_raw_data.attrs['units'] = [chan_unit]
            chan_raw_dsets.append(ds_raw_data)
        if verbose:
            print('Finished preparing raw datasets')

        # Prepare the tree structure
        # technically should change the date, etc.
        spm_data = MicroDataGroup('')
        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = 'IgorIBW_' + type_suffix
        global_parms['translator'] = 'IgorIBW'
        spm_data.attrs = global_parms
        #meas_grp = MicroDataGroup('Measurement_000')
        meas_grp = MicroDataGroup(subfolder)
        meas_grp.attrs = parm_dict
        spm_data.addChildren([meas_grp])

        if verbose:
            print('Finished preparing tree trunk')

        # Prepare the .h5 file:
        folder_path, base_name = path.split(file_path)
        base_name = base_name[:-4]
        h5_path = path.join(folder_path, base_name + '.h5')
        if path.exists(h5_path):
            remove(h5_path)

        # Write head of tree to file:
        hdf = ioHDF5(h5_path)
        # spm_data.showTree()
        hdf.writeData(spm_data, print_log=verbose)

        if verbose:
            print('Finished writing tree trunk')

        # Standard list of auxiliary datasets that get linked with the raw dataset:
        aux_ds_names = ['Position_Indices', 'Position_Values', 'Spectroscopic_Indices', 'Spectroscopic_Values']

        # Create Channels, populate and then link:
        for chan_index, raw_dset in enumerate(chan_raw_dsets):
            #chan_grp = MicroDataGroup('{:s}{:03d}'.format('Channel_', chan_index), '/Measurement_000/')
            chan_grp = MicroDataGroup(chan_labels[chan_index], '/'+subfolder+'/')
            chan_grp.attrs['name'] = raw_dset.attrs['quantity']
            chan_grp.addChildren([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals, raw_dset])
            h5_refs = hdf.writeData(chan_grp, print_log=verbose)
            h5_raw = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
            linkRefs(h5_raw, getH5DsetRefs(aux_ds_names, h5_refs))

        if verbose:
            print('Finished writing all channels')

        hdf.close()
        return h5_path

    @staticmethod
    def _read_parms(ibw_wave, codec='utf-8'):
        """
        Parses the parameters in the provided dictionary

        Parameters
        ----------
        ibw_wave : dictionary
            Wave entry in the dictionary obtained from loading the ibw file
        codec : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'

        Returns
        -------
        parm_dict : dictionary
            Dictionary containing parameters
        """
        parm_string = ibw_wave.get('note')
        if type(parm_string) == bytes:
            parm_string = parm_string.decode(codec)
        parm_string = parm_string.rstrip('\r')
        parm_list = parm_string.split('\r')
        parm_dict = dict()
        for pair_string in parm_list:
            temp = pair_string.split(':')
            if len(temp) == 2:
                temp = [item.strip() for item in temp]
                try:
                    num = float(temp[1])
                    parm_dict[temp[0]] = num
                    try:
                        if num == int(num):
                            parm_dict[temp[0]] = int(num)
                    except OverflowError:
                        pass
                except ValueError:
                    parm_dict[temp[0]] = temp[1]

        # Grab the creation and modification times:
        other_parms = ibw_wave.get('wave_header')
        for key in ['creationDate', 'modDate', 'bname']:
            try:
                parm_dict[key] = other_parms[key]
            except KeyError:
                pass
        return parm_dict

    @staticmethod
    def _get_chan_labels(ibw_wave, codec='utf-8'):
        """
        Retrieves the names of the data channels and default units

        Parameters
        ----------
        ibw_wave : dictionary
            Wave entry in the dictionary obtained from loading the ibw file
        codec : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'

        Returns
        -------
        labels : list of strings
            List of the names of the data channels
        default_units : list of strings
            List of units for the measurement in each channel
        """
        temp = ibw_wave.get('labels')
        labels = []
        for item in temp:
            if len(item) > 0:
                labels += item
        for item in labels:
            if item == '':
                labels.remove(item)

        default_units = list()
        for chan_ind, chan in enumerate(labels):
            # clean up channel names
            if type(chan) == bytes:
                chan = chan.decode(codec)
            if chan.lower().rfind('trace') > 0:
                labels[chan_ind] = chan[:chan.lower().rfind('trace') + 5]
            # Figure out (default) units
            if chan.startswith('Phase'):
                default_units.append('deg')
            elif chan.startswith('Current'):
                default_units.append('A')
            else:
                default_units.append('m')

        return labels, default_units

    def _parse_file_path(self, input_path):
        pass

    def _read_data(self):
        pass


    def _get_image_type(self, ibw_wave,ftype):
        """
        Generates correct channel labels based on the passed filetype
        """
        
        if ftype.lower() == 'ff':
            
            return ['height', 'charging', 'shift']
            
        elif ftype.lower() == 'trefm':
            
            return ['height', 'charging', 'shift', 'error']
            
        elif ftype.lower() == 'skpm':
            
            return ['height', 'CPD']
        
        elif ftype.lower() == 'ringdown':
            
            return ['height', 'Q', 'error']
            
        elif ftype.lower() == 'pl':
      
            return ['PL_volts', 'PL_current']
        
        else:
            raise Exception('Improper File Type')
            
    def _get_unit_factor(self, unit):
        """
        Returns numerical conversion of unit label
        Unit : str
        """
        
        fc = {'um': 1e-6,
              'nm': 1e-9,
              'pm': 1e-12,
              'fm': 1e-15,
              'mm': 1e-3,
              'm': 1}
        
        if unit.lower() in fc.keys():
            return fc[unit]
        
