# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:06:12 2018

@author: Raj Giridharagopal
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from os import path  # File Path formatting
import numpy as np  # For array operations

from igor import binarywave as bw

# from .translator import Translator  # Because this class extends the abstract Translator class
# from .utils import generate_dummy_main_parms, build_ind_val_dsets
# from ..hdf_utils import getH5DsetRefs, linkRefs
# from ..io_hdf5 import ioHDF5  # Now the translator is responsible for writing the data.
# from ..microdata import MicroDataGroup, \
#    MicroDataset  # The building blocks for defining hierarchical storage in the H5 file

from pyUSID.io.translator import Translator  # , generate_dummy_main_parms

from pycroscopy.io.write_utils import Dimension
from pyUSID.io.hdf_utils import write_ind_val_dsets
from pyUSID.io.hdf_utils import write_main_dataset, create_indexed_group
import h5py


class GLIBWTranslator(Translator):
	"""
	Translates Ginger Lab Igor Binary Wave (.ibw) files containing images or force curves to .h5
	"""

	def translate(self, file_path, verbose=False, parm_encoding='utf-8', ftype='FF',
				  subfolder='Measurement_000', h5_path='', channel_label_name=True):
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
		h5_path : str, optional
			Existing H5 file to append to
		channel_label_name : bool, optional
			If True, uses the Channel as the subfolder name (e.g. Height, Phase, Amplitude, Charging)

		Returns
		-------
		h5_path : String / unicode
			Absolute path of the .h5 file
		"""

		# Prepare the .h5 file:
		if not any(h5_path):
			folder_path, base_name = path.split(file_path)
			base_name = base_name[:-4]
			h5_path = path.join(folder_path, base_name + '.h5')
			# hard-coded exception, rarely occurs but can be useful
			if path.exists(h5_path):
				h5_path = path.join(folder_path, base_name + '_00.h5')

		h5_file = h5py.File(h5_path, 'w')

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

		# Get the data to figure out if this is an image or a force curve
		images = ibw_wave.get('wData')

		if images.shape[2] != len(chan_labels):
			chan_labels = chan_labels[1:]  # for weird null set errors in older AR software

		# Check if a Ginger Lab format ibw (has 'UserIn' in channel labels)
		_is_gl_type = any(['UserIn0' in str(s) for s in chan_labels])
		if _is_gl_type == True:
			chan_labels = self._get_image_type(chan_labels, ftype)

		if verbose:
			print('Processing image type', ftype, 'with channels', chan_labels)

		type_suffix = 'Image'

		num_rows = ibw_wave['wave_header']['nDim'][1]  # lines
		num_cols = ibw_wave['wave_header']['nDim'][0]  # points
		num_imgs = ibw_wave['wave_header']['nDim'][2]  # layers
		unit_scale = self._get_unit_factor(''.join([str(s)[-2] for s in ibw_wave['wave_header']['dimUnits'][0][0:2]]))
		data_scale = self._get_unit_factor(str(ibw_wave['wave_header']['dataUnits'][0])[-2])

		parm_dict['FastScanSize'] = unit_scale * num_cols * ibw_wave['wave_header']['sfA'][0]
		parm_dict['SlowScanSize'] = unit_scale * num_rows * ibw_wave['wave_header']['sfA'][1]

		images = images.transpose(2, 0, 1)  # now ordered as [chan, Y, X] image
		images = np.reshape(images, (images.shape[0], -1, 1))  # 3D [chan, Y*X points,1]

		pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
					Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]
		spec_desc = Dimension('arb', 'a.u.', [1])

		# Create Position and spectroscopic datasets
		h5_pos_inds, h5_pos_vals = write_ind_val_dsets(h5_file['/'], pos_desc, is_spectral=False)
		h5_spec_inds, h5_spec_vals = write_ind_val_dsets(h5_file['/'], spec_desc, is_spectral=True)

		# Prepare the list of raw_data datasets
		for chan_data, chan_name, chan_unit in zip(images, chan_labels, chan_units):
			chan_grp = create_indexed_group(h5_file['/'], chan_name)
			write_main_dataset(chan_grp, np.atleast_2d(chan_data), 'Raw_Data',
							   chan_name, chan_unit,
							   pos_desc, spec_desc,
							   dtype=np.float32)

		if verbose:
			print('Finished writing all channels')

		h5_file.close()
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
			try:
				parm_string = parm_string.decode(codec)
			except:
				parm_string = parm_string.decode('ISO-8859-1')
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

	def _get_image_type(self, ibw_wave, ftype):
		"""
		Generates correct channel labels based on the passed filetype
		"""

		if ftype.lower() == 'ff':

			del ibw_wave[0:3]
			ibw_wave = ['height', 'charging', 'shift'] + ibw_wave

		elif ftype.lower() == 'trefm':

			del ibw_wave[0:4]
			ibw_wave = ['height', 'charging', 'shift', 'error'] + ibw_wave

		elif ftype.lower() == 'skpm':

			del ibw_wave[0:2]
			ibw_wave = ['height', 'CPD'] + ibw_wave

		elif ftype.lower() == 'ringdown':

			del ibw_wave[0:4]
			ibw_wave = ['height', 'Q', 'shift', 'error'] + ibw_wave

		elif ftype.lower() == 'pl':

			del ibw_wave[0:2]
			ibw_wave = ['PL_volts', 'PL_current'] + ibw_wave

		else:
			raise Exception('Improper File Type')

		return ibw_wave

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

		elif unit[0] in fc.keys():
			return fc[unit[0]]
