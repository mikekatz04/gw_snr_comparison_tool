import numpy as np
import scipy.constants as ct
import pdb
import scipy.constants as ct
import datetime
import time
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
from astropy.io import ascii
import lal
import lalsimulation
from scipy import integrate
import fd_waveform_generator as fdw
from scipy.misc import derivative
from scipy.optimize import fsolve
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline

import sys
from collections import OrderedDict

import h5py

import json

from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

Msun=1.989e30



def parallel_func(num_proc, binaries, sig_types, sensitivity_dict):
	print(num_proc,'start', len(binaries.M))
	#initialize SNR array
	snr = OrderedDict()
	for sc in sensitivity_dict.keys():
		for sig_type in sig_types:
			snr[sc + '_' + sig_type] = np.zeros(len(binaries.M))

	binaries.create_waveforms()

	for sig_type in sig_types:
		binaries.prepare_snr(sig_type)
		for sc in sensitivity_dict:
				snr[sc + '_' + sig_type] = binaries.find_snr(sensitivity_dict[sc])

	print(num_proc, 'end')
	return snr

def hchar_try(m1,m2,redshift, s1, s2,st,et,waveform_type):
	try:
		f_obs, hc = fdw.hchar_func(m1,m2,redshift, s1, s2,st, et,waveform_type)
		return f_obs, hc
	except RuntimeError:
		return [],[]
	except IndexError:
		return [],[]


class CalculateSignalClass:
	def __init__(self, pid, M, q, z, s1, s2, start_time, end_time, base_parameters, extra_dict={}):
		self.pid = pid
		self.M, self.q, self.z, self.s1, self.s2 = M, q, z, s1, s2
		self.start_time, self.end_time = start_time, end_time
		self.extra_dict = extra_dict
		self.base_parameters = base_parameters

		#parameters of second binary to scale against base
		self.m1 = self.M*self.q/(1.0+self.q)
		self.m2 = self.M/(1.0+self.q)

		if 'averaging_factor' in extra_dict.keys():
			self.averaging_factor = extra_dict['averaging_factor']

		else:
			self.averaging_factor = 1.0

		if 'snr_factor' in extra_dict.keys():
			self.snr_factor = extra_dict['snr_factor']
		else:
			self.snr_factor = 1.0

		self.M_b, self.z_b, self.start_time_b, self.end_time_b, self.waveform_type, self.generate_function = base_parameters

	def create_waveforms(self):
		q_arr, s1_arr, s2_arr = np.meshgrid(np.unique(self.q), np.unique(self.s1), np.unique(self.s2))
		q_arr, s1_arr, s2_arr = q_arr.ravel(), s1_arr.ravel(), s2_arr.ravel()

		self.freq_len = 10000
		if 'freq_length' in self.extra_dict.keys():
			self.freq_len = int(self.extra_dict['freq_length'])

		self.base_waveforms_f_obs = np.zeros((len(self.M), self.freq_len))
		self.base_waveforms_hc_obs = np.zeros((len(self.M), self.freq_len))
		if 'mrg' in self.extra_dict['signal_types'] or 'ins' in self.extra_dict['signal_types'] or 'rd' in self.extra_dict['signal_types']:
			self.f_mrg = np.zeros(len(self.M))

		if 'rd' in self.extra_dict['signal_types'] or 'mrg' in self.extra_dict['signal_types']:
			self.f_rd = np.zeros(len(self.M))

		for mr, spin1, spin2 in np.array([q_arr, s1_arr, s2_arr]).T:
			self.fast_generate(mr, spin1, spin2)

		self.base_waveforms_hc_obs = self.cut_waveforms(freq_min=self.find_f_start())
		return

	def fast_generate(self, mr, spin1, spin2):
		inds = np.where((self.q == mr) & (self.s1 == spin1) & (self.s2 == spin2))
		trans_wf = BaseWaveform(self.pid, self.generate_function, mr, spin1, spin2, self.M_b, self.z_b, self.start_time_b, self.end_time_b, self.waveform_type)
		f_s2 = np.squeeze(trans_wf.Mf/self.M[inds, np.newaxis])
		
		self.base_waveforms_f_obs[inds] = np.squeeze(f_s2/(1.0 + self.z[inds, np.newaxis]))

		keep_hc = self.base_waveforms_f_obs[inds]

		#scale hc
		self.base_waveforms_hc_obs[inds] = self.averaging_factor*np.squeeze(trans_wf.hc_obs*(cosmo.luminosity_distance(self.z_b).value/cosmo.luminosity_distance(self.z[inds,np.newaxis]).value)*((1+self.z[inds,np.newaxis])/(1+self.z_b))*(self.M[inds,np.newaxis]/self.M_b)**(5./6.)*(trans_wf.f_s1/f_s2)**(1./6.))

		if 'mrg' in self.extra_dict['signal_types'] or 'ins' in self.extra_dict['signal_types'] or 'rd' in self.extra_dict['signal_types']:
			self.f_mrg[inds] = trans_wf.find_merger_frequency()/(self.M[inds] * (1.0 + self.z[inds]))

		if 'mrg' in self.extra_dict['signal_types'] or 'rd' in self.extra_dict['signal_types']:
			self.f_rd[inds] = trans_wf.find_ringdown_frequency()/(self.M[inds] * (1.0 + self.z[inds]))

		return

	def cut_waveforms(self, freq_min=[], freq_max=[]):
		factor = 1e-50

		if len(freq_max) == 0 and len(freq_min) != 0:
			hc_obs_trans = self.base_waveforms_hc_obs * (self.base_waveforms_f_obs >= freq_min[:,np.newaxis]) + factor*(self.base_waveforms_f_obs < freq_min[:,np.newaxis])

		elif len(freq_min) == 0 and len(freq_max) != 0:
			hc_obs_trans = self.base_waveforms_hc_obs * (self.base_waveforms_f_obs <=  freq_max[:,np.newaxis]) + factor*(self.base_waveforms_f_obs > freq_max[:,np.newaxis])

		else:
			hc_obs_trans = self.base_waveforms_hc_obs * ((self.base_waveforms_f_obs >= freq_min[:,np.newaxis]) & (self.base_waveforms_f_obs <= freq_max[:,np.newaxis])) + factor*((self.base_waveforms_f_obs < freq_min[:,np.newaxis]) & (self.base_waveforms_f_obs > freq_max[:,np.newaxis]))
		return hc_obs_trans


	def find_f_start(self):
		#FH formalism extended to unequal mass
		"""
		start_time = self.start_time/(1.+self.z)*ct.Julian_year
		flow = (start_time*(256.0/5.0)*(ct.G**(5/3)/ct.c**5)*(self.m1*self.m2*Msun)/(Msun*self.m1+self.m2))**(1/3)*(2.0*ct.pi)**(8/3) + f_merg**(-8/3))**(-3/8)
		"""
	
		N = self.m1*self.m2/(self.m1+self.m2)**2.
		tau = N*(self.start_time*ct.c*ct.Julian_year)/(5.*(self.m1+self.m2)*ct.G*Msun/(ct.c**2.))
		flow = 1./(8.*ct.pi*(self.m1+self.m2)*ct.G*Msun/(ct.c**2.)*tau**(3./8.))*(1.+((11./32)*N+743./2688.)*tau**(-1./4.))*ct.c
		return flow/(1+self.z)
		
	def prepare_snr(self, sig_type):
		self.f_in = self.base_waveforms_f_obs
		if sig_type == 'all':
			self.hc_in = self.base_waveforms_hc_obs

		elif sig_type == 'ins':

			self.hc_in = self.cut_waveforms(freq_max = self.f_mrg)

		elif sig_type == 'mrg':
			self.hc_in = self.cut_waveforms(freq_min = self.f_mrg, freq_max = self.f_rd)

		elif sig_type == 'rd':
			self.hc_in = self.cut_waveforms(freq_min = self.f_rd)

			#take first derivative and interpolate to find where first derivative is zero
			#this will represent the local maximum in the signal


			#if the first derivative never equals zero, find second derivative and interpolate
			#find where second derivative equals zero representing inflection point of signal from positive 2nd derivative to negative where the merger ends and ringdown begins

		return

	def find_snr(self, sensitivity_function):
		snr_integrand = 1.0/self.f_in * (self.hc_in/sensitivity_function(self.f_in))**2
		"""
		num_checks = 10
		
		out_check = []
		for i in np.arange(num_checks):
			st = time.time()
			snr_trapz = np.sqrt(np.trapz(snr_integrand, f_in, axis=1))*self.snr_factor
			out_check.append(time.time()-st)

		print('trapz', "max:", np.max(out_check), "min:", np.min(out_check), "mean:", np.mean(out_check))

		out_check = []
		for i in np.arange(num_checks):
			st = time.time()
			snr_simps = np.sqrt(integrate.simps(snr_integrand, f_in, axis=1))*self.snr_factor
			out_check.append(time.time()-st)

		print('simps', "max:", np.max(out_check), "min:", np.min(out_check), "mean:", np.mean(out_check))

		
		snr_quad = []
		for i in np.arange(len(f_in)):
			snr_integrand = interp1d(f_in[i], 1.0/f_in[i] * (hc_in[i]/sensitivity_function(f_in[i]))**2, bounds_error = False, fill_value=1e-30)
			snr_quad.append(np.sqrt(integrate.quad(snr_integrand, f_in[i][0], f_in[i][-1])[0])*self.snr_factor)
			print(i)

		snr_quad = np.asarray(snr_quad)
		
		pdb.set_trace()
		"""

		return np.sqrt(np.trapz(snr_integrand, self.f_in, axis=1))*self.snr_factor


class BaseWaveform:
	def __init__(self, pid, generate_function, q, s1, s2, M_b, z_b, start_time_b, end_time_b, waveform_type):
		self.pid = pid
		self.q, self.s1, self.s2 = q, s1, s2
		self.M_b, self.z_b, self.start_time_b, self.end_time_b, self.waveform_type = M_b, z_b, start_time_b, end_time_b, waveform_type

		#define own frequency array (allows you to control length)
		freq_len = 10000
		if 'freq_length' in self.pid['generate_info'].keys():
			freq_len = int(self.pid['generate_info']['freq_length'])

		getattr(self, generate_function)()
		#create interpolation function to define frequency points

		find_hc_obs = interp1d(self.f_obs, self.hc_obs, bounds_error = False, fill_value = 'extrapolate')
		self.f_obs = np.logspace(np.log10(self.f_obs[0]), np.log10(self.f_obs[-1]), freq_len)
		self.hc_obs = find_hc_obs(self.f_obs)

		#establish source frame frequency for scaling laws
		self.f_s1 = self.f_obs*(1+self.z_b)

		#establish dimensionless quantity of Mf to adjust frequencies from base
		self.Mf = self.f_s1*self.M_b

	def lal_sim_waveform_generate(self):
		m1_b =  self.M_b*self.q/(1.0+self.q)
		m2_b = self.M_b/(1.0+self.q)
		self.f_obs, self.hc_obs = hchar_try(m1_b, m2_b, self.z_b, self.s1, self.s2, self.start_time_b, self.end_time_b, self.waveform_type)
		return	

	def file_read_in_for_base(self):
		data = ascii.read(self.pid['input_info']['input_location'] + '/' + self.gid['waveform_generator'] + 'q_%.4f_s1_%.4f_s2_%.4f.txt'%(self.q, self.s1, self.s2))
		self.f_obs = data['f']
		self.hc_obs = data['hc']
		return

	def find_merger_frequency(self):
		#Flanagan and Hughes
		#return 0.018*ct.c**3/(ct.G*(m1+m2)*Msun) #hz
		f_mrg = 1.0/(6.0**(3./2.)*ct.pi*(self.M_b*Msun*ct.G/ct.c**2)*(1+self.z_b))*ct.c
		self.Mf_mrg = self.M_b*f_mrg*(1+self.z_b)
		return self.Mf_mrg

		 

	def find_ringdown_frequency(self):
		check = argrelextrema(self.hc_obs, np.greater)[0]

		if len(check) == 0:
			func = interp1d(self.Mf, self.hc_obs, bounds_error=False, fill_value='extrapolate')
			#deriv = np.asarray([derivative(func, f, dx = 1e-8) for f in self.Mf[1:-1]])
			deriv = np.gradient(self.hc_obs, self.f_obs)
			check = argrelextrema(deriv, np.greater)[0]
			check = check[check>np.where(self.Mf>=self.Mf_mrg)[0][0]]
		try:
			self.Mf[check[0]]
		except IndexError:
			pdb.set_trace()
		return self.Mf[check[0]]

class file_read_out:
	def __init__(self, pid, file_type, output_string, xvals, yvals, output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name):
		self.pid = pid
		self.file_type, self.output_string, self.xvals, self.yvals, self.output_dict, self.num_x, self.num_y, self.xval_name, self.yval_name, self.par_1_name, self.par_2_name, self.par_3_name = file_type, output_string,  xvals, yvals,output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name

	def prep_output(self):
		self.units_dict = {}
		for key in self.pid['generate_info'].keys():
			if key[-4::] == 'unit':
				self.units_dict[key] = self.pid['generate_info'][key]

		self.added_note = ''
		if 'added_note' in self.pid['output_info'].keys():
			self.added_note = self.pid['output_info']['added_note']
		return

	def hdf5_read_out(self):
		with h5py.File(WORKING_DIRECTORY + '/' + self.output_string + '.' + self.file_type, 'w') as f:

			header = f.create_group('header')
			header.attrs['Title'] = 'Generated SNR Out'
			header.attrs['Author'] = 'Generator by: Michael Katz'
			header.attrs['Date/Time'] = str(datetime.datetime.now())

			header.attrs['xval_name'] = self.xval_name
			header.attrs['num_x_pts'] = self.num_x
			header.attrs['xval_unit'] = self.units_dict['xval_unit']

			header.attrs['yval_name'] = self.yval_name
			header.attrs['num_y_pts'] = self.num_y
			header.attrs['yval_unit'] = self.units_dict['yval_unit']

			header.attrs['par_1_name'] = self.par_1_name
			header.attrs['par_1_unit'] = self.units_dict['par_1_unit']

			header.attrs['par_2_name'] = self.par_2_name
			header.attrs['par_2_unit'] = self.units_dict['par_2_unit']

			header.attrs['par_3_name'] = self.par_3_name
			header.attrs['par_3_unit'] = self.units_dict['par_3_unit']

			if self.added_note != '':
				header.attrs['Added note'] = self.added_note

			data = f.create_group('data')

			x_col_name = self.pid['generate_info']['xval_name']
			if 'x_col_name' in self.pid['output_info'].keys():
				x_col_name = self.pid['output_info']['x_col_name']

			dset = data.create_dataset(x_col_name, data = self.xvals, dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)

			y_col_name = self.pid['generate_info']['yval_name']
			if 'y_col_name' in self.pid['output_info'].keys():
				y_col_name = self.pid['output_info']['y_col_name']

			dset = data.create_dataset(y_col_name, data = self.yvals, dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)


			for key in self.output_dict.keys():
				dset = data.create_dataset(key, data = self.output_dict[key], dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)

	def txt_read_out(self):
		header = '#Generated SNR Out\n'
		header += '#Generator by: Michael Katz\n'
		header += '#Date/Time: %s\n'%str(datetime.datetime.now())

		header += '#xval_name: %s\n'%self.xval_name
		header += '#num_x_pts: %i\n'%self.num_x
		header += '#xval_unit: %s\n'%self.units_dict['xval_unit']
		
		header += '#yval_name: %s\n'%self.yval_name
		header += '#num_y_pts: %i\n'%self.num_y
		header += '#yval_unit: %s\n'%self.units_dict['yval_unit']

		header += '#par_1_name: %s\n'%self.par_1_name
		header += '#par_1_unit: %s\n'%self.units_dict['par_1_unit']
		header += '#par_2_name: %s\n'%self.par_2_name
		header += '#par_2_unit: %s\n'%self.units_dict['par_2_unit']
		header += '#par_3_name: %s\n'%self.par_3_name
		header += '#par_3_unit: %s\n'%self.units_dict['par_3_unit']

		if self.added_note != '':
			header+= '#Added note: ' + self.added_note + '\n'
		else:
			header+= '#Added note: None\n'

		header += '#--------------------\n'

		x_col_name = self.pid['generate_info']['xval_name']
		if 'x_col_name' in self.pid['output_info'].keys():
			x_col_name = self.pid['output_info']['x_col_name']

		header += x_col_name + '\t'

		y_col_name = self.pid['generate_info']['yval_name']
		if 'y_col_name' in self.pid['output_info'].keys():
			y_col_name = self.pid['output_info']['y_col_name']

		header += y_col_name + '\t'

		for key in self.output_dict.keys():
			header += key + '\t'

		x_and_y = np.asarray([self.xvals, self.yvals])
		snr_out = np.asarray([self.output_dict[key] for key in self.output_dict.keys()]).T

		data_out = np.concatenate([x_and_y.T, snr_out], axis=1)

		np.savetxt(WORKING_DIRECTORY + '/' + self.output_string + '.' + self.file_type, data_out, delimiter = '\t',header = header, comments='')
		return

		
class main_process:
	def __init__(self, pid):
		self.pid = pid

		self.gid = pid['generate_info']

		self.extra_dict = {}
		#Galactic Background Noise --> need f and hn
	
		if pid['general']['add_wd_noise'] == 'True' or pid['general']['add_wd_noise'] == 'Both':
			self.read_in_wd_noise()

		#Sensitivity curve files
		self.sensecurves = pid['input_info']['sensitivity_curves']

		self.read_in_sensitivity_curves()

		self.waveform_type = ''
		if 'waveform_type' in self.gid.keys():
			self.waveform_type = self.gid['waveform_type']

	def read_in_noise_file(self, file_dict, wd_noise=False):
		data = ascii.read(self.pid['input_info']['input_location'] + '/' + file_dict['name'])

		if wd_noise == False:
			self.labels.append(file_dict['name'][0:-4])
	
		f_col_name = 'f'
		if 'freq_column_label' in self.pid['input_info'].keys():
			f_col_name = self.pid['input_info']['freq_column_label']
		if 'freq_column_label' in file_dict.keys():
			f_col_name = file_dict['freq_column_label']

		amp_col_name = 'Sn'
		if 'amplitude_column_label' in self.pid['input_info'].keys():
			amp_col_name = self.pid['input_info']['amplitude_column_label']
		if 'amplitude_column_label' in file_dict.keys():
			amp_col_name = file_dict['amplitude_column_label']

		f		 = np.asarray(data[f_col_name])
		#convert from SA PSD to NSA characteristic strain in noise
		amp		 = np.asarray(data[amp_col_name])

		if file_dict['type'] == 'PSD':
			amp = np.sqrt(amp)

		if file_dict['type'] == 'PSD' or file_dict['type'] == 'ASD':
			amp = amp*np.sqrt(f)


		averaging_factor = np.sqrt(3./20.)
		if 'sensitivity_averaging_factor' in self.pid['input_info'].keys():
			averaging_factor = self.pid['input_info']['sensitivity_averaging_factor']
		if 'sensitivity_averaging_factor' in file_dict.keys():
			averaging_factor = file_dict['sensitivity_averaging_factor']

		hn = amp*averaging_factor
		
		return f, hn

	def read_in_sensitivity_curves(self):
		#declare dict for noise curve functions
		self.sensitivity_dict = OrderedDict()
		self.labels = []

		#read in Sensitvity data
		for i, file_dict in enumerate(self.sensecurves):
			f, hn = self.read_in_noise_file(file_dict, wd_noise=False)

			#place interpolated functions into dict with second including WD
			if self.pid['general']['add_wd_noise'] == 'True' or self.pid['general']['add_wd_noise'] == 'Both' or self.pid['general']['add_wd_noise'] == 'both':
				try:
					self.wd_noise
				except AttributeError:
					f_wd, hn_wd = self.read_in_noise_file(self.pid['input_info']['Galactic_background'], wd_noise=True)
					self.wd_noise = interp1d(f_wd, hn_wd, bounds_error=False, fill_value=1e-30)

				wd_up = (hn/self.wd_noise(f) <= 1.0)
				wd_down = (hn/self.wd_noise(f) > 1.0)

				self.sensitivity_dict[self.labels[i] + '_wd'] = interp1d(f, hn*wd_down+self.wd_noise(f)*wd_up, bounds_error=False, fill_value=1e30)

				if self.pid['general']['add_wd_noise'] == 'Both' or self.pid['general']['add_wd_noise'] == 'both':
					self.sensitivity_dict[self.labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

			else:
				self.sensitivity_dict[self.labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

		return

	def set_parameters(self):
		#dimensions of generation
		self.num_x = int(self.gid['num_x'])
		self.num_y = int(self.gid['num_y'])

		#declare 1D arrays of both paramters
		if self.gid['xscale'] != 'lin':
			self.xvals = np.logspace(np.log10(float(self.gid['x_low'])),np.log10(float(self.gid['x_high'])), self.num_x)

		else:
			self.xvals = np.linspace(float(self.gid['x_low']),float(self.gid['x_high']), self.num_x)

		if self.gid['yscale'] != 'lin':
			self.yvals = np.logspace(np.log10(float(self.gid['y_low'])),np.log10(float(self.gid['y_high'])), self.num_y)

		else:
			self.yvals = np.logspace(float(self.gid['y_low']),float(self.gid['y_high']), self.num_y)
		#Additional Parameters

		#mass ratio entry (b is for base, but mass ratio should not change becuase it changes the nature of the waveform)
		par_1 = float(self.gid['fixed_parameter_1'])
		par_2 = float(self.gid['fixed_parameter_2'])
		par_3 = float(self.gid['fixed_parameter_3'])

		self.xvals, self.yvals, par_1, par_2, par_3 = np.meshgrid(self.xvals, self.yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]))

		self.xvals, self.yvals, par_1, par_2, par_3 = self.xvals.ravel(),self.yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel()

		self.input_dict = {self.gid['xval_name']:self.xvals, self.gid['yval_name']:self.yvals, self.gid['par_1_name']:par_1, self.gid['par_2_name']:par_2, self.gid['par_3_name']:par_3, 'start_time': float(self.gid['start_time']), 'end_time':float(self.gid['end_time'])}

		self.define_base_parameters_for_fast_generate()
		return

	def define_base_parameters_for_fast_generate(self):
		M_b = float(self.gid['generation_base_parameters']['fast_generate_Mbase'])
		z_b = float(self.gid['generation_base_parameters']['fast_generate_zbase'])
		start_time_b = float(self.gid['generation_base_parameters']['fast_generate_stbase'])
		end_time_b = float(self.gid['generation_base_parameters']['fast_generate_etbase'])

		waveform_type = ''
		if 'waveform_type' in self.gid.keys():
			waveform_type = self.gid['waveform_type']

		#generate with lalsim
		if self.pid['generate_info']['waveform_generator'] == 'lalsimulation':
			generate_function = 'lal_sim_waveform_generate'

		#premade waveform
		else:
			generate_function =  'file_read_in_for_base'

		self.base_parameters = (M_b, z_b, start_time_b, end_time_b, waveform_type, generate_function)
		
		return


	def add_extras(self):
		if 'snr_calculation_factors' in self.gid.keys():
			if 'averaging_factor' in self.gid['snr_calculation_factors'].keys():
				self.extra_dict['averaging_factor'] = float(self.gid['snr_calculation_factors']['averaging_factor'])

			if 'snr_factor' in self.gid['snr_calculation_factors'].keys():
				self.extra_dict['snr_factor'] = float(self.gid['snr_calculation_factors']['snr_factor'])

		if 'freq_length' in self.gid.keys():
			self.extra_dict['freq_length'] = self.gid['freq_length']

		self.extra_dict['signal_types'] = self.pid['general']['signal_type']

		return


	def prep_parallel(self):
		st = time.time()
		num_processors = 4
		num_splits = 100

		num_splits = int(self.pid['general']['num_splits'])
		self.num_processors = int(self.pid['general']['num_processors'])

		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		split_val = int(np.ceil(len(self.xvals)/num_splits))

		split_inds = [num_splits*i for i in np.arange(1,split_val)]
		array_inds = np.arange(len(self.xvals))
		find_split = np.split(array_inds,split_inds)

		#start time ticker

		self.args = []
		for i, find_part in enumerate(find_split):
			binaries_class = CalculateSignalClass(self.pid, self.input_dict['total_mass'][find_part], self.input_dict['mass_ratio'][find_part], self.input_dict['redshift'][find_part], self.input_dict['spin_1'][find_part], self.input_dict['spin_2'][find_part],  self.input_dict['start_time'], self.input_dict['end_time'], self.base_parameters, self.extra_dict)
			self.args.append((i, binaries_class,  self.pid['general']['signal_type'], self.sensitivity_dict))
		return

	def run_parallel(self):
		results = []
		with Pool(self.num_processors) as pool:
			print('start pool\n')
			results = [pool.apply_async(parallel_func, arg) for arg in self.args]

			out = [r.get() for r in results]

		self.final_dict = OrderedDict()
		for sc in self.sensitivity_dict.keys():
			for sig_type in self.pid['general']['signal_type']:
				self.final_dict[sc + '_' + sig_type] = np.concatenate([r[sc + '_' + sig_type] for r in out])

		return

	def run_single(self):
		binaries_class = CalculateSignalClass(self.pid, self.input_dict['total_mass'], self.input_dict['mass_ratio'], self.input_dict['redshift'], self.input_dict['spin_1'], self.input_dict['spin_2'],  self.input_dict['start_time'], self.input_dict['end_time'], self.base_parameters, self.extra_dict)
		self.final_dict = parallel_func(0, binaries_class,  self.pid['general']['signal_type'], self.sensitivity_dict)
		return


def generate_contour_data(pid):
	#choose all phases or entire signal
	#ADD FILES LIKE IN MAKE PLOT WITH SEPARATE DICTS

	begin_time = time.time()
	global WORKING_DIRECTORY

	gid = pid['generate_info']

	WORKING_DIRECTORY = pid['general']['WORKING_DIRECTORY']

	running_process = main_process(pid)
	running_process.set_parameters()
	running_process.add_extras()

	if pid['general']['generation_type'] == 'parallel':
		running_process.prep_parallel()
		running_process.run_parallel()

	else:
		running_process.run_single()

	file_out = file_read_out(pid, pid['output_info']['output_file_type'], pid['output_info']['output_file_name'],  running_process.xvals, running_process.yvals, running_process.final_dict, running_process.num_x, running_process.num_y, gid['xval_name'], gid['yval_name'], gid['par_1_name'], gid['par_2_name'], gid['par_3_name'])

	file_out.prep_output()
	print('outputing file')
	getattr(file_out, pid['output_info']['output_file_type'] + '_read_out')()

	#create header for data_file


	print(time.time()-begin_time)

if __name__ == '__main__':
	plot_info_dict = json.load(open(sys.argv[1], 'r'),
		object_pairs_hook=OrderedDict)

	generate_contour_data(plot_info_dict)
				
