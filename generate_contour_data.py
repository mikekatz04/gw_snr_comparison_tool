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
from scipy.integrate import quad
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

Msun=1.989e30



def parallel_func(num_proc, binaries, hc_generation_type, sig_types, sensitivity_dict):
	print(num_proc,'start', len(binaries))
	#initialize SNR array
	snr = OrderedDict()
	for sc in sensitivity_dict.keys():
		for sig_type in sig_types:
			snr[sc + '_' + sig_type] = np.zeros(len(binaries))

	i=0
	for binary in binaries:
		getattr(binary, hc_generation_type)()

		for sc in sensitivity_dict:
			for sig_type in sig_types:
				snr[sc + '_' + sig_type][i] = binary.find_snr(sig_type, sensitivity_dict[sc])
		i += 1

		#clear memory
		for key in binary.__dict__.keys():
			binary.__dict__[key] = None

	print(num_proc, 'end')
	return snr

def hchar_try(m1,m2,redshift, s1, s2,st,et,waveform_type):
	try:
		f_obs, hc = fdw.hchar_func(m1,m2,redshift, s1, s2,st, et,waveform_type)
		return f_obs, hc
	except RuntimeError:
		return [],[]


class CalculateSignalClass:
	def __init__(self, M, q, z, s1, s2, start_time, end_time, waveform_type= '', extra_dict={}):
		self.M, self.q, self.z, self.s1, self.s2 = M, q, z, s1, s2
		self.start_time, self.end_time = start_time, end_time
		self.waveform_type = waveform_type
		self.extra_dict = extra_dict

		#parameters of second binary to scale against base
		self.m1 = self.M*self.q/(1.0+self.q)
		self.m2 = self.M/(1.0+self.q)

		if 'M_b' in self.extra_dict.keys():
			self.M_b = self.extra_dict['M_b']
			self.z_b = self.extra_dict['z_b']
			self.f_obs, self.hc_obs, self.f_s1, self.Mf = self.extra_dict['f_obs'], self.extra_dict['hc_obs'], self.extra_dict['f_s1'],self.extra_dict['Mf']

		if 'averaging_factor' in extra_dict.keys():
			self.averaging_factor = extra_dict['averaging_factor']

		else:
			self.averaging_factor = 1.0

		if 'snr_factor' in extra_dict.keys():
			self.snr_factor = extra_dict['snr_factor']
		else:
			self.snr_factor = 1.0



	def fast_generate(self):
		#scale frequencies
		f_s2 = self.Mf/self.M
		f_changed = f_s2/(1+self.z)

		f_start = self.find_f_start()

		inds = np.where(f_changed >= f_start)[0]

		if len(inds) <2:
			self.f_all, self.hc_all =  [],[]
			return

		#scale hc
		hc_changed = self.hc_obs*(cosmo.luminosity_distance(self.z_b).value/cosmo.luminosity_distance(self.z).value)*((1+self.z)/(1+self.z_b))*(self.M/self.M_b)**(5./6.)*(self.f_s1/f_s2)**(1./6.)


		#Post newtonian expansion to 1st order from SNR calibration docs
		

		self.f_all, self.hc_all =  f_changed[inds], hc_changed[inds]*self.averaging_factor

		return

	def lal_generate(self):
		self.f_all, self.hc_all = hchar_try(self.m1,self.m2,self.z, self.s1, self.s2, self.start_time, self.end_time,self.waveform_type)
		return

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

	def find_merger_frequency(self):
		#Flanagan and Hughes
		#return 0.018*ct.c**3/(ct.G*(m1+m2)*Msun) #hz
		self.f_mrg = 1.0/(6.0**(3./2.)*ct.pi*(self.M*Msun*ct.G/ct.c**2)*(1+self.z))*ct.c
		return 

	def find_ringdown_frequency(self):
		check = argrelextrema(self.hc_all, np.greater)[0]

		try:
			test = self.f_mrg

		except:
			self.f_mrg = self.find_merger_frequency()	

		if len(check) == 0:
			func = interp1d(self.f_all, self.hc_all, bounds_error=False, fill_value='extrapolate')
			deriv = np.asarray([derivative(func, f, dx = 1e-5) for f in self.f_all[1:-1]])
			check = argrelextrema(deriv, np.greater)[0]
			check = check[check>np.where(self.f_all>=self.f_mrg)[0][0]]

		
		self.f_rd = self.f_all[check[0]]
		return

	def find_snr(self, sig_type, sensitivity_function):
		if self.f_all == []:
			return 1e-30

		if sig_type == 'all':
			f_in, hc_in = self.f_all, self.hc_all

		elif sig_type == 'ins':
			try:
				test = self.f_mrg

			except AttributeError:
				self.find_merger_frequency()
			
			f_in, hc_in = self.f_all[self.f_all<self.f_mrg], self.hc_all[self.f_all<self.f_mrg]

		elif sig_type == 'mrg':
			try:
				test = self.f_mrg

			except AttributeError:
				self.find_merger_frequency()	

			try:
				test = self.f_rd

			except AttributeError:
				self.find_ringdown_frequency()	

			inds = np.where((self.f_all>=self.f_mrg) & (self.f_all<=self.f_rd))[0]

			f_in, hc_in = self.f_all[inds], self.hc_all[inds]

		elif sig_type == 'rd':
			try:
				test = self.f_rd

			except AttributeError:
				self.find_ringdown_frequency()	

			self.find_ringdown_frequency()
			f_in, hc_in = self.f_all[self.f_all>self.f_rd], self.hc_all[self.f_all>self.f_rd]

			#take first derivative and interpolate to find where first derivative is zero
			#this will represent the local maximum in the signal


			#if the first derivative never equals zero, find second derivative and interpolate
			#find where second derivative equals zero representing inflection point of signal from positive 2nd derivative to negative where the merger ends and ringdown begins



		if len(f_in) == 0:
			return 1e-30

		snr_integrand = interp1d(f_in, 1.0/f_in * (hc_in/sensitivity_function(f_in))**2, bounds_error = False, fill_value=1e-30)

		return np.sqrt(quad(snr_integrand, f_in[0], f_in[-1])[0])*self.snr_factor

class file_read_out:
	def __init__(self, pid, file_type, output_string, output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name):
		self.pid = pid
		self.file_type, self.output_string, self.output_dict, self.num_x, self.num_y, self.xval_name, self.yval_name, self.par_1_name, self.par_2_name, self.par_3_name = file_type, output_string, output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name

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

		for key in self.output_dict.keys():
			header += key + '\t'

		data_out = np.asarray([self.output_dict[key] for key in self.output_dict.keys()]).T

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

	def read_in_wd_noise(self):
		data = ascii.read(self.pid['input_info']['input_location'] + '/' + self.pid['input_info']['Galactic_background_file'])

		f_wd = data['f']
		hn_wd =  data['Sn']*np.sqrt(data['f'])*np.sqrt(3./20.)

		self.wd_noise = interp1d(f_wd, hn_wd, bounds_error=False, fill_value=1e-30)
		return

	def read_in_sensitivity_curves(self):
		#declare dict for noise curve functions
		self.sensitivity_dict = OrderedDict()
		self.labels = []

		#read in Sensitvity data
		for i, file_dict in enumerate(self.sensecurves):
			data = ascii.read(self.pid['input_info']['input_location'] + '/' + file_dict['name'])

			self.labels.append(file_dict['name'][0:-4])
			
			f_col_name = 'f'
			if 'freq_column_label' in self.pid['input_info'].keys():
				f_col_name = self.pid['input_info']['freq_column_label']
			if 'freq_column_label' in file_dict.keys():
				f_col_name = file_dict['freq_column_label']

			amp_col_name = 'f'
			if 'amplitude_column_label' in self.pid['input_info'].keys():
				amp_col_name = self.pid['input_info']['amplitude_column_label']
			if 'amplitude_column_label' in file_dict.keys():
				amp_col_name = file_dict['amplitude_column_label']

			f		 = np.asarray(data[f_col_name])
			#convert from SA PSD to NSA characteristic strain in noise
			amp		 = np.asarray(data[amp_col_name])

			if file_dict['type'] == 'PSD':
				amp = np.sqrt(amp)

			if file_dict['type'] == 'PSD' or file_dict['type'] != 'ASD':
				amp = np.sqrt(f)


			averaging_factor = np.sqrt(3./20.)
			if 'sensitivity_averaging_factor' in self.pid['input_info'].keys():
				averaging_factor = self.pid['input_info']['sensitivity_averaging_factor']
			if 'sensitivity_averaging_factor' in file_dict.keys():
				averaging_factor = file_dict['sensitivity_averaging_factor']

			hn = amp*averaging_factor

			#place interpolated functions into dict with second including WD
			if self.pid['general']['add_wd_noise'] == 'True' or self.pid['general']['add_wd_noise'] == 'Both':
				wd_up = (self.wd_noise(f)/hn >= 1.0)
				wd_down = (hn/self.wd_noise(f) < 1.0)

				self.sensitivity_dict[self.labels[i] + '_wd'] = interp1d(f, hn*wd_down+wd_noise(f)*wd_up, bounds_error=False, fill_value=1e30)

				if self.pid['general']['add_wd_noise'] == 'Both':
					self.sensitivity_dict[self.labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

			else:
				self.sensitivity_dict[self.labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

		return

	def set_parameters(self):
		#dimensions of generation
		self.num_x = int(self.gid['num_x'])
		self.num_y = int(self.gid['num_y'])

		#declare 1D arrays of both paramters
		if self.gid['xscale'] == 'lin':
			self.xvals = np.logspace(np.log10(float(self.gid['x_low'])),np.log10(float(self.gid['x_high'])), self.num_x)

		else:
			self.xvals = np.linspace(float(self.gid['x_low']),float(self.gid['x_high']), self.num_x)

		if self.gid['yscale'] == 'lin':
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
		return


	def read_in_base_waveform(self):

		self.define_base_parameters_for_fast_generate()
		#generate with lalsim
		if gid['waveform_generator'] == 'lalsimulation':
			self.lal_sim_waveform_generate()

		#premade waveform
		else:
			self.file_read_in_for_base()

		#create interpolation function to define frequency points
		find_hc_obs = interp1d(self.f_obs, self.hc_obs, bounds_error = False, fill_value = 'extrapolate')

		#define own frequency array (allows you to control length)
		freq_len = 10000
		if 'freq_length' in self.gid.keys():
			freq_len = int(self.gid['freq_length'])

		self.f_obs = np.logspace(np.log10(f_obs[0]), np.log10(f_obs[-1]), freq_len)
		self.hc_obs = find_hc_obs(f_obs)

		#establish source frame frequency for scaling laws
		self.f_s1 = self.f_obs*(1+self.z_b)

		#establish dimensionless quantity of Mf to adjust frequencies from base
		self.Mf = self.f_s1*self.M_b

		self.extra_dict['M_b'] = self.M_b
		self.extra_dict['z_b'] = self.z_b
		self.extra_dict['f_obs'] = self.f_obs
		self.extra_dict['hc_obs'] = self.hc_obs
		self.extra_dict['f_s1'] = self.f_s1
		self.extra_dict['Mf'] = self.Mf

		return

	def define_base_parameters_for_fast_generate(self):
		self.M_b = float(self.gid['generation_base_parameters']['fast_generate_Mbase'])
		self.q_b = float(self.gid['generation_base_parameters']['fast_generate_qbase'])
		self.z_b = float(self.gid['generation_base_parameters']['fast_generate_qbase'])
		self.s1_b = float(self.gid['generation_base_parameters']['fast_generate_s1base'])
		self.s2_b = float(self.gid['generation_base_parameters']['fast_generate_s2base'])
		self.start_time_b = float(self.gid['generation_base_parameters']['fast_generate_stbase'])
		self.end_time_b = float(self.gid['generation_base_parameters']['fast_generate_etbase'])
		self.m1 =  M_b*q_b/(1.0+q_b)
		self.m2 = M_b/(1.0+q_b)
		self.waveform_type = self.gid['waveform_type']

		return


	def lal_sim_waveform_generate(self):
		self.f_obs, self.hc_obs = hchar_try(self.m1, self.m2, self.z_b, self.s1_b, self.s2_b, self.start_time_b, self.end_time_b, self.waveform_type)
		return

	def file_read_in_for_base(self):
		data = ascii.read(self.pid['input_info']['input_location'] + '/' + self.gid['waveform_generator'])
		self.f_obs = data['f']
		self.hc_obs = data['hc']

		return

	def add_averaging_factors(self):
		if 'snr_calculation_factors' in self.gid.keys():
			if 'averaging_factor' in self.gid['snr_calculation_factors'].keys():
				self.extra_dict['averaging_factor'] = float(self.gid['snr_calculation_factors']['averaging_factor'])

			if 'snr_factor' in self.gid['snr_calculation_factors'].keys():
				self.extra_dict['snr_factor'] = float(self.gid['snr_calculation_factors']['snr_factor'])

		return

	def prep_find_list(self):
		self.find_all = [CalculateSignalClass(self.input_dict['total_mass'][j], self.input_dict['mass_ratio'][j], self.input_dict['redshift'][j], self.input_dict['spin_1'][j], self.input_dict['spin_2'][j], self.input_dict['start_time'], self.input_dict['end_time'], self.waveform_type, self.extra_dict) for j in range(len(self.xvals))]

	def prep_parallel(self):
		st = time.time()
		num_processors = 4
		num_splits = 100

		num_splits = int(self.pid['general']['num_splits'])
		self.num_processors = int(self.pid['general']['num_processors'])

		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		split_val = int(np.ceil(len(self.find_all)/num_splits))

		split_inds = [num_splits*i for i in np.arange(1,split_val)]

		find_split = np.split(self.find_all,split_inds)

		#start time ticker

		self.args = []
		for i, find_part in enumerate(find_split):
			self.args.append((i, find_part, self.gid['hc_generation_type'],  self.pid['general']['signal_type'], self.sensitivity_dict))
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
		self.final_dict = parallel_func(0, self.find_all, self.gid['hc_generation_type'], self.pid['general']['signal_type'], self.sensitivity_dict)
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

	#b -> base --> these values with be used within scaling laws to produce output waveforms
	if gid['hc_generation_type'] == 'fast_generate':
		running_process.read_in_base_waveform()

	running_process.prep_find_list()
	if pid['general']['generation_type'] == 'parallel':
		running_process.prep_parallel()
		running_process.run_parallel()

	else:
		running_process.run_single()

	"""
	trans_dict = OrderedDict()

	trans_dict['x'], trans_dict['y'] = xvals, yvals

	for key in final_dict.keys():
		trans_dict[key] = final_dict[key]
		final_dict[key] = None

	final_dict = trans_dict
	"""
	file_out = file_read_out(pid, pid['output_info']['output_file_type'], pid['output_info']['output_file_name'], running_process.final_dict, running_process.num_x, running_process.num_y, gid['xval_name'], gid['yval_name'], gid['par_1_name'], gid['par_2_name'], gid['par_3_name'])

	file_out.prep_output()

	getattr(file_out, pid['output_info']['output_file_type'] + '_read_out')()

	#create header for data_file


	print(time.time()-begin_time)

if __name__ == '__main__':
	plot_info_dict = json.load(open(sys.argv[1], 'r'),
		object_pairs_hook=OrderedDict)

	generate_contour_data(plot_info_dict)
				
