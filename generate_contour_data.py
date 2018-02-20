import numpy as np
import scipy.constants as ct
import pdb
import scipy.constants as ct
import datetime
import time
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
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
	"""
	if len(inds)!=0:
		ind_start = inds[0]
		hc_int = interp1d(f_changed, hc_changed, bounds_error = False, fill_value = 1e-30)

		#compensate if ringdown starts before the starting point (which is wrong) 
		if phases == True:
			if ind_rd_start-1 < ind_start:
				j = 0
				for k in labels:
					for nc in np.arange(num_cols):
						SNR[i][j] = 1e-30
						j+=1
				i+=1
				print(i)
				continue

	"""

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
	def __init__(self, file_type, output_string, output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name, units_dict={}, added_note=''):
		self.file_type, self.output_string, self.output_dict, self.num_x, self.num_y, self.xval_name, self.yval_name, self.par_1_name, self.par_2_name, self.par_3_name, self.units_dict, self.added_note = file_type, output_string, output_dict, num_x, num_y, xval_name, yval_name, par_1_name, par_2_name, par_3_name, units_dict, added_note

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

def generate_contour_data(pid):
	#choose all phases or entire signal

	global WORKING_DIRECTORY

	if pid['generation_type'][0] == 'parallel':
		from multiprocessing import Pool

	t_or_f_dict = {'True': True, 'False':False}

	WORKING_DIRECTORY = pid['WORKING_DIRECTORY'][0]

	#string for output data file
	out_string = pid['generation_output_string'][0]

	#Galactic Background Noise --> need f and hn
	if pid['add_wd_noise'][0] == 'True' or pid['add_wd_noise'][0] == 'Both':
		data = np.genfromtxt(WORKING_DIRECTORY + '/' + pid['Galactic_background_file'][0], names = True)

		f_wd = data['f']
		hn_wd =  data['Sn']*np.sqrt(data['f'])*np.sqrt(3./20.)

		wd_noise = interp1d(f_wd, hn_wd, bounds_error=False, fill_value=1e-30)

	#Sensitivity curve files
	Sensecurves = pid['Sensitivity_curves']

	#Sensitivity curve files labels
	labels = [sc[0:-4] for sc in Sensecurves]
	
	#declare dict for noise curve functions
	sensitivity_dict = OrderedDict()

	#read in Sensitvity data
	for i, k in enumerate(Sensecurves):
		data = np.genfromtxt(WORKING_DIRECTORY + '/' + k, names=True)	
		
		f		 = np.asarray(data['f'])
		#convert from SA PSD to NSA characteristic strain in noise
		hn		 = np.asarray(data['Sn'])*np.sqrt(3./20.)*np.sqrt(f)
	
		#place interpolated functions into dict with second including WD
		if pid['add_wd_noise'][0] == 'True':
			wd_up = (wd_noise(f)/hn)
			wd_down = (hn/wd_noise(f))
			sensitivity_dict[labels[i] + '_wd'] = interp1d(f, hn*wd_down+wd_noise(f)*wd_up, bounds_error=False, fill_value=1e30)

		elif pid['add_wd_noise'][0] == 'Both':
			wd_up = (wd_noise(f)/hn)
			wd_down = (hn/wd_noise(f))

			sensitivity_dict[labels[i] + '_wd'] = interp1d(f, hn*wd_down+wd_noise(f)*wd_up, bounds_error=False, fill_value=1e30)
			sensitivity_dict[labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

		else:
			sensitivity_dict[labels[i]] = interp1d(f, hn, bounds_error=False, fill_value=1e30)

	#dimensions of generation
	num_x = int(pid['num_x'][0])
	num_y = int(pid['num_y'][0])

	#declare 1D arrays of both paramters
	xvals = np.logspace(np.log10(float(pid['x_low'][0])),np.log10(float(pid['x_high'][0])), num_x)
	yvals = np.logspace(np.log10(float(pid['y_low'][0])),np.log10(float(pid['y_high'][0])), num_y)

	#Additional Parameters

	#mass ratio entry (b is for base, but mass ratio should not change becuase it changes the nature of the waveform)
	par_1 = float(pid['fixed_parameter_1'][0])
	par_2 = float(pid['fixed_parameter_2'][0])
	par_3 = float(pid['fixed_parameter_3'][0])

	xvals, yvals, par_1, par_2, par_3 = np.meshgrid(xvals, yvals, np.array([par_1]), np.array([par_2]), np.array([par_3]))

	xvals, yvals, par_1, par_2, par_3 = xvals.ravel(), yvals.ravel(), par_1.ravel(), par_2.ravel(), par_3.ravel()

	input_dict = {pid['xval_name'][0]:xvals, pid['yval_name'][0]:yvals, pid['par_1_name'][0]:par_1, pid['par_2_name'][0]:par_2, pid['par_3_name'][0]:par_3, 'start_time': float(pid['start_time'][0]), 'end_time':float(pid['end_time'][0])}

	extra_dict={}
	#b -> base --> these values with be used within scaling laws to produce output waveforms
	if pid['hc_generation_type'][0] == 'fast_generate':
		M_b = float(pid['fast_generate_Mbase'][0])
		q_b = float(pid['fast_generate_qbase'][0])
		z_b = float(pid['fast_generate_qbase'][0])
		s1_b = float(pid['fast_generate_s1base'][0])
		s2_b = float(pid['fast_generate_s2base'][0])
		start_time_b = float(pid['fast_generate_stbase'][0])
		end_time_b = float(pid['fast_generate_etbase'][0])
		m1 =  M_b*q_b/(1.0+q_b)
		m2 = M_b/(1.0+q_b)
		waveform_type = pid['waveform_type'][0]


		#generate with lalsim
		if pid['waveform_generator'][0] == 'lalsimulation':
			f_obs, hc_obs = hchar_try(m1,m2,z_b, s1_b, s2_b,start_time_b, end_time_b,waveform_type)

		#premade waveform
		else:
			data = np.genfromtxt(WORKING_DIRECTORY + '/' + pid['waveform_generator'][0], names=True)
			f_obs = data['f']
			hc_obs = data['hc']


		#create interpolation function to define frequency points
		find_hc_obs = interp1d(f_obs,hc_obs, bounds_error = False, fill_value = 'extrapolate')

		#define own frequency array (allows you to control length)
		if 'freq_length' in pid.keys():
			freq_len = int(pid['freq_length'][0])
		else:
			freq_len = 10000

		f_obs = np.logspace(np.log10(f_obs[0]), np.log10(f_obs[-1]), freq_len)
		hc_obs = find_hc_obs(f_obs)

		#establish source frame frequency for scaling laws
		f_s1 = f_obs*(1+z_b)

		#establish dimensionless quantity of Mf to adjust frequencies from base
		Mf = f_s1*M_b

		extra_dict = {'M_b':M_b, 'z_b':z_b, 'f_obs':f_obs, 'hc_obs':hc_obs, 'f_s1':f_s1, 'Mf':Mf}

	extra_dict['averaging_factor'] = float(pid['averaging_factor'][0])
	extra_dict['snr_factor'] = float(pid['snr_factor'][0])

	waveform_type = ''
	if 'waveform_type' in pid.keys():
		waveform_type = pid['waveform_type'][0]

	find_all = [CalculateSignalClass(input_dict['total_mass'][j], input_dict['mass_ratio'][j], input_dict['redshift'][j], input_dict['spin_1'][j], input_dict['spin_2'][j], input_dict['start_time'], input_dict['end_time'], pid['waveform_type'][0], extra_dict) for j in range(len(xvals))]

	st = time.time()
	if pid['generation_type'][0] == 'parallel':
		num_processors = 4
		num_splits = 100

		num_splits = int(pid['num_splits'][0])
		num_processors = int(pid['num_processors'][0])

		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		split_val = int(np.ceil(len(find_all)/num_splits))

		split_inds = [num_splits*i for i in np.arange(1,split_val)]

		find_split = np.split(find_all,split_inds)

		#start time ticker

		args = []
		for i, find_part in enumerate(find_split):
			args.append((i, find_part, pid['hc_generation_type'][0],  pid['signal_type'], sensitivity_dict))
		
		results = []
		with Pool(num_processors) as pool:
			print('start pool\n')
			results = [pool.apply_async(parallel_func, arg) for arg in args]

			out = [r.get() for r in results]

		final_dict = OrderedDict()
		for sc in sensitivity_dict.keys():
			for sig_type in pid['signal_type']:
				final_dict[sc + '_' + sig_type] = np.concatenate([r[sc + '_' + sig_type] for r in out])

	else:
		final_dict = parallel_func(0, find_all, pid['hc_generation_type'][0], pid['signal_type'], sensitivity_dict)

	trans_dict = OrderedDict()

	trans_dict['x'], trans_dict['y'] = xvals, yvals

	for key in final_dict.keys():
		trans_dict[key] = final_dict[key]
		final_dict[key] = None

	final_dict = trans_dict

	"""
	if phases == True:
		#determine merger frequency in the source frame of the base --> based on 1PN order
		mrgr_frq_s = 1.0/(6.**(3./2.)*ct.pi*(m1+m2)*Msun*ct.G/(ct.c**3.))

		#set inspiral end
		ins_end = np.where(f_s1<=mrgr_frq_s)[0][-1]

		#set start of ringdown
		if 'ringdown_start_freq' in pid.keys():
			ind_rd_start = np.where(f_s1>= float(pid['ringdown_start_freq'][0]))[0][0]
		else:
			for e in np.arange(1,len(f_s1)-1):
				if hc_obs[e-1]< hc_obs[e] and hc_obs[e+1]<hc_obs[e]:
					ind_rd_start = np.where(f_s1 >= f_s1[e])[0][0]
					break
	"""
	#set column number based on phases or not (2 for each w/ and w/o GB)

	#create header for data_file

	units_dict = {}
	for key in pid.keys():
		if key[-4::] == 'unit':
			units_dict[key] = pid[key][0]




	added_note = ''
	if 'added_note' in pid.keys():
		for a_n in pid[added_note]:
			added_note += a_n + ' '
	file_out = file_read_out(pid['output_file_type'][0], pid['output_string'][0], final_dict, num_x, num_y, pid['xval_name'][0], pid['yval_name'][0], pid['par_1_name'][0], pid['par_2_name'][0], pid['par_3_name'][0], units_dict, added_note)

	getattr(file_out, pid['output_file_type'][0] + '_read_out')()


	print(time.time()-st)

if __name__ == '__main__':
	f = open(sys.argv[1], 'r')
	lines = f.readlines()
	lines = [line for line in lines if line[0]!= '#']
	lines = [line for line in lines if line[0]!= '\n']

	plot_info_dict = OrderedDict()
	for line in lines:
		if ':' in line:
			plot_info_dict[line.split()[0][0:-1]] = line.split()[1::]


	generate_contour_data(plot_info_dict)
				
