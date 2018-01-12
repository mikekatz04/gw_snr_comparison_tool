import numpy as np
import scipy.constants as ct
import matplotlib.pyplot as plt
import pdb
import scipy.constants as ct
import time
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
#import lal
#import lalsimulation
from scipy.integrate import quad
#import FD_waveform_working as fdw
import multiprocessing
import sys

from multiprocessing import Pool

import sys
from collections import OrderedDict


def parallel_func(num_proc, binaries, sig_types, wd_add):
	print(num_proc,'start')
	#initialize SNR array
	num_col_dict = {'True': 1, 'False':1, 'Both':2}
	snr = np.zeros((len(binaries),num_col_dict[wd_add]*len(sig_types)))

	pdb.set_trace()
	i = 0
	for binary in enumerate(binaries):
			

		#parameters of second binary to scale against base
		m1 = M*q/(1.0+q)
		m2 = M/(1.0+q)
		Ms2 = M
		z2 =z
		
		#scale frequencies
		f_s2 = Mf/Ms2
		f_changed = f_s2/(1+z2)

		#scale hc
		hc_changed = hc_obs*(cosmo.luminosity_distance(z1).value/cosmo.luminosity_distance(z2).value)*((1+z2)/(1+z1))*(Ms2/Ms1)**(5./6.)*(f_s1/f_s2)**(1./6.)

		#FH formalism extended to unequal mass
		"""
		f_merg = 0.018*ct.c**3/(ct.G*(m1+m2)*Msun) #hz
		flow = (start_time*(256.0/5.0)*(ct.G**(5/3)/ct.c**5)*(m1*m2*Msun)/(Msun*m1+m2))**(1/3)*(2.0*ct.pi)**(8/3) + f_merg**(-8/3))**(-3/8)
		"""
	
		start_time = float(pid['time_before_merger'][0])/(1.+z2)*ct.Julian_year

		#Post newtonian expansion to 1st order from SNR calibration docs
		N = m1*m2/(m1+m2)**2.
		tau = N*(start_time*ct.c)/(5.*(m1+m2)*ct.G*Msun/(ct.c**2.))
		flow = 1./(8.*ct.pi*(m1+m2)*ct.G*Msun/(ct.c**2.)*tau**(3./8.))*(1.+((11./32)*N+743./2688.)*tau**(-1./4.))*ct.c

		f_start = flow/(1+z2)
		inds = np.where(f_changed >= f_start)[0]

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

			#compute SNR - doubled - one for w/ and w/o WD noise
			j = 0
			for k in labels:
				
				#compute function values inside integral
				val = 1./f_changed*(hc_int(f_changed)/interp_funcs_dict[k][0](f_changed))**2.
				#create an interpolated function of val
				val_int = interp1d(f_changed,val)

				#calculate SNR over all frequencies with val interpolated
				SNR2sq,err2 = quad(SNR_calculation, f_changed[ind_start],f_changed[-1],args = (val_int),epsabs=1.49e-08, epsrel=1.49e-08)
				SNR[i][j] = np.sqrt(2*SNR2sq)

				j+=1
						
				#same as above with WD noise
				val = 1./f_changed*(hc_int(f_changed)/interp_funcs_dict[k][1](f_changed))**2.
				val_int_WD = interp1d(f_changed,val)
				SNR2sq,err2 = quad(SNR_calculation, f_changed[ind_start],f_changed[-1],args = (val_int_WD),epsabs=1.49e-08, epsrel=1.49e-08)
				SNR[i][j] = np.sqrt(2*SNR2sq)
				j+=1
		
				if phases == True:
					#Inspiral w/o GB
					SNR2sq,err2 = quad(SNR_calculation, f_changed[ind_start],f_changed[ins_end],args = (val_int),epsabs=1.49e-08, epsrel=1.49e-08)
					SNR[i][j] = np.sqrt(2*SNR2sq)
					j+=1

					#Merger w/o GB
					SNR2sq,err2 = quad(SNR_calculation, f_changed[ins_end+1],f_changed[ind_rd_start-1],args = (val_int),epsabs=1.49e-08, epsrel=1.49e-08)
					SNR[i][j] = np.sqrt(2*SNR2sq)
					j+=1

					#Ringdown w/o GB
					SNR2sq,err2 = quad(SNR_calculation, f_changed[ind_rd_start],f_changed[-1],args = (val_int),epsabs=1.49e-08, epsrel=1.49e-08)
					SNR[i][j] = np.sqrt(2*SNR2sq)
					j+=1

					#Inspiral w/ GB
					SNR2sq,err2 = quad(SNR_calculation, f_changed[ind_start],f_changed[ins_end],args = (val_int_WD),epsabs=1.49e-08, epsrel=1.49e-08)
					SNR[i][j] = np.sqrt(2*SNR2sq)
					j+=1

					#Merger w/ GB
					SNR2sq,err2 = quad(SNR_calculation, f_changed[ins_end+1],f_changed[ind_rd_start-1],args = (val_int_WD),epsabs=1.49e-08, epsrel=1.49e-08)
					SNR[i][j] = np.sqrt(2*SNR2sq)
					j+=1

					#Ringdown w/ GB
					SNR2sq,err2 = quad(SNR_calculation, f_changed[ind_rd_start],f_changed[-1],args = (val_int_WD),epsabs=1.49e-08, epsrel=1.49e-08)
					SNR[i][j] = np.sqrt(2*SNR2sq)
					j+=1

		#length of inds from where statement is zero, so fill in 1e-30 for all values
		else:
			j = 0
			SNR[i][j] = 1e-30

			j+=1

			SNR[i][j] = 1e-30
			j+=1
	
			if phases == True:

				SNR[i][j] = 1e-30
				j+=1

				SNR[i][j] = 1e-30
				j+=1

				SNR[i][j] = 1e-30
				j+=1


				SNR[i][j] = 1e-30
				j+=1

				SNR[i][j] = 1e-30
				j+=1


				SNR[i][j] = 1e-30
				j+=1	

		i+=1
		if i % 100 == 0:
			print(num_proc,i,'out of', len(find))
	print(num_proc,'end')
	return SNR


def SNR_calculation(f,val):
	return val(f)

def hchar_try(m1,m2,redshift, s1, s2,st,et,waveform_type):
	try:
		f_obs, hc = fdw.hchar_func(m1,m2,redshift, s1, s2,st, et,waveform_type)
		return f_obs, hc
	except RuntimeError:
		return [],[]


class CalculateSignalClass:
	def __init__(self, M, q, z, s1, s2, start_time, end_time, sensitivity_curves, white_dwarf_noise, extra_dict={}):
		self.sensitivity_curves, self.white_dwarf_noise = sensitivity_curves, white_dwarf_noise
		self.M, self.q, self.z, self.s1, self.s2 = M, q, z, s1, s2
		self.start_time, self.end_time = start_time, end_time
		self.extra_dict = extra_dict

		self.M_b = extra_dict['M_b']
		self.q_b = extra_dict['q_b']

		self.m1b = self.M_b/(1+self.q_b)
		self.m2b = self.M_b*self.q_b/(1+self.q_b)
		self.z_b = extra_dict['z_b']
		self.f_obs, self.hc_obs, self.f_s1, self.Mf = extra_dict['f_obs'], extra_dict['hc_obs'], extra_dict['f_s1'], extra_dict['Mf']


	def fast_generate_snr(self):

		#parameters of second binary to scale against base

		self.m1 = self.M*self.q/(1.0+self.q)
		self.m2 = self.M/(1.0+self.q)
		
		#scale frequencies
		f_s2 = self.Mf/self.M
		f_changed = f_s2/(1+self.z)

		#scale hc
		hc_changed = self.hc_obs*(cosmo.luminosity_distance(self.z_b).value/cosmo.luminosity_distance(self.z).value)*((1+self.z)/(1+self.z_b))*(self.M/self.M_b)**(5./6.)*(self.f_s1/f_s2)**(1./6.)

		#FH formalism extended to unequal mass
		"""
		f_merg = 0.018*ct.c**3/(ct.G*(m1+m2)*Msun) #hz
		flow = (start_time*(256.0/5.0)*(ct.G**(5/3)/ct.c**5)*(m1*m2*Msun)/(Msun*m1+m2))**(1/3)*(2.0*ct.pi)**(8/3) + f_merg**(-8/3))**(-3/8)
		"""
	
		start_time = float(pid['time_before_merger'][0])/(1.+z2)*ct.Julian_year

		#Post newtonian expansion to 1st order from SNR calibration docs
		f_start = self.find_f_start()


		inds = np.where(f_changed >= f_start)[0]
		

	def find_f_start(self):
		N = self.m1*self.m2/(self.m1+self.m2)**2.
		tau = N*(self.start_time*ct.c)/(5.*(self.m1+self.m2)*ct.G*Msun/(ct.c**2.))
		flow = 1./(8.*ct.pi*(self.m1+self.m2)*ct.G*Msun/(ct.c**2.)*tau**(3./8.))*(1.+((11./32)*N+743./2688.)*tau**(-1./4.))*ct.c
		return flow/(1+self.z)

def generate_contour_data(pid):
	#choose all phases or entire signal

	global WORKING_DIRECTORY

	t_or_f_dict = {'True': True, 'False':False}

	WORKING_DIRECTORY = pid['WORKING_DIRECTORY'][0]

	#string for output data file
	out_string = pid['generation_output_string'][0]

	#Galactic Background Noise --> need f and hn
	data = np.genfromtxt(WORKING_DIRECTORY + '/' + pid['Galactic_background_file'][0], names = True)

	f_WD = data['f']
	hn_WD =  data['Sn']*np.sqrt(data['f'])*np.sqrt(3./20.)

	noise_curve = [f_WD, hn_WD]

	#Sensitivity curve files
	Sensecurves = pid['Sensitivity_curves']

	#Sensitivity curve files labels
	labels = pid['Sensitivity_curve_labels']
	
	#declare dict for noise curve functions
	sensitivity_dict = dict()

	#read in Sensitvity data
	for i, k in enumerate(Sensecurves):
		data = np.genfromtxt(WORKING_DIRECTORY + '/' + k, names=True)	
		
		f		 = np.asarray(data['f'])
		#convert from SA PSD to NSA characteristic strain in noise
		hn		 = np.asarray(data['Sn'])*np.sqrt(3./20.)*np.sqrt(f)
	
		#place interpolated functions into dict with second including WD
		sensitivity_dict[labels[i]] = [f, hn]

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

	input_dict = {pid['xval_name'][0]:xvals, pid['yval_name'][0]:yvals, pid['par_1_name'][0]:par_1, pid['par_2_name'][0]:par_2, pid['par_3_name'][0]:par_3}

	extra_dict={}
	#b -> base --> these values with be used within scaling laws to produce output waveforms
	if 'fast_generate' in pid.keys():
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
			f_obs, hc = hchar_try(m1,m2,z_b, s1_b, s2_b,start_time_b, end_time_b,waveform_type)

		#premade waveform
		else:
			data = np.genfromtxt(WORKING_DIRECTORY + '/' + pid['waveform_generator'][0], names=True)
			f_obs = data['f']
			hc = data['hc']


		#reduce hc by factors associated with LISA (see paper or Average_SNR.pdf from EB NC SB)
		hc_obs = hc*float(pid['lisa_averaging_factor'][0])

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


	find_all = [CalculateSignalClass(input_dict['total_mass'][j], input_dict['mass_ratio'][j], input_dict['redshift'][j], input_dict['spin_1'][j], input_dict['spin_2'][j], sensitivity_dict, noise_curve, extra_dict) for j in range(len(xvals))]

	if pid['generation_type'][0] == 'parallel':
		num_processors = 4
		num_splits = sid['num_splits'][0]
		num_processors = sid['num_processors'][0]

		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		split_val = int(np.ceil(len(find_all)/num_splits))

		split_inds = [num_splits*i for i in np.arange(1,split_val)]

		find_split = np.split(find_all,split_inds)

		#start time ticker
		st = time.time()

		args = []
		for i, find_part in enumerate(find_split):
			args.append((i, find_part, pid['signal_type'], pid['add_wd_noise'][0]))
		
		results = []
		with Pool(num_processors) as pool:
			print('start pool\n')
			results = [pool.apply_async(parallel_func, arg) for arg in args]

			out = [r.get() for r in results]

		out_parsed = np.concatenate([r for r in out])

	else:
		parallel_func(0, find_all, pid['signal_type'], pid['add_wd_noise'][0])

	pdb.set_trace()

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

	#set column number based on phases or not (2 for each w/ and w/o GB)
	if phases == True:
		num_cols = 8
	else:
		num_cols = 2
		
	print(len(find_all))


	sort_inds = np.argsort(np.transpose(out_parsed)[-1])	

	args = []
	for i, bc_split in enumerate(coord_split):
		args.append((i, kernel_in, bc_split, sid))

	print('Total Processes: ',len(args))

	split_val = int(len(find_all)/num_processors)

	split_inds = [split_val*i for i in np.arange(1,num_processors)]

	find_split = np.split(find_all,split_inds)

	with Pool(num_processors) as pool:
		res = pool.starmap(parallel_func,args)	
		SNR = np.concatenate(res)
	#create header for data_file
	header = 'num_M_pts=%i\nnum_z_pts=%i\n'%(num_M,num_z) + 'M_s' + '\t' + 'z' + '\t'
	

	#add header for all SNR columns
	for k in labels:
		header+= k + '_all' + '\t'
		header+= k + '_all_WD' + '\t'
		if phases == True:
			header+= k + '_ins' + '\t'
			header+= k + '_merg' + '\t'
			header+= k + '_rd' + '\t'
			header+= k + '_ins_WD' + '\t'
			header+= k + '_merg_WD' + '\t'
			header+= k + '_rd_WD' + '\t'


	nans = np.transpose(np.where(np.isnan(SNR)==True))
	for i,j in nans:
		if M_T_source.ravel()[i]>1e9:
			SNR[i][j] = 1e-30

	out = np.append(np.transpose(np.array([redshifts.ravel()])),SNR, axis = 1)
	out = np.append(np.transpose(np.array([M_T_source.ravel()])),out, axis = 1)

	np.savetxt(out_string, out, delimiter = '\t',header = header)


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
				
