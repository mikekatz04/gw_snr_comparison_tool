import numpy as np
import scipy.constants as ct
import pdb
import scipy.constants as ct
import time
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
import lal
import lalsimulation
from scipy.integrate import quad


def generate_FD_waveform(m1,m2,z, s1, s2, start_time,end_time, waveform_type='PhenomD'):
	# Generate waveforms
	# This function takes in a bunch of parameters  and generates the frequency domain waveforms
	# Intrinsic parameters:
	# component masses in kg
	m1, m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
	# component spins (Cartesian, normalized to be unitless)
	s1x, s1y, s1z = 0.0, 0.0, s1
	s2x, s2y, s2z = 0.0, 0.0, s2

	dist = cosmo.comoving_distance(z).value
	

	start_time = start_time*ct.Julian_year #seconds
	end_time = end_time*ct.Julian_year #seconds

	f_merg = 1.0/(6.**(3./2.)*ct.pi*(m1+m2)*ct.G/(ct.c**3.))
	
	"""
	#PhenomD merger frequency
	f_merg = 0.018*ct.c**3/(ct.G*(m1+m2)) 
	"""	

	#FH formalism extended to unequal mass
	"""
	f_merg = 0.02*ct.c**3/(ct.G*(m1+m2)) 

	flow = (start_time*(256.0/5.0)*(ct.G**(5/3)/ct.c**5)*(m1*m2)/(m1+m2)**(1/3)*(2.0*ct.pi)**(8/3) + f_merg**(-8/3))**(-3/8)
	
	if end_time == 0.0:
		fmax = fnyq = 3e-1*ct.c**3/(ct.G*(m1+m2))
	else:
		fmax = fnyq = (end_time*(256.0/5.0)*(ct.G**(5/3)/ct.c**5)*(m1*m2)/(m1+m2)**(1/3)*(2.0*ct.pi)**(8/3) + f_merg**(-8/3))**(-3/8)
	"""

	#Post newtonian expansion to 1st order from SNR calibration docs
	N = m1*m2/(m1+m2)**2.
	tau = N*(start_time*ct.c)/(5.*(m1+m2)*ct.G/(ct.c**2.))
	flow = 1./(8.*ct.pi*(m1+m2)*ct.G/(ct.c**2.)*tau**(3./8.))*(1.+((11./32)*N+743./2688.)*tau**(-1./4.))*ct.c
	
	if end_time == 0.0:
		fmax = fnyq = 3e-1*ct.c**3/(ct.G*(m1+m2))
	else:
		tau = N*(end_time*ct.c)/(5.*(m1+m2)*ct.G/(ct.c**2.))
		fmax = fnyq = 1./(8.*ct.pi*(m1+m2)*ct.G/(ct.c**2.)*tau**(3./8.))*(1.+((11./32)*N+743./2688.)*tau**(-1./4.))*ct.c


	compare = 1e-3*ct.c**3/(ct.G*(m1+m2))
	if flow/compare>= 1.0:
		delta_f = flow/1e1
	else:
		delta_f = flow/1e1

	if m1+m2 >= 1e9 * lal.MSUN_SI:
		delta_f = flow/1e1

	if end_time >= 1.0*ct.Julian_year/(1.+z):
		delta_f = flow/1e4


	# Extrinsic parameters
	# coalescence phase and polarization angle
	coa_phase, polarization_angle = 0.0, 0.0
	# sky position
	ra, dec = 0.0, 0.0
	# distance in m and inclination
	dist, incl = dist*1e6 * lal.PC_SI, ct.pi/2.

	# waveform model to use
	if waveform_type == 'PhenomD':
		approx1 = lalsimulation.IMRPhenomD
	elif waveform_type == 'PhenomC':
		approx1 = lalsimulation.IMRPhenomC
	# We are doing a discrete set of dF not an infinite as the equation above shows
	
	samp_rate = 2* fnyq

	# We're actually doing a numerical integral so df -> \delta f
	#delta_f = flow/change

	#help(lalsimulation.SimInspiralFD)
	hpf, hxf = lalsimulation.SimInspiralFD(
		                    m1, m2,
		                    s1x, s1y, s1z,
		                    s2x, s2y, s2z,
		                    dist, incl, coa_phase,
		                    0.0,0.0,0.0, 
		                    delta_f, flow, fmax, flow,
		                    None,approx1)

	# For convenience, we'll reuse hpf and redefine to be h1
	#hpf.data.data += hxf.data.data
	h1 = hpf
	#pdb.set_trace()
	h1_magnitude = np.absolute(h1.data.data)

	freqs = np.array([hpf.f0+ k*hpf.deltaF for k in np.arange(int(hpf.data.length))])

	ind_start = np.where(h1_magnitude == h1_magnitude.max())[0][0]

	h1_magnitude = h1_magnitude[ind_start::]
	freqs = freqs[ind_start::]

	rid = np.where(h1_magnitude == 0.0)[0]
	keep = np.delete(np.arange(len(h1_magnitude)), rid)

	h1_magnitude = h1_magnitude[keep]
	freqs = freqs[keep] 
	
	return freqs, h1_magnitude

def hchar_func(m1,m2,z, s1, s2, start_time, end_time,waveform_type='PhenomD'):

	#time in years before merger
	#redshift times back to source frame
	start_time_source = start_time/(1+z)
	end_time_source = end_time/(1+z)

	f_s, h_tilde_f_s = generate_FD_waveform(m1,m2,z,s1,s2, start_time_source,end_time_source,waveform_type)

	f_obs = f_s/(1.0+z)
	
	h_char =  np.sqrt(4.0*f_s**2*(h_tilde_f_s)**2)

	
	return f_obs, h_char


