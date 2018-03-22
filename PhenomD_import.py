import ctypes
import pdb
from astropy.cosmology import Planck15 as cosmo
import numpy as np
import time


class PhenomDWaveforms:
	def __init__(self, m1, m2, chi1, chi2, z, st, et, num_points=8192, exec_call="c_code_phenomd_try.so"):

		remove_axis = False
		try:
			len(m1)
			try:
				len(st)
			except TypeError:
				st = np.full(len(m1), st)
				et = np.full(len(m1), et)


		except TypeError:
			remove_axis=True
			m1, m2,chi1, chi2, z, st, et = np.array([m1]), np.array([m2]), np.array([chi1]), np.array([chi2]), np.array([z]), np.array([st]), np.array([et]) 


		self.m1, self.m2, self.chi1, self.chi2, self.z, self.st, self.et = m1, m2, chi1, chi2, z, st, et

		length = len(m1)
		dist = cosmo.luminosity_distance(z).value

		freq_amp_cast=ctypes.c_double*num_points*length
		freqs = freq_amp_cast()
		amplitude = freq_amp_cast()

		fmrg_fpeak_cast =ctypes.c_double*length
		fmrg = fmrg_fpeak_cast()
		fpeak = fmrg_fpeak_cast()



		c_obj = ctypes.CDLL(exec_call)

		start = time.time()
		print('start')
		c_obj.Amplitude(ctypes.byref(freqs), ctypes.byref(amplitude), ctypes.byref(fmrg), ctypes.byref(fpeak), m1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), chi1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), chi2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), st.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), et.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(length), ctypes.c_int(num_points))


		self.freqs, self.amplitude, self.fmrg, self.fpeak = np.ctypeslib.as_array(freqs), np.ctypeslib.as_array(amplitude), np.ctypeslib.as_array(fmrg), np.ctypeslib.as_array(fpeak)
		
		if remove_axis:
			self.freqs, self.amplitude, = np.squeeze(self.freqs), np.squeeze(self.amplitude)
			self.fmrg, self.fpeak = self.fmrg[0], self.fpeak[0]

		print(time.time()-start)


if __name__ == "__main__":

	#test = "testlib.so"

	m1 =m2 = np.logspace(1, 5, 20)
	z = np.logspace(-2, 2, 5)
	
	chi_1 = chi_2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
	
	m1, m2, z, chi_1, chi_2 = np.meshgrid(m1, m2, z, chi_1, chi_2)

	m1, m2, z, chi_1, chi_2 = m1.ravel(), m2.ravel(), z.ravel(), chi_1.ravel(), chi_2.ravel()
	dist = cosmo.luminosity_distance(z).value
	st = 1.0
	et = 0.0
	
	phenomd_wfs = PhenomDWaveforms(m1, m2, chi_1, chi_2, z, st, et, num_points=4096)

	pdb.set_trace()
	print(phenomd_wfs.freqs.shape, phenomd_wfs.fmrg)

	

