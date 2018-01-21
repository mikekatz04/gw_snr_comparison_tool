"""
Make_plot module turns gridded datasets into helpful plots. It is designed for LISA Signal-to-Noise (SNR) comparisons across sennsitivity curves and parameters.
	
	The three main classes are plot types: waterfall, horizon, and ratio. 

	Waterfall: 
		SNR contour plot based on plots from LISA proposal.

	Ratio:
		Comparison plot of the log10 of the ratio of SNRs for two different inputs. This plot also contains loss/gain contours, which describe when sources are gained or lost compared to one another based on a user specified SNR cut.

	Horizon:
		SNR contour plots comparing multipile inputs. User can specify contour value. The default is the user specified SNR cut. 
"""


import json
import pdb
import sys
from collections import OrderedDict

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib  import cm
from matplotlib import colors
from astropy.io import ascii


class CreateSinglePlot:
	"""
	This is the base class for the subclasses designed for creating the plots.

		Mandatory Inputs:
			fig - Figure environment for the plots.
			axis - Axis object representing specific plot.
			xvals - list of x-value arrays for the plot.
			yvals - list of y-value arrays for the plot.
			zvals - list of z-value arrays for the plot.

		Optional Inputs:
			limits_dict - dict containing axis limits and axes labels information.

				limits_dict inputs/keys:
					xlims, ylims - length 2 list with min followed by max. default is log for x and linear for y. If log, the limits should be log of values.
					dx, dy - x-change and y-change
					xscale, yscale - scaling for axes

			label_dict - dict containing label information for x labels, y labels, and title.

			extra_dict - dict containing extra plot information to aid in customization.

			legend_dict - dict describing legend labels and properties. This is mainly used for horizon plots.
	"""


	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={}, 
		label_dict={}, extra_dict={}, legend_dict={}):

		self.fig = fig
		self.axis = axis
		self.xvals = xvals
		self.yvals = yvals
		self.zvals = zvals

		self.limits_dict, self.label_dict, self.extra_dict, self.legend_dict = limits_dict, label_dict, extra_dict, legend_dict

	def setup_plot(self):

		xticks = np.arange(float(self.limits_dict['xlims'][0]), 
			float(self.limits_dict['xlims'][1]) 
			+ float(self.limits_dict['dx']), 
			float(self.limits_dict['dx']))

		yticks = np.arange(float(self.limits_dict['ylims'][0]), 
			float(self.limits_dict['ylims'][1])
			 + float(self.limits_dict['dy']), 
			 float(self.limits_dict['dy']))

		self.axis.set_xlim(xticks.min(), xticks.max())
		self.axis.set_ylim(yticks.min(), yticks.max())
		
		if self.extra_dict['spacing'] == 'wide':
			x_inds = np.arange(len(xticks))
			y_inds = np.arange(len(yticks))
		else:
			x_inds = np.arange(1, len(xticks)-1)
			y_inds = np.arange(1, len(yticks)-1)

		self.axis.set_xticks(xticks[x_inds])
		self.axis.set_yticks(yticks[y_inds])

		if self.limits_dict['xscale'] == 'log':
			self.axis.set_xticklabels([r'$10^{%i}$'%int(i) 
				for i in xticks[x_inds]])
		else:
			self.axis.set_xticklabels([r'$%i$'%int(i) 
				for i in xticks[x_inds]])

		if self.limits_dict['yscale'] == 'log':
			self.axis.set_yticklabels([r'$10^{%i}$'%int(i) 
				for i in yticks[y_inds]])
		else:
			self.axis.set_yticklabels([r'$%i$'%int(i) 
				for i in yticks[y_inds]])

		self.axis.grid(True, linestyle='-', color='0.75')

		title_fontsize = 20
		if 'title' in self.label_dict.keys():
			if 'title_fontsize' in self.label_dict.keys():
				title_fontsize = float(self.label_dict['title_fontsize'])
			self.axis.set_title(r'%s'%self.label_dict['title'],
				fontsize=title_fontsize)

		label_fontsize = 20
		if 'xlabel' in self.label_dict.keys():
			if 'xlabel_fontsize' in self.label_dict.keys():
				label_fontsize = float(self.label_dict['xlabel_fontsize'])
			self.axis.set_xlabel(r'%s'%self.label_dict['xlabel'],
				fontsize=label_fontsize)

		label_fontsize = 20
		if 'ylabel' in self.label_dict.keys():
			if 'ylabel_fontsize' in self.label_dict.keys():
				label_fontsize = float(self.label_dict['ylabel_fontsize'])
			self.axis.set_ylabel(r'%s'%self.label_dict['ylabel'],
				fontsize=label_fontsize)

		return

	def interpolate_data(self):

		points = [np.shape(x_arr)[0]*np.shape(x_arr)[1]
			for x_arr in self.xvals]
		min_points = np.argmin(points)
		max_points = np.argmax(points)

		new_x = np.linspace(self.xvals[min_points].min(),
			self.xvals[min_points].max(),
			np.shape(self.xvals[min_points])[1])

		new_y = np.logspace(np.log10(self.yvals[min_points]).min(),
			np.log10(self.yvals[min_points]).max(),
			np.shape(self.xvals[min_points])[0])

		new_x, new_y = np.meshgrid(new_x, new_y)

		new_z = griddata((self.xvals[max_points].ravel(),
			self.yvals[max_points].ravel()),
			self.zvals[max_points].ravel(),
			(new_x, new_y), method='linear')
		
		self.xvals[max_points], self.yvals[max_points], self.zvals[max_points] = new_x, new_y, new_z
		
		return


class Waterfall(CreateSinglePlot):


	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={},
		label_dict={}, extra_dict={}, legend_dict={}):

		CreateSinglePlot.__init__(self, fig, axis, xvals,yvals,zvals,
			limits_dict, label_dict, extra_dict, legend_dict)

	def make_plot(self):

		#sets levels of main contour plot
		colors1 = ['None','darkblue', 'blue', 'deepskyblue', 'aqua',
			'greenyellow', 'orange', 'red','darkred']
		levels = np.array([0.,10,20,50,100,200,500,1000,3000,1e6])
		
		#produce filled contour of SNR vs. z vs. Mtotal
		sc=self.axis.contourf(self.xvals[0],self.yvals[0],self.zvals[0],
			levels = levels, colors=colors1)

		#add colorbar axes for both contour plots
		cbar_ax = self.fig.add_axes([0.83, 0.55, 0.03, 0.4])

		#establish colorbar and labels for main contour plot
		self.fig.colorbar(sc, cax=cbar_ax, ticks=levels)
		cbar_ax.set_yticklabels([int(i)
			for i in np.delete(levels,-1)], fontsize = 17)
		cbar_ax.set_ylabel(r"$\rho_i$", fontsize = 20)

		return


class Ratio(CreateSinglePlot):
	"""
	Test docstring
	"""


	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={},
		label_dict={}, extra_dict={}, legend_dict={}):

		CreateSinglePlot.__init__(self, fig, axis, xvals,yvals,zvals,
			limits_dict, label_dict, extra_dict, legend_dict)

	def make_plot(self):

		if len(self.xvals) != 2:
			raise Exception("Length of vals not equal to 2. Ratio plots must have 2 inputs.")

		#interpolate data so that the dimensions of each data set are the same
		if np.shape(self.xvals[0]) != np.shape(self.xvals[1]):
			self.interpolate_data()

		#sets colormap for ratio comparison plot
		cmap2 = cm.seismic

		#set attributes of ratio comparison contour
		normval2 = 2.0
		num_contours2 = 40 #must be even
		levels2 = np.linspace(-normval2, normval2,num_contours2)
		norm2 = colors.Normalize(-normval2, normval2)

		#initialize dimensions of loss/gain contour and ratio contour
		diffout2, loss_gain_contour = self.find_difference_contour()

		#plot ratio contours
		sc3=self.axis.contourf(self.xvals[0],self.yvals[0],diffout2,
			levels = levels2, norm=norm2, extend='both', cmap=cmap2)

		#toggle line contours of orders of magnitude of ratio comparisons
		#ax[w].contour(xout,yout,Diffout2, np.array([-2.0, -1.0,1.0, 2.0]), norm=norm2, extend='both', colors = 'grey', linewidths = 1.0, linestyles= 'dashed')

		self.axis.contour(self.xvals[0],self.yvals[0],loss_gain_contour,1,colors = 'grey', linewidths = 2)

		#establish colorbar and labels for ratio comp contour plot
		cbar_ax2 = self.fig.add_axes([0.83, 0.1, 0.03, 0.4])
		self.fig.colorbar(sc3, cax=cbar_ax2,
			ticks=np.array([-3.0,-2.0,-1.0,0.0, 1.0,2.0, 3.0]))
		cbar_ax2.set_yticklabels([r'$10^{%i}$'%i
			for i in np.arange(-normval2, normval2+1.0, 1.0)], fontsize = 17)
		cbar_ax2.set_ylabel(r"$\rho_i/\rho_0$", fontsize = 20)

		return

	def find_difference_contour(self):

		#set indices of loss,gained. Also set rid for when neither curve measures source. inds_check tells the ratio calculator not to control_zout if both SNRs are below 1
		zout = self.zvals[0]
		control_zout = self.zvals[1]

		inds_gained = np.where((zout>=SNR_CUT) & (control_zout< SNR_CUT))
		inds_lost = np.where((zout<SNR_CUT) & (control_zout>=SNR_CUT))
		inds_rid = np.where((zout<1.0) & (control_zout<1.0))
		inds_check = np.where((zout.ravel()<1.0)
			& (control_zout.ravel()<1.0))[0]

		#set diff2 to ratio for purposed of determining raito differences
		diff2 = zout/control_zout

		#flatten arrays	
		diffcheck = diff2.ravel()
		diff2 =  diff2.ravel()

		# the following determines the log10 of the ratio difference if it is extremely small, we neglect and put it as zero (limits chosen to resemble ratios of less than 1.05 and greater than 0.952)
		i = 0
		for d in diff2:
			if i not in inds_check:
				if d >= 1.05:
					diff2[i] = np.log10(diff2[i])
				elif d<= 0.952:
					diff2[i] = -np.log10(1.0/diff2[i])
				else:
					diff2[i] = 0.0
			else: 
				diff2[i] = 0.0
			i+=1

		#reshape difference array for dimensions of contour
		diffout2 = np.reshape(diff2, np.shape(zout))

		#change inds rid value to zero
		diffout2[inds_rid] = 0.0

		#initialize loss/gain
		loss_gain_contour = np.zeros(np.shape(zout))

		#fill out loss/gain
		j = -1
		for i in (inds_lost, inds_gained):
			loss_gain_contour[i] = j
			j += 2

		return diffout2, loss_gain_contour


class Horizon(CreateSinglePlot):


	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={},
		label_dict={}, extra_dict={}, legend_dict={}):

		CreateSinglePlot.__init__(self, fig, axis, xvals,yvals,zvals,
			limits_dict, label_dict, extra_dict, legend_dict)

	def make_plot(self):
		"""
		test_test
		"""

		#sets levels of main contour plot
		colors1 = ['blue', 'green', 'red','purple', 'orange',
			'gold','magenta']

		contour_val = SNR_CUT

		if 'snr_contour_value' in self.extra_dict.keys():
			contour_val = float(self.extra_dict['snr_contour_value'])
		
		for j in range(len(self.zvals)):
			hz = self.axis.contour(self.xvals[j],self.yvals[j],
				self.zvals[j], np.array([contour_val]), 
				colors = colors1[j], linewidths = 1., linestyles= 'solid')

			v = np.transpose(hz.collections[0].get_paths()[0].vertices)

			#plot invisible lines for purpose of creating a legend
			if self.legend_dict != {}:
				self.axis.plot([0.1,0.2],[0.1,0.2],color = colors1[j],
					label = self.legend_dict['labels'][j])

			else:
				self.axis.plot([0.1,0.2],[0.1,0.2],color = colors1[j])
			
		if self.legend_dict != {}:
			#defaults followed by change
			loc = 'upper left'
			if 'loc' in self.legend_dict.keys():
				loc = self.legend_dict['loc']

			size = 10
			if 'size' in self.legend_dict.keys():
				size = float(self.legend_dict['size'])

			bbox_to_anchor = None
			if 'bbox_to_anchor' in self.legend_dict.keys():
				bbox_to_anchor = self.legend_dict['bbox_to_anchor']

			ncol = 1
			if 'ncol' in self.legend_dict.keys():
				ncol = int(self.legend_dict['ncol'])	

			self.axis.legend(loc=loc, bbox_to_anchor=bbox_to_anchor,
				ncol=ncol, prop={'size':size})

		return


class PlotVals:
	""" 
	This class is designed to carry around the data for each plot as an attribute of self.

		Inputs/Attributes:
			x_arr_list: list of gridded, 2D datasets representing the x-values.
			y_arr_list: list of gridded, 2D datasets representing the y-values.
			z_arr_list: list of gridded, 2D datasets representing the z-values.
	"""

	def __init__(self, x_arr_list, y_arr_list, z_arr_list):

		self.x_arr_list, self.y_arr_list, self.z_arr_list = x_arr_list, y_arr_list, z_arr_list


class ReadInData:


	def __init__(self, pid, file_dict, limits_dict={}):

		self.file_name = file_dict['name']

		self.file_type = self.file_name.split('.')[-1]

		if self.file_type == 'csv':
			self.file_type = 'txt'

		self.x_col_name = 'x'
		if 'x_column_label' in file_dict.keys():
			self.x_col_name = file_dict['x_column_label']
		else:
			if 'x_column_label' in pid['general'].keys():
				self.x_col_name = pid['general']['x_column_label']


		self.y_col_name = 'y'
		if 'y_column_label' in file_dict.keys():
			self.y_col_name = file_dict['y_column_label']
		else:
			if 'y_column_label' in pid['general'].keys():
				self.y_col_name = pid['general']['y_column_label']

		self.z_col_name = file_dict['label']

		getattr(self, self.file_type + '_read_in')()	

		self.x_append_value = self.xvals
		if 'xscale' in limits_dict.keys():
			if limits_dict['xscale'] =='log':
				self.x_append_value = np.log10(self.xvals)
		else:
			if 'xscale' in pid['general'].keys():
				if pid['general']['xscale'] == 'log':
					self.x_append_value = np.log10(self.xvals)

		#default is linear y scale
		self.y_append_value = self.yvals
		if 'yscale' in limits_dict.keys():
			if limits_dict =='log':
				self.y_append_value = np.log10(self.yvals)
		else:
			if 'yscale' in pid['general'].keys():
				if pid['general']['yscale'] == 'log':
					self.y_append_value = np.log10(self.yvals)

		self.z_append_value = self.zvals

	def txt_read_in(self):

		data = ascii.read(WORKING_DIRECTORY + '/' + self.file_name)

		num_x_pts = len(np.unique(data[self.x_col_name]))
		num_y_pts = len(np.unique(data[self.y_col_name]))

		self.xvals = np.reshape(np.asarray(data[self.x_col_name]), (num_y_pts,num_x_pts))
		self.yvals = np.reshape(np.asarray(data[self.y_col_name]), (num_y_pts,num_x_pts))
		self.zvals = np.reshape(np.asarray(data[self.z_col_name]), (num_y_pts,num_x_pts))

		return

	def hdf5_read_in(self):

		with hdf5.File(WORKING_DIRECTORY + '/' + self.file_name) as f:
			data = f['data']

			num_x_pts = len(np.unique(data[self.x_col_name][:]))
			num_y_pts = len(np.unique(data[self.y_col_name][:]))

			self.xvals = np.reshape(data[self.x_col_name][:],
				(num_y_pts,num_x_pts))
			self.yvals = np.reshape(data[self.y_col_name][:],
				(num_y_pts,num_x_pts))
			self.zvals = np.reshape(data[self.z_col_name][:],
				(num_y_pts,num_x_pts))

		return	


def read_in_data(pid):


	control_dict = pid['plot_info']

	x = [[]for i in np.arange(len(control_dict.keys()))]
	y = [[] for i in np.arange(len(control_dict.keys()))]
	z = [[] for i in np.arange(len(control_dict.keys()))]

	for k, axis_string in enumerate(control_dict.keys()):
		limits_dict = {}
		if 'limits' in control_dict[axis_string].keys():
			liimits_dict = control_dict[axis_string]['limits']

		for j, file_dict in enumerate(control_dict[axis_string]['file']):
			data_class = ReadInData(pid, file_dict, limits_dict)

			x[k].append(data_class.x_append_value)
			y[k].append(data_class.y_append_value)
			z[k].append(data_class.z_append_value)

		print(axis_string)

	for k, axis_string in enumerate(control_dict.keys()):
		if 'control' in control_dict[axis_string]:
			if 'name' in control_dict[axis_string]['control']:
				file_dict = control_dict[axis_string]['control']
				if 'limits' in control_dict[axis_string].keys():
					liimits_dict = control_dict[axis_string]['limits']

				data_class = ReadInData(pid, file_dict, limits_dict)

				x[k].append(data_class.x_append_value)
				y[k].append(data_class.y_append_value)
				z[k].append(data_class.z_append_value)

			elif 'index' in control_dict[axis_string]:
				index = int(control_dict[axis_string]['control']['index'])

				x[k].append(x[index][0])
				y[k].append(y[index][0])
				z[k].append(z[index][0])

		print(axis_string)


	for k, axis_string in enumerate(control_dict.keys()):
		
		#takes first file from plot
		if 'indices' in control_dict[axis_string]:
			for index in control_dict[axis_string]['indices']:

				index = int(index)

				x[k].append(x[index][0])
				y[k].append(y[index][0])
				z[k].append(z[index][0])


	value_classes = []
	for k in range(len(x)):
		value_classes.append(PlotVals(x[k],y[k],z[k]))

	return value_classes


def plot_main(pid):

	global WORKING_DIRECTORY, SNR_CUT

	WORKING_DIRECTORY = '.'
	if 'WORKING_DIRECTORY' in pid['general'].keys():
		WORKING_DIRECTORY = pid['general']['WORKING_DIRECTORY']

	SNR_CUT = 5.0
	if 'SNR_CUT' in pid['general'].keys():
		SNR_CUT = pid['general']['SNR_CUT']

	if "switch_backend" in pid['general'].keys():
		plt.switch_backend(pid['general']['switch_backend'])

	plot_class_dict = {'horizon':Horizon, 'waterfall':Waterfall,
		'ratio':Ratio}

	#defaults for sharing axes
	sharex = True
	sharey = True

	#if share axes options are in input, change to option
	if 'sharex' in pid['general'].keys():
		sharex = pid['general']['sharex']

	if 'sharey' in pid['general'].keys():
		sharey = pid['general']['sharey']

	#declare figure and axes environments
	fig, ax = plt.subplots(nrows = int(pid['general']['num_rows']),
		ncols = int(pid['general']['num_cols']),
		sharex = sharex, sharey = sharey)

	#set figure size
	figure_width = 8
	if 'figure_width' in pid['general'].keys():
		figure_width = pid['general']['figure_width']

	figure_height = 8
	if 'figure_height' in pid['general'].keys():
		figure_height = pid['general']['figure_height']

	fig.set_size_inches(figure_width,figure_height)

	try:
		ax = ax.ravel()
	except AttributeError:
		ax = [ax]

	plot_data = read_in_data(pid)

	for i, axis in enumerate(ax):
		trans_dict = pid['plot_info'][str(i)]
		for name in ['legend', 'limits', 'label', 'extra']:
			if name not in trans_dict:
				trans_dict[name] = {}
		if 'xlims' in pid['general'].keys() and 'xlims' not in trans_dict['limits']:
			trans_dict['limits']['xlims'] = pid['general']['xlims']
		if 'dx' in pid['general'].keys() and 'dx' not in trans_dict['limits']:
			trans_dict['limits']['dx'] = float(pid['general']['dx'])
		if 'xscale' in pid['general'].keys() and 'xscale' not in trans_dict['limits']:
			trans_dict['limits']['xscale'] = pid['general']['xscale']

		if 'ylims' in pid['general'].keys() and 'ylims' not in trans_dict['limits']:
			trans_dict['limits']['ylims'] = pid['general']['ylims']
		if 'dy' in pid['general'].keys() and 'dy' not in trans_dict['limits']:
			trans_dict['limits']['dy'] = float(pid['general']['dy'])
		if 'yscale' in pid['general'].keys() and 'yscale' not in trans_dict['limits']:
			trans_dict['limits']['yscale'] = pid['general']['yscale']

		trans_dict['extra']['spacing'] = 'tight'
		if 'spacing' in pid['general'].keys():
			if pid['general']['spacing'] == 'wide':
				trans_dict['extra']['spacing'] = 'wide'

		trans_plot_class = plot_class_dict[trans_dict['type']](fig, axis,
			plot_data[i].x_arr_list,plot_data[i].y_arr_list,
			plot_data[i].z_arr_list,
			trans_dict['limits'], trans_dict['label'],
			trans_dict['extra'], trans_dict['legend'])


		trans_plot_class.make_plot()
		trans_plot_class.setup_plot()

		print(i)

	fig.subplots_adjust(left=0.12, top=0.92, bottom=0.1, wspace=0.3, hspace=0.3)

	if 'spacing' in pid['general'].keys():
		if pid['general']['spacing'] == 'wide':
			pass

		else:
			fig.subplots_adjust(wspace=0.0, hspace=0.0)

	else:
		fig.subplots_adjust(wspace=0.0, hspace=0.0)


	plot_types = [pid['plot_info'][axis_string]['type']
		for axis_string in pid['plot_info'].keys()]

	if 'ratio' in plot_types or 'waterfall' in plot_types:
		fig.subplots_adjust(right=0.79)

	#label for axis
	fig_label_fontsize = 20
	if 'fig_label_fontsize' in pid['general'].keys():
		fig_label_fontsize = float(pid['general']['fig_label_fontsize'])

	fig.text(0.01, 0.51, r'%s'%(pid['general']['fig_y_label']),
		rotation = 'vertical', va = 'center', fontsize = fig_label_fontsize)

	fig.text(0.45, 0.02, r'%s'%(pid['general']['fig_x_label']),
		ha = 'center', fontsize = fig_label_fontsize)
		
	if 'save_figure' in pid['general'].keys():
		if pid['general']['save_figure'] == True:
			plt.savefig(WORKING_DIRECTORY + '/' + 
				pid['general']['output_path'], dpi=200)
	
	if 'show_figure' in pid['general'].keys():
		if pid['general']['show_figure'] == True:
			plt.show()

	return

if __name__ == '__main__':

	plot_info_dict = json.load(open(sys.argv[1], 'r'),
		object_pairs_hook=OrderedDict)
	plot_main(plot_info_dict)


