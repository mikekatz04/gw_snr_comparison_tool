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
import h5py

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib  import cm
from matplotlib import colors
from astropy.io import ascii

import warnings
warnings.filterwarnings("ignore")


class CreateSinglePlot:

	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={}, 
		label_dict={}, extra_dict={}, legend_dict={}):
		
		"""
		This is the base class for the subclasses designed for creating the plots.

			Mandatory Inputs:
				fig - figure object - Figure environment for the plots.
				axis - axes object - Axis object representing specific plot.
				xvals - list of 2d arrays - list of x-value arrays for the plot.
				yvals - list of 2d arrays - list of y-value arrays for the plot.
				zvals - list of 2d arrays - list of z-value arrays for the plot.

			Optional Inputs:
				limits_dict - dict containing axis limits and axes labels information.

					limits_dict inputs/keys:
						xlims, ylims - length 2 list of floats - min followed by max. default is log for x and linear for y. If log, the limits should be log of values.
						dx, dy - float - x-change and y-change
						xscale, yscale - string - scaling for axes. Either 'log' or 'lin'.

				label_dict - dict containing label information for x labels, y labels, and title.

					label_dict inputs/keys:
						title - string - title for each plot
						title_fontsize - float - fontsize for title_fontsize
						xlabel, ylabel - string - x, y axis label
						xlabel_fontsize, ylabel_fontsize - float - x,y axis label fontsize		

				extra_dict - dict containing extra plot information to aid in customization.

					extra_dict inputs/keys:
						snr_contour_value - float - snr value for contour lines on a horizon plot. This will override SNR_CUT for horizon plots. 
						spacing - string - Choices are 'tight' or 'wide' spacing for overall plots. 'tight' spacing will cutouf the last entry in ticklabels.

				legend_dict - dict describing legend labels and properties. This is mainly used for horizon plots.

					legend_dict inputs/keys given under horizon plot docstring. 

		"""

		self.fig = fig
		self.axis = axis
		self.xvals = xvals
		self.yvals = yvals
		self.zvals = zvals

		self.limits_dict, self.label_dict, self.extra_dict, self.legend_dict = limits_dict, label_dict, extra_dict, legend_dict

	def setup_plot(self):
		"""
		This method takes an axis and sets up plot limits and labels according to label_dict and limits_dict from CreateSinglePlot __init__.

			limits_dict - dict containing axis limits and axes labels information.

				limits_dict inputs/keys:
					xlims, ylims - length 2 list of floats - min followed by max. default is log for x and linear for y. If log, the limits should be log of values.
					dx, dy - float - x-change and y-change
					xscale, yscale - string - scaling for axes. Either 'log' or 'lin'.

			label_dict - dict containing label information for x labels, y labels, and title.

				label_dict inputs/keys:
					title - string - title for each plot
					title_fontsize - float - fontsize for title_fontsize
					xlabel, ylabel - string - x, y axis label
					xlabel_fontsize, ylabel_fontsize - float - x,y axis label fontsize	

		"""

		#setup xticks and yticks and limits
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
		
		#adjust ticks for spacing. If 'wide' then show all labels, if 'tight' remove end labels. 
		if self.extra_dict['spacing'] == 'wide':
			x_inds = np.arange(len(xticks))
			y_inds = np.arange(len(yticks))
		else:
			#remove end labels
			x_inds = np.arange(1, len(xticks)-1)
			y_inds = np.arange(1, len(yticks)-1)

		self.axis.set_xticks(xticks[x_inds])
		self.axis.set_yticks(yticks[y_inds])

		#set tick labels based on scale
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

		#add grid
		self.axis.grid(True, linestyle='-', color='0.75')

		#add title
		title_fontsize = 20
		if 'title' in self.label_dict.keys():
			if 'title_fontsize' in self.label_dict.keys():
				title_fontsize = float(self.label_dict['title_fontsize'])
			self.axis.set_title(r'%s'%self.label_dict['title'],
				fontsize=title_fontsize)

		#add x,y labels
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
		"""
		Method interpolates data if two data sets have different shapes. This method is mainly used on ratio plots to allow for direct comparison of snrs. The higher resolution array is reduced to the lower resolution. However, functionality will be added for waterfall and horizon plts. 
		"""

		#check the number of points in the two arrays and select max and min
		points = [np.shape(x_arr)[0]*np.shape(x_arr)[1]
			for x_arr in self.xvals]

		min_points = np.argmin(points)
		max_points = np.argmax(points)

		#create a new arrays to replace array with more to interpolated array with less points.
		new_x = np.linspace(self.xvals[min_points].min(),
			self.xvals[min_points].max(),
			np.shape(self.xvals[min_points])[1])

		new_y = np.logspace(np.log10(self.yvals[min_points]).min(),
			np.log10(self.yvals[min_points]).max(),
			np.shape(self.xvals[min_points])[0])

		#grid new arrays
		new_x, new_y = np.meshgrid(new_x, new_y)

		#use griddate fro scipy.interpolate to create new z array
		new_z = griddata((self.xvals[max_points].ravel(),
			self.yvals[max_points].ravel()),
			self.zvals[max_points].ravel(),
			(new_x, new_y), method='linear')
		
		self.xvals[max_points], self.yvals[max_points], self.zvals[max_points] = new_x, new_y, new_z
		
		return


class Waterfall(CreateSinglePlot):


	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={},
		label_dict={}, extra_dict={}, legend_dict={}):
		"""
		Waterfall is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information. 

		Waterfall creates an snr filled contour plot similar in style to those seen in the LISA proposal. Contours are displayed at snrs of 10, 20, 50, 100, 200, 500, 1000, and 3000 and above. If lower contours are needed, adjust contour_vals in extra_dict for the specific plot. 

			Contour_vals needs to start with zero and end with a higher value than the max in the data. Contour_vals needs to be a list of max length 9 including zero and max value. 
		"""


		CreateSinglePlot.__init__(self, fig, axis, xvals,yvals,zvals,
			limits_dict, label_dict, extra_dict, legend_dict)

	def make_plot(self):
		"""
		This methd creates the waterfall plot. 
		"""

		#sets levels of main contour plot
		colors1 = ['None','darkblue', 'blue', 'deepskyblue', 'aqua',
			'greenyellow', 'orange', 'red','darkred']

		levels = np.array([0.,10,20,50,100,200,500,1000,3000,1e6])

		if 'contour_vals' in self.extra_dict.keys():
			levels = np.asarray(self.extra_dict['contour_vals'])
		
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
	

	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={},
		label_dict={}, extra_dict={}, legend_dict={}):
		"""
		Ratio is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information. 

		Ratio creates an filled contour plot comparing snrs from two different data sets. Typically, it is used to compare sensitivty curves and/or varying binary parameters. It takes the snr of the first dataset and divides it by the snr from the second dataset. The log10 of this ratio is ploted. Additionally, a loss/gain contour is plotted. Loss/gain contour is based on SNR_CUT but can be overridden with 'snr_contour_value' in extra_dict. A gain indicates the first dataset reaches the snr threshold while the second does not. A loss is the opposite.  
		"""
		CreateSinglePlot.__init__(self, fig, axis, xvals,yvals,zvals,
			limits_dict, label_dict, extra_dict, legend_dict)

	def make_plot(self):
		"""
		This methd creates the ratio plot. 
		"""

		#check to make sure ratio plot has 2 arrays to compare. 
		if len(self.xvals) != 2:
			raise Exception("Length of vals not equal to 2. Ratio plots must have 2 inputs.")

		#check and interpolate data so that the dimensions of each data set are the same
		if np.shape(self.xvals[0]) != np.shape(self.xvals[1]):
			self.interpolate_data()

		#sets colormap for ratio comparison plot
		cmap2 = cm.seismic

		#set values of ratio comparison contour
		normval2 = 2.0
		num_contours2 = 40 #must be even
		levels2 = np.linspace(-normval2, normval2,num_contours2)
		norm2 = colors.Normalize(-normval2, normval2)

		#find loss/gain contour and ratio contour
		diff_out, loss_gain_contour = self.find_difference_contour()

		#plot ratio contours
		sc3=self.axis.contourf(self.xvals[0],self.yvals[0],diff_out,
			levels = levels2, norm=norm2, extend='both', cmap=cmap2)

		#toggle line contours of orders of magnitude of ratio comparisons
		if 'ratio_contour_lines' in self.extra_dict.keys():
			if self.extra_dict['ratio_contour_lines'] == True:
				self.axis.contour(self.xvals[0],self.yvals[0],diff_out, np.array([-2.0, -1.0, 1.0, 2.0]), colors = 'black', linewidths = 1.0)

		#plot loss gain contour
		loss_gain_status = True
		if "turn_off_loss_gain" in self.extra_dict.keys():
			loss_gain_status = False
			
		if loss_gain_status == True:
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
		"""
		This method finds the ratio contour and the loss gain contour values. Its inputs are the two datasets for comparison where the second is the control to compare against the first. 

			The input data sets need to be the same shape. CreateSinglePlot.interpolate_data corrects for two datasets of different shape.

			Returns: loss_gain_contour, ratio contour (diff)
			
		"""

		zout = self.zvals[0]
		control_zout = self.zvals[1]

		#set comparison value. Default is SNR_CUT
		comparison_value = SNR_CUT
		if 'snr_contour_value' in self.extra_dict.keys():
			comparison_value = self.extra_dict['snr_contour_value']

		#set indices of loss,gained. inds_check tells the ratio calculator not to control_zout if both SNRs are below 1
		inds_gained = np.where((zout>=comparison_value) & (control_zout< comparison_value))
		inds_lost = np.where((zout<comparison_value) & (control_zout>=comparison_value))

		#Also set rid for when neither curve measures source. 
		inds_rid = np.where((zout<1.0) & (control_zout<1.0))

		#set diff2 to ratio for purposed of determining raito differences
		diff = zout/control_zout

		#flatten arrays	
		diff =  diff.ravel()

		# the following determines the log10 of the ratio difference if it is extremely small, we neglect and put it as zero (limits chosen to resemble ratios of less than 1.05 and greater than 0.952)
		diff = np.log10(diff)*(diff >= 1.05) + (-np.log10(1.0/diff))*(diff<=0.952) + 0.0*((diff<1.05) & (diff>0.952))

		"""
		#inds_check tells the ratio calculator not to make comparison if both SNRs are below 0.1
		inds_check = np.where((zout.ravel()<0.1)
			& (control_zout.ravel()<0.1))[0]

		i = 0
		for d in diff:
			if i not in inds_check:
				if d >= 1.05:
					diff[i] = np.log10(diff[i])
				elif d<= 0.952:
					diff[i] = -np.log10(1.0/diff[i])
				else:
					diff[i] = 0.0
			else: 
				diff[i] = 0.0
			i+=1
		"""

		#reshape difference array for dimensions of contour
		diff = np.reshape(diff, np.shape(zout))

		#change inds rid and inds_check value to zero
		diff[inds_rid] = 0.0

		#initialize loss/gain
		loss_gain_contour = np.zeros(np.shape(zout))

		#fill out loss/gain
		j = -1
		for i in (inds_lost, inds_gained):
			#change all of inds at one time to j
			loss_gain_contour[i] = j
			j += 2

		return diff, loss_gain_contour


class Horizon(CreateSinglePlot):


	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={},
		label_dict={}, extra_dict={}, legend_dict={}):
		"""
		Horizon is a subclass of CreateSinglePlot. Refer to CreateSinglePlot class docstring for input information. 

		Horizon plots snr contour lines for a designated SNR value. The defaul is SNR_CUT, but can be overridden with "snr_contour_value" in extra_dict. Horizon can take as many curves as the user prefers and will plot a legend to label those curves. With its current design, it can only contour one snr value. 

		Additional Inputs:

		legend_dict - dict describing legend labels and properties.
		legend_dict inputs/keys:
						labels - list of strings - contains labels for each plot that will appear in the legend.
						loc - string or int - location of legend. Refer to matplotlib documentation for legend placement for choices. Default is 'upper left'. 
						size - float - size of legend. Default is 10. 
						bbox_to_anchor - list of floats, length 2 or 4 - Places legend in custom location. First two entries represent corner of box is placed. Second two entries (not required) represent how to stretch the legend box from there. See matplotlib documentation on bbox_to_anchor for specifics. 
						ncol - int - number of columns in legend. Default is 1. 
		"""

		CreateSinglePlot.__init__(self, fig, axis, xvals,yvals,zvals,
			limits_dict, label_dict, extra_dict, legend_dict)

	def make_plot(self):
		"""
		This method adds a horizon plot as desribed in the Horizon class docstring. The class contains an axis as a part of self. The horizon plot is added to this axis. Can compare up to 7 curves.
		"""
		#sets levels of main contour plot
		colors1 = ['blue', 'green', 'red','purple', 'orange',
			'gold','magenta']

		#set contour value. Default is SNR_CUT.
		contour_val = SNR_CUT
		if 'snr_contour_value' in self.extra_dict.keys():
			contour_val = float(self.extra_dict['snr_contour_value'])
		
		#plot contours
		for j in range(len(self.zvals)):
			hz = self.axis.contour(self.xvals[j],self.yvals[j],
				self.zvals[j], np.array([contour_val]), 
				colors = colors1[j], linewidths = 1., linestyles= 'solid')

			#plot invisible lines for purpose of creating a legend
			if self.legend_dict != {}:
				#plot a curve off of the grid with same color for legend label.
				self.axis.plot([0.1,0.2],[0.1,0.2],color = colors1[j],
					label = self.legend_dict['labels'][j])

			
		if self.legend_dict != {}:
			#Add legend. Defaults followed by change.
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


	def __init__(self, x_arr_list, y_arr_list, z_arr_list):
		""" 
		This class is designed to carry around the data for each plot as an attribute of self.

			Inputs/Attributes:
				x_arr_list -list of 2D arrays - list of gridded, 2D datasets representing the x-values.
				y_arr_list - list of 2d arrays - list of gridded, 2D datasets representing the y-values.
				z_arr_list - list of 2d arrays - list of gridded, 2D datasets representing the z-values.
		"""

		self.x_arr_list, self.y_arr_list, self.z_arr_list = x_arr_list, y_arr_list, z_arr_list


class ReadInData:


	def __init__(self, pid, file_dict, limits_dict={}):
		"""
		This class reads in data based on the pid and file_dict. The file_dict provides information about the files to read in. This information is transferred to the read in methods that work for .txt, .csv, and .hdf5. 

			Mandatory Inputs:
				pid - dict - plot_info_dict used in main code. It contains information for types of plots created and the general settings in the pid['general'] dict. 

					pid inputs/keys:
						see config file and README documentation. This dict contains everything for code.

						x_column_label, y_column_label - string - general x and y column names in file_dict
						xscale, yscale - string - 'log' or 'lin' representing general scale of data in x and y


				file_dict - dict - contains info about the file to read in
					file_dict inputs/keys:
						Mandatory:
						name - string - file name including path extension from WORKING_DIRECTORY.
						label - string - name of column for the z values.

						Optional:
						x_column_label, y_column_label - string - x and y column names in file_dict

			Optional Inputs:
				limits_dict - dict - contains info on scaling of x and y
					xscale, yscale - string - 'log' or 'lin' representing scale of data in x and y. 'lin' is default.

						


			Optional Inputs:
				limits_dict - dict containing axis limits and axes labels information.

					limits_dict inputs/keys:
						xlims, ylims - length 2 list of floats - min followed by max. default is log for x and linear for y. If log, the limits should be log of values.
						dx, dy - float - x-change and y-change
						xscale, yscale - string - scaling for axes. Either 'log' or 'lin'.
		"""

		self.file_name = file_dict['name']

		#get file type
		self.file_type = self.file_name.split('.')[-1]

		if self.file_type == 'csv':
			self.file_type = 'txt'

		#find x,y column names
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

		#z column name
		self.z_col_name = file_dict['label']

		#Extract data with either txt or hdf5 methods.
		getattr(self, self.file_type + '_read_in')()	

		#append x,y values based on xscale, yscale.
		self.x_append_value = self.xvals
		if 'xscale' in limits_dict.keys():
			if limits_dict['xscale'] =='log':
				self.x_append_value = np.log10(self.xvals)
		else:
			if 'xscale' in pid['general'].keys():
				if pid['general']['xscale'] == 'log':
					self.x_append_value = np.log10(self.xvals)

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
		"""
		Method for reading in text or csv files. This uses ascii class from astropy.io for flexible input. It is slower than numpy, but has greater flexibility with less input.

			Inputs: 
				file_name, x and y column names from self. 

			Return: Add x,y,z values to self.
		"""

		#read in
		data = ascii.read(WORKING_DIRECTORY + '/' + self.file_name)

		#find number of distinct x and y points.
		num_x_pts = len(np.unique(data[self.x_col_name]))
		num_y_pts = len(np.unique(data[self.y_col_name]))

		#create 2D arrays of x,y,z
		self.xvals = np.reshape(np.asarray(data[self.x_col_name]), (num_y_pts,num_x_pts))
		self.yvals = np.reshape(np.asarray(data[self.y_col_name]), (num_y_pts,num_x_pts))
		self.zvals = np.reshape(np.asarray(data[self.z_col_name]), (num_y_pts,num_x_pts))

		return

	def hdf5_read_in(self):
		"""
		Method for reading in hdf5 files. This uses ascii class from astropy.io for flexible input. It is slower than numpy, but has greater flexibility with less input.

			Inputs: 
				file_name, x and y column names from self. 

			Return: Add x,y,z values to self.
		"""
		
		with h5py.File(WORKING_DIRECTORY + '/' + self.file_name) as f:

			#read in
			data = f['data']

			#find number of distinct x and y points.
			num_x_pts = len(np.unique(data[self.x_col_name][:]))
			num_y_pts = len(np.unique(data[self.y_col_name][:]))

			#create 2D arrays of x,y,z
			self.xvals = np.reshape(data[self.x_col_name][:],
				(num_y_pts,num_x_pts))
			self.yvals = np.reshape(data[self.y_col_name][:],
				(num_y_pts,num_x_pts))
			self.zvals = np.reshape(data[self.z_col_name][:],
				(num_y_pts,num_x_pts))

		return	


class MainProcess:
	def __init__(self, pid):

		self.pid = pid


	def input_data(self):
		"""
		Function to extract data from files according to pid. 

			Inputs:
				pid - dict - (Mandatory) dict containing all information for this code. 

				pid inputs/keys:
					plot_info - dict holding info specifically for the plots. This is converted into control_dict.

						plot_info inputs/keys:
							'limits' - dict - represents limits_dict

							'indices' - list of ints - can add first set of z values from plot of index (int) to values for this plot. 

					control - dict - containing file information for the control in raito plots.
						control_dict inputs/keys:
							name - string - file name

							or

							index - int - represents the index of a plot to compare to its first set of z values. 
		"""

		#set control_dict to plot_info part of pid.
		control_dict = self.pid['plot_info']

		ordererd = np.sort(np.asarray(list(control_dict.keys())).astype(int))

		trans_cont_dict = OrderedDict()
		for i in ordererd:
			trans_cont_dict[str(i)] = control_dict[str(i)]

		control_dict = trans_cont_dict
		
		#set empty lists for x,y,z
		x = [[]for i in np.arange(len(control_dict.keys()))]
		y = [[] for i in np.arange(len(control_dict.keys()))]
		z = [[] for i in np.arange(len(control_dict.keys()))]

		#read in base files/data
		for k, axis_string in enumerate(control_dict.keys()):
			limits_dict = {}
			if 'limits' in control_dict[axis_string].keys():
				liimits_dict = control_dict[axis_string]['limits']

			if 'file' not in control_dict[axis_string].keys():
				continue
			for j, file_dict in enumerate(control_dict[axis_string]['file']):
				data_class = ReadInData(self.pid, file_dict, limits_dict)

				x[k].append(data_class.x_append_value)
				y[k].append(data_class.y_append_value)
				z[k].append(data_class.z_append_value)

			#print(axis_string)


		#add data from plots to current plot based on index
		for k, axis_string in enumerate(control_dict.keys()):
			
			#takes first file from plot
			if 'indices' in control_dict[axis_string]:
				if type(control_dict[axis_string]['indices']) == int:
					control_dict[axis_string]['indices'] = [control_dict[axis_string]['indices']]

				for index in control_dict[axis_string]['indices']:

					index = int(index)

					x[k].append(x[index][0])
					y[k].append(y[index][0])
					z[k].append(z[index][0])
		
		#read or append control values for ratio plots
		for k, axis_string in enumerate(control_dict.keys()):
			if 'control' in control_dict[axis_string]:
				if 'name' in control_dict[axis_string]['control']:
					file_dict = control_dict[axis_string]['control']
					if 'limits' in control_dict[axis_string].keys():
						liimits_dict = control_dict[axis_string]['limits']

					data_class = ReadInData(self.pid, file_dict, limits_dict)
					x[k].append(data_class.x_append_value)
					y[k].append(data_class.y_append_value)
					z[k].append(data_class.z_append_value)

				elif 'index' in control_dict[axis_string]['control']:
					index = int(control_dict[axis_string]['control']['index'])

					x[k].append(x[index][0])
					y[k].append(y[index][0])
					z[k].append(z[index][0])

			#print(axis_string)

		#transfer lists in PlotVals class.
		value_classes = []
		for k in range(len(x)):
			value_classes.append(PlotVals(x[k],y[k],z[k]))

		self.value_classes = value_classes
		return 

	def setup_figure(self):
		#defaults for sharing axes
		sharex = True
		sharey = True

		#if share axes options are in input, change to option
		if 'sharex' in self.pid['general'].keys():
			sharex = self.pid['general']['sharex']

		if 'sharey' in self.pid['general'].keys():
			sharey = self.pid['general']['sharey']

		#declare figure and axes environments
		fig, ax = plt.subplots(nrows = int(self.pid['general']['num_rows']),
			ncols = int(self.pid['general']['num_cols']),
			sharex = sharex, sharey = sharey)

		#set figure size
		figure_width = 8
		if 'figure_width' in self.pid['general'].keys():
			figure_width = self.pid['general']['figure_width']

		figure_height = 8
		if 'figure_height' in self.pid['general'].keys():
			figure_height = self.pid['general']['figure_height']

		fig.set_size_inches(figure_width,figure_height)

		#create list of ax. Catch error if it is a sinlge plot.
		try:
			ax = ax.ravel()
		except AttributeError:
			ax = [ax]

		#create list of plot types
		plot_types = [self.pid['plot_info'][axis_string]['type']
			for axis_string in self.pid['plot_info'].keys()]

		adjust_bottom = 0.1
		if 'adjust_figure_bottom' in self.pid['general'].keys():
			adjust_bottom = self.pid['general']['adjust_figure_bottom']

		adjust_right = 0.9
		if 'Ratio' in plot_types or 'Waterfall' in plot_types:
			adjust_right = 0.79
		if 'adjust_figure_right' in self.pid['general'].keys():
			adjust_right = self.pid['general']['adjust_figure_right']

		adjust_left = 0.12
		if 'adjust_figure_left' in self.pid['general'].keys():
			adjust_left = self.pid['general']['adjust_figure_left']

		adjust_top = 0.92
		if 'adjust_figure_top' in self.pid['general'].keys():
			adjust_top = self.pid['general']['adjust_figure_top']

		adjust_wspace = 0.3
		if 'spacing' in self.pid['general'].keys():
			if self.pid['general']['spacing'] == 'wide':
				pass
			else:
				adjust_wspace = 0.0
		if 'adjust_wspace' in self.pid['general'].keys():
			adjust_wspace = self.pid['general']['adjust_wspace']

		adjust_hspace = 0.3
		if 'spacing' in self.pid['general'].keys():
			if self.pid['general']['spacing'] == 'wide':
				pass
			else:
				adjust_hspace = 0.0
		if 'adjust_hspace' in self.pid['general'].keys():
			adjust_hspace = self.pid['general']['adjust_hspace']

		#adjust figure sizes
		fig.subplots_adjust(left=adjust_left, top=adjust_top, bottom=adjust_bottom, right=adjust_right, wspace=adjust_wspace, hspace=adjust_hspace)


		#labels for figure
		fig_label_fontsize = 20
		if 'fig_label_fontsize' in self.pid['general'].keys():
			fig_label_fontsize = float(pid['general']['fig_label_fontsize'])

		if 'fig_x_label' in self.pid['general'].keys():
			fig.text(0.01, 0.51, r'%s'%(self.pid['general']['fig_y_label']),
				rotation = 'vertical', va = 'center', fontsize = fig_label_fontsize)

		if 'fig_y_label' in self.pid['general'].keys():
			fig.text(0.45, 0.02, r'%s'%(self.pid['general']['fig_x_label']),
				ha = 'center', fontsize = fig_label_fontsize)

		fig_title_fontsize = 20
		if 'fig_title_fontsize' in self.pid['general'].keys():
			fig_title_fontsize = float(self.pid['general']['fig_title_fontsize'])

		if 'fig_title' in self.pid['general'].keys():
			fig.suptitle(0.45, 0.02, r'%s'%(self.pid['general']['fig_title']),
				ha = 'center', fontsize = fig_label_fontsize)

		self.fig, self.ax = fig, ax
		return


	def create_plots(self):
		for i, axis in enumerate(self.ax):

			self.setup_trans_dict(i)
			#plot everything. First check general dict for parameters related to plots.		#create plot class

			trans_plot_class_call = globals()[self.trans_dict['type']]
			trans_plot_class = trans_plot_class_call(self.fig, axis,
				self.value_classes[i].x_arr_list,self.value_classes[i].y_arr_list,
				self.value_classes[i].z_arr_list,
				self.trans_dict['limits'], self.trans_dict['label'],
				self.trans_dict['extra'], self.trans_dict['legend'])

			#create the plot
			trans_plot_class.make_plot()

			#setup the plot
			trans_plot_class.setup_plot()

			#print("Axis", i, "Complete")

		return

	def setup_trans_dict(self, i):

		trans_dict = self.pid['plot_info'][str(i)]
		for name in ['legend', 'limits', 'label', 'extra']:
			if name not in trans_dict:
				trans_dict[name] = {}
		if 'xlims' in self.pid['general'].keys() and 'xlims' not in trans_dict['limits']:
			trans_dict['limits']['xlims'] = self.pid['general']['xlims']
		if 'dx' in self.pid['general'].keys() and 'dx' not in trans_dict['limits']:
			trans_dict['limits']['dx'] = float(self.pid['general']['dx'])
		if 'xscale' in self.pid['general'].keys() and 'xscale' not in trans_dict['limits']:
			trans_dict['limits']['xscale'] = self.pid['general']['xscale']

		if 'ylims' in self.pid['general'].keys() and 'ylims' not in trans_dict['limits']:
			trans_dict['limits']['ylims'] = self.pid['general']['ylims']
		if 'dy' in self.pid['general'].keys() and 'dy' not in trans_dict['limits']:
			trans_dict['limits']['dy'] = float(self.pid['general']['dy'])
		if 'yscale' in self.pid['general'].keys() and 'yscale' not in trans_dict['limits']:
			trans_dict['limits']['yscale'] = self.pid['general']['yscale']

		trans_dict['extra']['spacing'] = 'tight'
		if 'spacing' in self.pid['general'].keys():
			if self.pid['general']['spacing'] == 'wide':
				trans_dict['extra']['spacing'] = 'wide'

		self.trans_dict = trans_dict

		return



def plot_main(pid):

	"""
	Main function for creating these plots. Reads in plot info dict from json file. 
	"""

	global WORKING_DIRECTORY, SNR_CUT

	WORKING_DIRECTORY = '.'
	if 'WORKING_DIRECTORY' in pid['general'].keys():
		WORKING_DIRECTORY = pid['general']['WORKING_DIRECTORY']

	SNR_CUT = 5.0
	if 'SNR_CUT' in pid['general'].keys():
		SNR_CUT = pid['general']['SNR_CUT']

	if "switch_backend" in pid['general'].keys():
		plt.switch_backend(pid['general']['switch_backend'])

	running_process = MainProcess(pid)
	running_process.input_data()
	running_process.setup_figure()
	running_process.create_plots()
		

	#save or show figure
	if 'save_figure' in pid['general'].keys():
		if pid['general']['save_figure'] == True:
			running_process.fig.savefig(WORKING_DIRECTORY + '/' + 
				pid['general']['output_path'], dpi=200)
	
	if 'show_figure' in pid['general'].keys():
		if pid['general']['show_figure'] == True:
			plt.show()

	return

if __name__ == '__main__':
	"""
	starter function to read in json and pass to plot_main function. 
	"""
	#read in json
	plot_info_dict = json.load(open(sys.argv[1], 'r'),
		object_pairs_hook=OrderedDict)
	plot_main(plot_info_dict)


