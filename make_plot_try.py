import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct
import pdb
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as units
import scipy.constants as ct
import time
from matplotlib  import cm
import math as mh
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from operator import itemgetter, attrgetter
from scipy.interpolate import interp1d, interp2d
from cycler import cycler
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from collections import OrderedDict

#plt.switch_backend('Agg')

class PlotVals:
	def __init__(self, x_arr_list, y_arr_list, z_arr_list):
		self.x_arr_list, self.y_arr_list, self.z_arr_list = x_arr_list, y_arr_list, z_arr_list

	def return_x_list(self):
		return self.x_arr_list

	def return_y_list(self):
		return self.y_arr_list

	def return_z_list(self):
		return self.z_arr_list

def find_difference_contour(zout, control_zout):
	#set indices of loss,gained. Also set rid for when neither curve measures source. inds_check tells the ratio calculator not to control_zout if both SNRs are below 1

	inds_gained = np.where((zout>=SNR_CUT) & (control_zout< SNR_CUT))
	inds_lost = np.where((zout<SNR_CUT) & (control_zout>=SNR_CUT))
	inds_rid = np.where((zout<1.0) & (control_zout<1.0))
	inds_check = np.where((zout.ravel()<1.0) & (control_zout.ravel()<1.0))[0]

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

	loss_gain_contour = np.zeros(np.shape(zout))

	j = -1
	for i in (inds_lost, inds_gained):
		loss_gain_contour[i] = j
		j += 2

	return diffout2, loss_gain_contour

class CreateSinglePlot:
	def __init__(self, fig, axis, xvals,yvals,zvals, limits_dict={}, label_dict={}, legend_dict={}, extra_dict={}):
		self.fig = fig
		self.axis = axis
		self.xvals = xvals
		self.yvals = yvals
		self.zvals = zvals

		self.limits_dict, self.label_dict, self.legend_dict, self.extra_dict = limits_dict, label_dict, legend_dict, extra_dict

	def ratio(self):
		#sets colormap for ratio comparison plot
		cmap2 = cm.seismic
		#set attributes of ratio comparison contour
		normval2 = 2.0
		num_contours2 = 40 #must be even
		levels2 = np.linspace(-normval2, normval2,num_contours2)
		norm2 = colors.Normalize(-normval2, normval2)

		#initialize dimensions of loss/gain contour


		diffout2, loss_gain_contour = find_difference_contour(self.zvals[0], self.zvals[1])

		#plot ratio contours
		sc3=self.axis.contourf(self.xvals[0],self.yvals[0],diffout2, levels = levels2, norm=norm2, extend='both', cmap=cmap2)

		#toggle line contours of orders of magnitude of ratio comparisons
		#ax[w].contour(xout,yout,Diffout2, np.array([-2.0, -1.0,1.0, 2.0]), norm=norm2, extend='both', colors = 'grey', linewidths = 1.0, linestyles= 'dashed')

		#custom way to get loss/gain to be -1 for loss and +1 for gain

		self.axis.contour(self.xvals[0],self.yvals[0],loss_gain_contour,1, colors = 'grey', linewidths = 2)
		#axis.set_ylim(float(pid['y_min'][0]), float(pid['y_max'][0]))

		#establish colorbar and labels for ratio comp contour plot

		cbar_ax2 = self.fig.add_axes([0.83, 0.1, 0.03, 0.4])
		self.fig.colorbar(sc3, cax=cbar_ax2,ticks=np.array([-3.0,-2.0,-1.0,0.0, 1.0,2.0, 3.0]))
		cbar_ax2.set_yticklabels([r'$10^{%i}$'%i for i in np.arange(-normval2, normval2+1.0, 1.0)], fontsize = 17)
		cbar_ax2.set_ylabel(r"$\rho_i/\rho_0$", fontsize = 20)

		"""
		if ('comparison', 'labels') in control_dict[axis_string].keys():
			fontsize=16
			if ('comparison', 'labels', 'fontsize') in control_dict[axis_string].keys():
				fontsize = control_dict[axis_string][('comparison', 'labels','fontsize')]
			axis.set_title(r'%s vs. %s'%(control_dict[axis_string][('comparison', 'labels')][0].replace('*',' '), control_dict[axis_string][('comparison', 'labels')][1].replace('*',' ')), fontsize=fontsize)
		"""
		return

	def waterfall(self):
		#sets levels of main contour plot
		colors1 = ['None','darkblue', 'blue', 'deepskyblue', 'aqua','greenyellow', 'orange', 'red','darkred']
		levels = np.array([0.,10,20,50,100,200,500,1000,3000,1e6])
		
		#produce filled contour of SNR vs. z vs. Mtotal
		sc=self.axis.contourf(self.xvals[0],self.yvals[0],self.zvals[0], levels = levels, colors=colors1)

		#add colorbar axes for both contour plots
		cbar_ax = self.fig.add_axes([0.83, 0.55, 0.03, 0.4])

			#establish colorbar and labels for main contour plot
		self.fig.colorbar(sc, cax=cbar_ax, ticks=levels)
		cbar_ax.set_yticklabels([int(i) for i in np.delete(levels,-1)], fontsize = 17)
		cbar_ax.set_ylabel(r"$\rho_i$", fontsize = 20)
		return

	def horizon(self):
		#sets levels of main contour plot
		colors1 = ['blue', 'green', 'red','purple', 'orange', 'gold','magenta']

		contour_val = SNR_CUT

		if ('extra', 'snr', 'contour', 'value') in self.extra_dict.keys():
			contour_val = float(self.extra_dict[('extra', 'snr', 'contour', 'value')])
		
		for j in range(len(self.zvals)):
			hz = self.axis.contour(self.xvals[j],self.yvals[j],self.zvals[j], np.array([contour_val]), colors = colors1[j], linewidths = 1., linestyles= 'solid')

			"""
			v = np.transpose(hz.collections[0].get_paths()[0].vertices)

			#plot invisible lines for purpose of creating a legend
			if ('legend', 'labels') in control_dict[axis_string].keys():
				axis.plot([0.1,0.2],[0.1,0.2],color = colors1[j], label = control_dict[axis_string][('legend', 'labels')][j].replace('*', ' '))

			else:
				axis.plot([0.1,0.2],[0.1,0.2],color = colors1[j])
			"""
		"""
		if ('legend', 'labels') in control_dict[axis_string].keys():
			#defaults followed by change
			loc = 'upper left'
			if ('legend', 'loc') in control_dict[axis_string].keys():
				loc = control_dict[axis_string][('legend','loc')].replace('*', ' ')

			size = 10
			if ('legend','size') in control_dict[axis_string].keys():
				size = int(control_dict[axis_string][('legend','size')])

			bbox_to_anchor = None
			if ('legend', 'bbox', 'to', 'anchor') in control_dict[axis_string].keys():
				bbox_to_anchor = tuple(float(i) for i in control_dict[axis_string][('legend', 'bbox', 'to', 'anchor')])

			ncol = 1
			if ('legend','ncol') in control_dict[axis_string].keys():
				ncol = int(control_dict[axis_string][('legend','ncol')])	

			axis.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, prop={'size':size})

			if ('axis', 'xlabel') in control_dict[axis_string].keys():
				axis.set_xlabel(r'%s'%(control_dict[axis_string][('axis', 'xlabel')].replace('*',' ')))

			if ('axis', 'ylabel') in control_dict[axis_string].keys():
				axis.set_xlabel(r'%s'%(control_dict[axis_string][('axis', 'ylabel')].replace('*',' ')))
		"""

		return

	def setup_plot(self):

		xticks = np.arange(self.limits_dict[('limits','xlims')][0], self.limits_dict[('limits','xlims')][1] + self.limits_dict[('limits','dx')], self.limits_dict[('limits','dx')])

		self.axis.set_xlim(xticks.min(), xticks.max())
		self.axis.set_xticks(xticks[1:-1])
		self.axis.set_xticklabels([r'$10^{%i}$'%i for i in xticks[1:-1]])

		yticks = np.arange(self.limits_dict[('limits','ylims')][0], self.limits_dict[('limits','ylims')][1] + self.limits_dict[('limits','dy')], self.limits_dict[('limits','dy')])

		self.axis.set_ylim(yticks.min(), yticks.max())
		self.axis.set_yticks(yticks[1:-1])

		if self.limits_dict[('limits', 'yscale')] == 'log':
			self.axis.set_yticklabels([r'$10^{%i}$'%int(i) for i in yticks[1:-1]])
		else:
			self.axis.set_yticklabels([r'$%i$'%int(i) for i in yticks[1:-1]])

		self.axis.grid(True,linestyle='-',color='0.75')

		title_fontsize = 20
		if ('label', 'title') in self.label_dict.keys():
			if ('label', 'title', 'fontsize') in self.label_dict.keys():
				title_fontsize = float(self.label_dict[('label', 'title', 'fontsize')])
			self.axis.set_title(r'%s'%self.label_dict[('label', 'title')].replace('*',' '), fontsize=title_fontsize)

		return


def compile_plot_information(ax, pid):
	control_dict = OrderedDict()
	keys_for_scalar_entry = ['type', ('control', 'name'), ('control', 'label'), ('control', 'index'), ('legend','loc'), ('legend','size'), ('legend','ncol'), ('label','xlabel'), ('label', 'title'), ('label', 'ylabel'), ('label', 'title', 'fontsize'), ('label', 'xlabel', 'fontsize'), ('label', 'ylabel', 'fontsize'), ('limits','dx'), ('limits','dy'), ('limits', 'yscale'), ('extra','snr','contour','value')]

	for i in range(len(ax)):
		control_dict[str(i)] = {}

	global SNR_CUT
	#establish SNR cut
	SNR_CUT = float(pid['SNR_CUT'][0])

	for key_name in pid.keys():
		if key_name[0:5] == 'plot_':

			axis_string = key_name.split('_')[1]
			names = key_name.split('_')[2::]
			if len(names) > 1:
				key = tuple(name for name in names)

				if key in keys_for_scalar_entry:
					control_dict[axis_string][key] = pid[key_name][0]
				else:
					control_dict[axis_string][key] = pid[key_name]
			elif key_name.split('_')[2] in keys_for_scalar_entry:
				control_dict[axis_string][key_name.split('_')[2]] = pid[key_name][0]

			else:
				control_dict[axis_string][key_name.split('_')[2]] = pid[key_name]

	return control_dict

def read_in_data(control_dict, pid):
	x = [[]for i in np.arange(len(control_dict.keys()))]
	y = [[] for i in np.arange(len(control_dict.keys()))]
	z = [[] for i in np.arange(len(control_dict.keys()))]

	for k, axis_string in enumerate(control_dict.keys()):

		for j, f1 in enumerate(control_dict[axis_string][('file', 'names')]):
	
			data = np.genfromtxt(WORKING_DIRECTORY + '/' + f1, names=True, skip_header=2)

			f = open(f1, 'r')

			#find dimensions of data
			line = f.readline()
			line = line.replace('\n','')
			i=0
			while line[i] != '=':
				i+= 1
			i += 1
			#set Mass points
			num_M_pts = int(line[i::])


			line = f.readline()
			line = line.replace('\n','')
			i=0
			while line[i] != '=':
				i+= 1
			i += 1
			#set redshift points
			num_z_pts = int(line[i::])

			x[k].append(np.log10(np.reshape(data['M_s'], (num_z_pts,num_M_pts))))

			if ('limits', 'yscale') in control_dict[axis_string].keys():
				if control_dict[axis_string][('limits', 'yscale')] == 'lin':
					y[k].append(np.reshape(data['z'], (num_z_pts,num_M_pts)))
				else:
					y[k].append(np.log10(np.reshape(data['z'], (num_z_pts,num_M_pts))))

			else:
				if pid['gen_yscale'][0] == 'lin':
					y[k].append(np.reshape(data['z'], (num_z_pts,num_M_pts)))
				else:
					y[k].append(np.log10(np.reshape(data['z'], (num_z_pts,num_M_pts))))

			z[k].append(np.reshape(data[control_dict[axis_string][('file', 'labels')][j]], (num_z_pts,num_M_pts)))

	for k, axis_string in enumerate(control_dict.keys()):
		if ('control', 'name') in control_dict[axis_string]:

			data = np.genfromtxt(WORKING_DIRECTORY + '/' + control_dict[axis_string][('control', 'name')], names=True, skip_header=2)

			f = open(f1, 'r')

			#find dimensions of data
			line = f.readline()
			line = line.replace('\n','')
			i=0
			while line[i] != '=':
				i+= 1
			i += 1
			#set Mass points
			num_M_pts = int(line[i::])


			line = f.readline()
			line = line.replace('\n','')
			i=0
			while line[i] != '=':
				i+= 1
			i += 1
			#set redshift points
			num_z_pts = int(line[i::])

			x[k].append(np.log10(np.reshape(data['M_s'], (num_z_pts,num_M_pts))))

			if ('limits', 'yscale') in control_dict[axis_string].keys():
				if control_dict[axis_string][('limits', 'yscale')] == 'lin':
					y[k].append(np.reshape(data['z'], (num_z_pts,num_M_pts)))
				else:
					y[k].append(np.log10(np.reshape(data['z'], (num_z_pts,num_M_pts))))

			else:
				if pid['gen_yscale'][0] == 'lin':
					y[k].append(np.reshape(data['z'], (num_z_pts,num_M_pts)))
				else:
					y[k].append(np.log10(np.reshape(data['z'], (num_z_pts,num_M_pts))))


			z[k].append(np.reshape(data[control_dict[axis_string][('control', 'label')]], (num_z_pts,num_M_pts)))

		if ('control', 'index') in control_dict[axis_string]:
			index = int(control_dict[axis_string][('control', 'index')])
			x[k].append(x[index][0])

			y[k].append(y[index][0])

			z[k].append(z[index][0])




	for k, axis_string in enumerate(control_dict.keys()):
		
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
"""
def change_axis_settings(ax, control_dict):
	for axis_string in control_dict:
		for key in control_dict[axis_string].keys():
			if key[0] != 'axis':
				continue

			if 'xlabel' in key:
				fontsize = 16
				if 'fontsize' in key:
					fontsize = float(control_dict[axis_string][('axis', 'xlabel', 'fontsize')])
				ax[int(axis_string)].set_xlabel(r'%s'%control_dict[axis_string][('axis', 'xlabel')].replace('*',' '), fontsize = fontsize)

			if 'ylabel' in key:
				fontsize = 16
				if 'fontsize' in key:
					fontsize = float(control_dict[axis_string][('axis', 'ylabel', 'fontsize')])
				ax[int(axis_string)].set_ylabel(r'%s'%control_dict[axis_string][('axis', 'ylabel')].replace('*',' '), fontsize = fontsize)
"""

def make_plot(pid):

	global WORKING_DIRECTORY

	WORKING_DIRECTORY = pid['WORKING_DIRECTORY'][0]

	#set up figure environment
	t_or_f_dict = {'True':True, 'False':False}

	sharex = True
	sharey = True

	if 'sharex' in pid.keys():
		sharex = t_or_f_dict[pid['sharex'][0]]

	if 'sharey' in pid.keys():
		sharey = t_or_f_dict[pid['sharey'][0]]

	fig, ax = plt.subplots(nrows = int(pid['num_rows'][0]), ncols = int(pid['num_cols'][0]), sharex = sharex, sharey = sharey)

	fig.set_size_inches(float(pid['figure_width'][0]),float(pid['figure_height'][0]))
	ax = ax.ravel()

	control_dict = compile_plot_information(ax, pid)
	plot_data = read_in_data(control_dict, pid)

	for i, axis in enumerate(ax):
		legend_dict = {}
		label_dict = {}
		limits_dict = {}
		extra_dict = {}

		if 'gen_xlims' in pid.keys():
			limits_dict[('limits','xlims')] = [float(val) for val in pid['gen_xlims']]
		if 'gen_dx' in pid.keys():
			limits_dict[('limits','dx')] = float(pid['gen_dx'][0])

		if 'gen_ylims' in pid.keys():
			limits_dict[('limits','ylims')] = [float(val) for val in pid['gen_ylims']]
		if 'gen_dy' in pid.keys():
			limits_dict[('limits','dy')] = float(pid['gen_dy'][0])
		if 'gen_yscale' in pid.keys():
			limits_dict[('limits', 'yscale')] = pid['gen_yscale']


		for key in control_dict[str(i)]:
			if key[0] == 'legend':
				legend_dict[key] = control_dict[str(i)][key]
			if key[0] == 'label':
				label_dict[key] = control_dict[str(i)][key]
			if key[0] == 'limits':
				limits_dict[key] = control_dict[str(i)][key]
			if key[0] == 'extra':
				extra_dict[key] = control_dict[str(i)][key]


		trans_plot_class = CreateSinglePlot(fig, axis, plot_data[i].return_x_list(),plot_data[i].return_y_list(), plot_data[i].return_z_list(), limits_dict, label_dict, legend_dict, extra_dict)

		getattr(trans_plot_class, control_dict[str(i)]['type'])()

		trans_plot_class.setup_plot()

	fig.subplots_adjust(left=0.12, top=0.92, bottom=0.1)

	if 'gen_spacing' in pid.keys():
		if pid['gen_spacing'][0] == 'wide':
			pass

		else:
			fig.subplots_adjust(wspace=0.0, hspace=0.0)

	else:
		fig.subplots_adjust(wspace=0.0, hspace=0.0)


	adjusted = False

	for axis_string in control_dict.keys():
		if adjusted == True:
			continue
		else:
			if control_dict[axis_string]['type'] == 'ratio' or control_dict[axis_string]['type'] == 'waterfall':
				fig.subplots_adjust(right=0.79)
				adjusted = True

	#label for axis
	fig_label_fontsize = 20
	if 'fig_label_fontsize' in pid.keys():
		fig_label_fontsize = float(pid['fig_label_fontsize'])

	fig.text(0.01, 0.51, r'%s'%(pid['fig_y_label'][0].replace('*',' ')), rotation = 'vertical', va = 'center', fontsize = fig_label_fontsize)

	fig.text(0.45, 0.02, r'%s'%(pid['fig_x_label'][0].replace('*',' ')), ha = 'center', fontsize = fig_label_fontsize)
		

	if 'save_figure' in pid.keys():
		if t_or_f_dict[pid['save_figure'][0]] == True:
			plt.savefig(WORKING_DIRECTORY + '/' + pid['output_path'][0], dpi=200)
	
	if 'show_figure' in pid.keys():
		if t_or_f_dict[pid['show_figure'][0]] == True:
			plt.show()



if __name__ == '__main__':
	f = open('make_plot_input_try.txt', 'r')
	lines = f.readlines()
	lines = [line for line in lines if line[0]!= '#']
	lines = [line for line in lines if line[0]!= '\n']

	plot_info_dict = OrderedDict()
	for line in lines:
		if ':' in line:
			plot_info_dict[line.split()[0][0:-1]] = line.split()[1::]


	make_plot(plot_info_dict)


