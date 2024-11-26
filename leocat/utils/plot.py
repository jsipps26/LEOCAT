
import matplotlib.pyplot as plt
import os, sys
import numpy as np

from leocat.utils.const import *
from leocat.utils.time import jd_to_date, ymd_to_str
from leocat.utils.index import find_region
from leocat.utils.geodesy import RADEC_to_cart
from leocat.utils.math import unit
from leocat.utils.orbit import point_in_view

from copy import deepcopy

def plot_FOMs(Sim, FOM_type='num_obs', FOM_name=None, log_scale=False, pause=False, \
				return_fig_ax=False, vmin=None, vmax=None):
	#
	from leocat.utils.general import read_pickle
	from matplotlib.colors import LogNorm

	OUTPUT_DIR = Sim.OUTPUT_DIR
	sim_name = Sim.name
	SIM_DIR = os.path.join(OUTPUT_DIR,sim_name)
	if FOM_name is None:
		FOM_basename = 'FOM.pkl'
	else:
		FOM_basename = 'FOM_' + FOM_name + '.pkl'
	sim_file = os.path.join(SIM_DIR,'sim.pkl')
	FOM_file = os.path.join(SIM_DIR,FOM_basename)

	Sim = read_pickle(sim_file)
	FOM_data = read_pickle(FOM_file)

	FOM_type_to_zz_type = {'num_obs': 'zz', 'percent_coverage': 'zz_p', 'num_pass': 'zz_pass',
							'revisit_min': 'zz_dt_min', 'revisit_max': 'zz_dt_max',
							'revisit_avg': 'zz_dt_avg', 'revisit_count': 'zz_dt_count',
							'meas_sum': 'zz_meas_sum',
							'revisit_percent': 'zz_dt_percent'}
	#
	zz_type = FOM_type_to_zz_type[FOM_type]

	"""
	FOM_type

	num_obs
	percent_coverage
	num_pass
	revisit_max
	revisit_min

	FOM_data = {'xx': FOM.xx, 'yy': FOM.yy, 'zz': FOM.zz,
					'zz_p': FOM.zz_percent, 'zz_pass': FOM.zz_pass}
	#
	
	FOM_data['x'] = x_GP
	FOM_data['y'] = y_GP
	FOM_data['num_obs'] = z
	FOM_data['percent_coverage'] = zp
	FOM_data['num_pass'] = z_pass

	if revisit:
		z_dt_max = FOM.zz_dt_max.flatten()[bp]
		FOM_data['revisit_max'] = z_dt_max
		z_dt_min = FOM.zz_dt_min.flatten()[bp]
		FOM_data['revisit_min'] = z_dt_min

	"""

	tr_grid_ecf, tr_lla_ecf, tr_lla_grid = Sim.get_proj_transform()
	x_boundary, y_boundary = Sim.get_projection_boundary()
	ROI_global = Sim.space_params['ROI_global']
	if not ROI_global:
		ROI_poly = Sim.get_ROI_polygon('grid')

	COASTLINE_DIR = Sim.COASTLINE_DIR
	lon_coast, lat_coast = get_coastline(COASTLINE_DIR)
	x_coast, y_coast = tr_lla_grid.transform(lon_coast, lat_coast)
	r_coast = tr_lla_ecf.transform(lon_coast, lat_coast, np.zeros(lon_coast.shape))
	r_coast = np.transpose([r_coast[0], r_coast[1], r_coast[2]])

	projection = Sim.space_params['projection']
	JD1, JD2 = Sim.time_params['JD1'], Sim.time_params['JD2']
	ymd1_str, ymd2_str = get_plot_t_bounds(JD1,JD2)
	x_min, x_max = Sim.space_params['x_min'], Sim.space_params['x_max']
	y_min, y_max = Sim.space_params['y_min'], Sim.space_params['y_max']

	x, y = FOM_data['x'], FOM_data['y']
	z = FOM_data[FOM_type]
	is_LLA = projection == 'LLA'

	if not is_LLA:
		xx, yy = FOM_data['xx'], FOM_data['yy']
		zz = FOM_data[zz_type]

	title = None
	c_label = None
	if FOM_type == 'num_obs':
		title = 'Number of Observations (%s)' % projection
		c_label = r'$N$' + ' (counts)'
		if is_LLA:
			b = z > 0.0
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz[zz == 0] = np.nan

	elif FOM_type == 'percent_coverage':
		title = 'Percent Coverage (%s)' % projection
		c_label = r'$P$' + ' (%)'
		z = z*100
		if is_LLA:
			b = z > 0.0
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz = zz*100
			zz[zz == 0] = np.nan

	elif FOM_type == 'revisit_max':
		title = 'Maximum Revisit Interval (%s)' % projection
		c_label = 'Revisit Interval (hrs)'
		z = z/3600
		if is_LLA:
			# b = z > 0.0
			b = ~np.isnan(z)
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz = zz/3600
		# 	zz[zz == 0] = np.nan

	elif FOM_type == 'revisit_min':
		title = 'Minimum Revisit Interval (%s)' % projection
		c_label = 'Revisit Interval (hrs)'
		z = z/3600
		if is_LLA:
			# b = z > 0.0
			b = ~np.isnan(z)
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz = zz/3600
		# else:
		# 	zz[zz == 0] = np.nan

	elif FOM_type == 'revisit_avg':
		title = 'Average Revisit Interval (%s)' % projection
		c_label = 'Revisit Interval (hrs)'
		z = z/3600
		if is_LLA:
			# b = z > 0.0
			b = ~np.isnan(z)
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz = zz/3600
		# else:
		# 	zz[zz == 0] = np.nan

	elif FOM_type == 'revisit_count':
		title = 'Number of Revisits (%s)' % projection
		c_label = 'Revisit Count (counts)'
		if is_LLA:
			b = z > 0.0
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz[zz == 0] = np.nan

	elif FOM_type == 'revisit_percent':
		title = 'Percent Revisit Coverage (%s)' % projection
		c_label = 'Revisit Percent (%)'
		z = z*100
		if is_LLA:
			b = z > 0.0
			x, y = x[b], y[b]
			z = z[b]
		else:
			zz = zz*100
			zz[zz == 0] = np.nan

	elif FOM_type == 'meas_sum':
		title = 'Measurement Sum (%s)' % projection
		c_label = 'Sum' + ' (unit)'
		# if is_LLA:
		# 	b = z > 0.0
		# 	x, y = x[b], y[b]
		# 	z = z[b]
		# else:
		# 	zz[zz == 0] = np.nan


	fig, ax = make_fig()
	ax.plot(x_coast, y_coast, 'k,')
	if is_LLA:
		# vmin, vmax = np.nanmin(z), np.nanmax(z)
		# if vmax/vmin > 100:
		# 	log_scale = True
		if log_scale:
			im = ax.scatter(x, y, marker='o', s=2, c=z, cmap='jet',
							norm=LogNorm(vmin=vmin, vmax=vmax), edgecolors='none')
			#
		else:
			im = ax.scatter(x, y, marker='o', s=2, c=z, cmap='jet', vmin=vmin, vmax=vmax)

	else:
		ax.plot(x_boundary, y_boundary, 'k', lw=0.5, alpha=0.5)
		if not ROI_global:
			ax.plot(ROI_poly.T[0], ROI_poly.T[1], 'r--')
		im = ax.pcolormesh(xx, yy, zz, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)

	cbar = fig.colorbar(im, ax=ax, label=c_label)
	if 0:
		# failed attempt at setting colorbar log-scale
		cbar = fig.colorbar(im, ax=ax, label=c_label)
		tick_locations = cbar.get_ticks()
		custom_labels = ['%.3e' % 10**tick for tick in tick_locations]
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', UserWarning)
			cbar.set_ticklabels(custom_labels)

	ax.set_xlim(x_min,x_max)
	ax.set_ylim(y_min,y_max)
	ax.set_title('%s\n%s to %s' % (title, ymd1_str, ymd2_str))
	if is_LLA:
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')
	else:
		ax.set_xlabel('x (km)')
		ax.set_ylabel('y (km)')

	if not return_fig_ax:
		fig.show()

	if pause:
		input('enter to continue')

	if return_fig_ax:
		return fig, ax


def add_colorbar(fig, ax, im, label=None, offset=0.065):
	# Sort-of makes an independent colorbar
	bbox = ax.get_position()
	x0, y0, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
	ax = fig.add_axes((x0 + offset, y0, width, height))
	remove_axes(fig,ax)
	cbar = fig.colorbar(im, ax=ax, label=label)
	return cbar


def plot_circles(ax, x, y, s=25, edgecolors='r', *args):
	ax.scatter(x, y, s=s, facecolors='none', edgecolors=edgecolors, *args)

def plot_axes(ax, L=1.0, R=None):
	if R is None:
		R = np.eye(3)*L
	draw_vector(ax,[0,0,0],R[0],'r')
	draw_vector(ax,[0,0,0],R[1],[0,0.9,0])
	draw_vector(ax,[0,0,0],R[2],'b')


def pcolormesh(zz, xx=None, yy=None):
	fig, ax = make_fig()
	if (xx is not None) and (yy is not None):
		ax.pcolormesh(xx, yy, zz, shading='auto')
	else:
		ax.pcolormesh(zz.T, shading='auto')
	fig.show()
	
def hist(data, bins=100, check_nan=True):
	"""
	Makes a simple histogram for quick visual;
	doesn't interrupt ipython session like plt.figure
	Ex:
		import icesatPlot as ip
		ip.hist(arr, 100)
	"""
	if check_nan:
		b = np.isnan(data)
		if b.sum() > 0:
			data = data[~b]

	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	hist, bins = np.histogram(data, bins=bins)
	ax.plot(bins[:-1], hist)
	fig.show()
	
def plot(x=None, y=None, line=True):

	import matplotlib.pyplot as plt

	bx = x is not None
	by = y is not None
	if (not bx) and (not by):
		raise Exception('Either x or y must be input')

	if bx and (not by):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		if line:
			ax.plot(x)
		else:
			ax.plot(x, '.')
		fig.show()

	elif ((not bx) and by):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		if line:
			ax.plot(y)
		else:
			ax.plot(y, '.')
		fig.show()

	elif bx and by:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		if line:
			ax.plot(x, y)
		else:
			ax.plot(x, y, '.')
		fig.show()



def pro_plot():
	import matplotlib.pyplot as plt
	from matplotlib import font_manager
	fonts = set(f.name for f in font_manager.fontManager.ttflist)
	if 'Times New Roman' in fonts:
		plt.rcParams['font.family'] = 'Times New Roman'
	elif 'DejaVu Sans' in fonts:
		plt.rcParams['font.family'] = 'DejaVu Sans'
	else:
		pass # don't set fonts

	plt.rcParams['lines.linewidth'] = 1.0
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['grid.alpha'] = 0.5


def save_image(fig, name, dpi=300, format_type='png'):
	fig.savefig(os.path.join(IMAGE_DIR,'%s.%s' % (name,format_type)), bbox_inches='tight', dpi=dpi)

# def save_image(fig, name, dpi=500, format_type='svg'):
# 	fig.savefig(os.path.join(IMAGE_DIR,'%s.%s' % (name,format_type)), \
# 				format=format_type, bbox_inches='tight', dpi=dpi)
# #


def get_plot_t_bounds(JD_start, JD_end):

	t_bounds = [JD_start, JD_end]

	t_bounds_valid = 0
	if not (t_bounds is None):
		t_start, t_end = t_bounds
		t_bounds_valid = 1

	# def ymd_to_str(ymd):
	# 	ymd_str = ''
	# 	for i, val in enumerate(ymd):
	# 		ymd_str += '%s' % str(val).zfill(2)
	# 		if i < 2:
	# 			ymd_str += '-'

	# 	return ymd_str

	if t_bounds_valid:
		ymd1 = list(jd_to_date(t_start))
		ymd1[2] = int(np.round(ymd1[2]))
		if ymd1[2] > 31:
			ymd1[2] = 31
		ymd1_str = ymd_to_str(ymd1)

		ymd2 = list(jd_to_date(t_end))
		ymd2[2] = int(np.round(ymd2[2]))
		if ymd2[2] > 31:
			ymd2[2] = 31
		ymd2_str = ymd_to_str(ymd2)

		return ymd1_str, ymd2_str


def make_fig(proj='2d', figsize=None):
	import matplotlib.pyplot as plt
	proj = proj.lower()
	if figsize is not None:
		fig = plt.figure(figsize=figsize)
	else:
		fig = plt.figure()
	if proj == '2d':
		ax = fig.add_subplot(111)
	elif proj == '3d':
		ax = fig.add_subplot(111, projection='3d')

	return fig, ax

	
def get_coastline(COASTLINE_DIR=None, lon_bounds=[], lat_bounds=[]):

	try:
		import shapefile
	except ImportError:
		return np.array([]), np.array([])
		
	# from leocat.utils.const import COASTLINE_DIR
	if COASTLINE_DIR is None:
		return np.array([]), np.array([])

	lon_coast = []
	lat_coast = []

	sf = shapefile.Reader(os.path.join(COASTLINE_DIR, 'ne_10m_coastline.shp'))
	for shape in sf.shapeRecords():
		vert = np.array(shape.shape.points)
		lon, lat = vert.T[0], vert.T[1]
		lon_coast.append(lon)
		lat_coast.append(lat)

	lon_coast = np.concatenate(lon_coast)
	lat_coast = np.concatenate(lat_coast)

	b_lon = np.ones(lon_coast.shape).astype(bool)
	b_lat = np.ones(lat_coast.shape).astype(bool)
	if lon_bounds != []:
		b_lon = (lon_coast >= lon_bounds[0]) & (lon_coast <= lon_bounds[1])
	if lat_bounds != []:
		b_lat = (lat_coast >= lat_bounds[0]) & (lat_coast <= lat_bounds[1])

	b = b_lon & b_lat

	lon_coast = lon_coast[b]
	lat_coast = lat_coast[b]

	return lon_coast, lat_coast

def set_axes_equal(ax, center='auto', set_x=True, set_y=True, set_z=True):
	# https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	ax: a matplotlib axis, e.g., as output from plt.gca().
	'''
	try:
		# 3d plotting
		x_limits = ax.get_xlim3d()
		y_limits = ax.get_ylim3d()
		z_limits = ax.get_zlim3d()

		x_range = abs(x_limits[1] - x_limits[0])
		x_middle = np.mean(x_limits)
		y_range = abs(y_limits[1] - y_limits[0])
		y_middle = np.mean(y_limits)
		z_range = abs(z_limits[1] - z_limits[0])
		z_middle = np.mean(z_limits)

		# The plot bounding box is a sphere in the sense of the infinity
		# norm, hence I call half the max range the plot radius.
		plot_radius = 0.5*max([x_range, y_range, z_range])

		c = np.array([0,0,0])
		if center != 'auto':
			c = np.array(center)
			x_middle = c[0]
			y_middle = c[1]
			z_middle = c[2]

		if set_x:
			ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
		if set_y:
			ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
		if set_z:
			ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

	except AttributeError:
		# 2d plotting
		ax.axis('equal')


def load_3d():
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	

def remove_axes_main(ax):
	# https://stackoverflow.com/questions/6963035/how-to-set-common-axes-labels-for-subplots
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

def remove_axes(fig, ax):
	# https://stackoverflow.com/questions/51107968/change-3d-background-to-black-in-matplotlib
	dim = '3d'
	try:
		ax.set_zticklabels([])
	except AttributeError:
		dim = '2d'

	ax.set_xticklabels([])
	ax.set_yticklabels([])

	if dim == '3d':
		fig.set_facecolor([1,1,1])
		ax.set_facecolor([1,1,1]) 
		ax.grid(False) 
		try:
			ax.w_xaxis.pane.fill = False
			ax.w_yaxis.pane.fill = False
			ax.w_zaxis.pane.fill = False
			ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
			ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
			ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		except AttributeError:
			pass

		ax.axis('off')

	elif dim == '2d':
		fig.set_facecolor([1,1,1])
		ax.set_facecolor([1,1,1]) 
		ax.grid(False) 
		# ax.w_xaxis.pane.fill = False
		# ax.w_yaxis.pane.fill = False
		# ax.w_zaxis.pane.fill = False
		# ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		# ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		# ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		ax.axis('off')


def remove_axes_ax(ax):
	# https://stackoverflow.com/questions/51107968/change-3d-background-to-black-in-matplotlib
	dim = '3d'
	try:
		ax.set_zticklabels([])
	except AttributeError:
		dim = '2d'

	ax.set_xticklabels([])
	ax.set_yticklabels([])

	if dim == '3d':
		# fig.set_facecolor([1,1,1])
		ax.set_facecolor([1,1,1]) 
		ax.grid(False)
		try:
			ax.w_xaxis.pane.fill = False
			ax.w_yaxis.pane.fill = False
			ax.w_zaxis.pane.fill = False
			ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
			ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
			ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		except AttributeError:
			pass
		ax.axis('off')

	elif dim == '2d':
		# fig.set_facecolor([1,1,1])
		ax.set_facecolor([1,1,1]) 
		ax.grid(False) 
		# ax.w_xaxis.pane.fill = False
		# ax.w_yaxis.pane.fill = False
		# ax.w_zaxis.pane.fill = False
		# ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		# ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		# ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
		ax.axis('off')

def set_aspect_equal(ax):
	try:
		# 3D
		ax.set_box_aspect((1, 1, 1))  # aspect ratio is 1:1:1 in data space
	except TypeError:
		# 2D
		ax.set_aspect('equal')

def draw_vector(ax,p1,p2,c=None,linestyle=None,label=None,alpha=None,zorder=None,linewidth=None):
	pos_x = [p1[0], p2[0]]
	pos_y = [p1[1], p2[1]]
	try:
		# 3d
		pos_z = [p1[2], p2[2]]
		ax.plot(pos_x, pos_y, pos_z, c=c, label=label, linestyle=linestyle, alpha=alpha, zorder=zorder, linewidth=linewidth)
	except IndexError:
		# 2d
		ax.plot(pos_x, pos_y, c=c, label=label, linestyle=linestyle, alpha=alpha, zorder=zorder, linewidth=linewidth)


def draw_arc(ax, p1, p2, radius, offset=None, c=None, linestyle=None, label=None, alpha=None, zorder=None,
				arrow_length=None, angle_arrow=45.0, direction='short', N=1000):
	#
	# pos_x = [p1[0], p2[0]]
	# pos_y = [p1[1], p2[1]]
	if len(p1) != len(p2):
		raise ValueError('p1 and p2 must both be the same length')

	if N < 3:
		raise ValueError('N must be greater than or equal to 3')

	vec1 = unit(p1)
	vec2 = unit(p2)
	switch_to_2d = False
	if len(p1) == 2:
		switch_to_2d = True
		vec1 = np.array([vec1[0],vec1[1],0.0])
		vec2 = np.array([vec2[0],vec2[1],0.0])

	theta0 = np.arccos(np.dot(vec1,vec2)) # short path
	if direction != 'short':
		theta0 = theta0 - 2*np.pi
	theta = np.linspace(0,theta0,N)

	i_hat = vec1
	k_hat = unit(np.cross(vec1,vec2))
	j_hat = np.cross(k_hat,i_hat)
	pos_x = i_hat[0]*np.cos(theta) + j_hat[0]*np.sin(theta)
	pos_y = i_hat[1]*np.cos(theta) + j_hat[1]*np.sin(theta)
	pos_z = i_hat[2]*np.cos(theta) + j_hat[2]*np.sin(theta)
	pos = np.transpose([pos_x,pos_y,pos_z]) * radius
	if offset is not None:
		pos = pos + offset

	if arrow_length is not None:
		# ratio = 0.75
		# i1, i2 = int(N*ratio), N-1
		# r1, r2 = pos[i1], pos[i2]
		# r1, r2 = pos[-2], pos[-1]
		i = int(N*0.025)
		if offset is None:
			r1, r2 = pos[-(i+1)], pos[-i]
		else:
			r1, r2 = pos[-(i+1)]-offset, pos[-i]-offset
		# L_c = radius * theta0 # arc length
		# L_a = arrow_length # L_c * arrowhead # arrowhead length

		phi_arrow = np.pi - np.radians(angle_arrow/2)
		v_hat = unit(r2-r1) # direction of motion at end of arc
		r_hat = unit(r2)
		a1_hat = v_hat*np.cos(phi_arrow) + r_hat*np.sin(phi_arrow)
		a2_hat = v_hat*np.cos(-phi_arrow) + r_hat*np.sin(-phi_arrow)


	if switch_to_2d:
		im = ax.plot(pos.T[0], pos.T[1], c=c, label=label, linestyle=linestyle, alpha=alpha)
		if arrow_length is not None:
			vec = pos[-1]
			vec_2d = np.array([vec[0],vec[1]])
			a1_hat_2d = np.array([a1_hat[0],a1_hat[1]])
			a2_hat_2d = np.array([a2_hat[0],a2_hat[1]])
			draw_vector(ax, vec_2d, vec_2d + a1_hat_2d*arrow_length, \
							c=im[0].get_c(), alpha=im[0].get_alpha())
			#
			draw_vector(ax, vec_2d, vec_2d + a2_hat_2d*arrow_length, \
							c=im[0].get_c(), alpha=im[0].get_alpha())
			#

	else:
		im = ax.plot(pos.T[0], pos.T[1], pos.T[2], c=c, label=label, linestyle=linestyle, alpha=alpha)
		if arrow_length is not None:
			vec = pos[-1]
			draw_vector(ax, vec, vec + a1_hat*arrow_length, c=im[0].get_c(), alpha=im[0].get_alpha())
			draw_vector(ax, vec, vec + a2_hat*arrow_length, c=im[0].get_c(), alpha=im[0].get_alpha())





# def make_gif(name, fps, IMAGE_DIR, OUT_DIR, include_last=True):
# 	import imageio # imageio==2.9.0
# 	VALID_EXTENSIONS = ('png', 'jpg')
# 	def make_gif_func(filenames, duration, ofile):
# 		images = []
# 		for filename in filenames:
# 			images.append(imageio.imread(filename))
# 		output_file = ofile
# 		imageio.mimwrite(output_file, images, duration=duration, subrectangles=True)

# 	# fps = 5

# 	files = []
# 	for file in os.listdir(IMAGE_DIR):
# 		if file.endswith('.png'):
# 			files.append(IMAGE_DIR + '/' + file)

# 	if not include_last:
# 		files = files[:-1]

# 	make_gif_func(files, 1.0 / fps, os.path.join(OUT_DIR, '%s.gif' % name))

def split_lon(lon):
	idx = np.where(np.abs(np.diff(lon)) > 180)[0].astype(int)
	split_index = []
	i_prev = 0
	for i in idx:
		split_index.append(np.arange(i_prev,i))
		i_prev = i
	split_index.append(np.arange(i_prev+1,len(lon)))
	return split_index


def circular_edge(cam_vec, radius):
	theta = np.linspace(0,2*np.pi,500)
	cy = np.cos(theta)
	cz = np.sin(theta)
	circle = np.transpose([np.zeros(theta.shape), cy, cz])
	x_hat = cam_vec
	z_hat = np.array([0,0,1])
	y_hat = unit(np.cross(z_hat,x_hat))
	z_hat = np.cross(x_hat,y_hat)
	Rc = np.array([x_hat,y_hat,z_hat])
	circle = circle @ Rc
	return radius*circle

def split_index(r, cam_vec):
	b = point_in_view(r, cam_vec)
	index = np.where(b)[0].astype(int)
	index_split, _ = find_region(index, 1, 0)
	return index_split

def split_line(r, index_split):
	r_plot_list = []
	for j,index0 in enumerate(index_split):
		i1, i2 = index0[0], index0[1] #-1
		r_plot_list.append(r[i1:i2+1])
	return r_plot_list

def get_wireframe_sphere(a, b, cam_vec=None, Nx=100, Ny=100):

	u = np.linspace(0, 2 * np.pi, Nx)
	v = np.linspace(0, np.pi, Ny)
	X = a * np.outer(np.cos(u), np.sin(v))
	Y = a * np.outer(np.sin(u), np.sin(v))
	Z = b * np.outer(np.ones(np.size(u)), np.cos(v))

	if cam_vec is not None:
		x = X.flatten()
		y = Y.flatten()
		z = Z.flatten()
		bo = np.zeros(x.shape).astype(bool)
		proj = cam_vec[0]*x + cam_vec[1]*y + cam_vec[2]*z
		bo[proj < 0] = True
		B = bo.reshape(X.shape)
		X[B] = np.nan
		Y[B] = np.nan
		Z[B] = np.nan

	# ax.plot_wireframe(X, Y, Z, color='r', alpha=0.25)
	return X, Y, Z


def plot_sim(lines, RA, DEC, zoom=1.0, target=np.array([0,0,0]), alpha_edge=1.0, figsize=None):

	cam_vec = RADEC_to_cart(RA,DEC)
	L = R_earth*0.75 / zoom

	edge = circular_edge(cam_vec, R_earth)

	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
	for line in lines:
		if type(line) == list:
			line = line[0]
			x_data, y_data, z_data = line.get_data_3d()
			r = np.transpose([x_data,y_data,z_data])
			ls = line.get_linestyle()
			m = line.get_marker()
			c = line.get_color()
			ms = line.get_markersize()
			zorder = line.get_zorder()
			alpha = line.get_alpha()
			if m == 'None':
				index_split = split_index(r,cam_vec)
				r_plot_list = split_line(r,index_split)
				for r_plot in r_plot_list:
					ax.plot(r_plot.T[0], r_plot.T[1], r_plot.T[2], ls=ls, c=c, zorder=zorder, alpha=alpha)
				# if len(index_split) > 1:
				# else:
				# 	ax.plot(r.T[0], r.T[1], r.T[2], ls=ls, c=c, zorder=zorder)
			else:
				b = point_in_view(r, cam_vec)
				ax.plot(r[b].T[0], r[b].T[1], r[b].T[2], marker=m, ms=ms, c=c, ls='', zorder=zorder, alpha=alpha)

		# else:
		# 	# assuming Path3DCollection object
			

	ax.plot(edge.T[0], edge.T[1], edge.T[2], 'k', alpha=alpha_edge)
	set_aspect_equal(ax)
	remove_axes(fig, ax)
	ax.view_init(azim=RA, elev=DEC)

	ax.set_xlim(target[0]-L,target[0]+L)
	ax.set_ylim(target[1]-L,target[1]+L)
	ax.set_zlim(target[2]-L,target[2]+L)

	# fig.show()
	return fig, ax


def plot_sim_subplot(ax, lines, RA, DEC, zoom=1.0, target=np.array([0,0,0])):

	cam_vec = RADEC_to_cart(RA,DEC)
	L = R_earth*0.75 / zoom

	edge = circular_edge(cam_vec, R_earth)
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

	for line in lines:
		# line = deepcopy(line[0])
		line = line[0]
		x_data, y_data, z_data = line.get_data_3d()
		r = np.transpose([x_data,y_data,z_data])
		ls = line.get_linestyle()
		m = line.get_marker()
		c = line.get_color()
		ms = line.get_markersize()
		zorder = line.get_zorder()
		alpha = line.get_alpha()
		if m == 'None':
			index_split = split_index(r,cam_vec)
			r_plot_list = split_line(r,index_split)
			for r_plot in r_plot_list:
				ax.plot(r_plot.T[0], r_plot.T[1], r_plot.T[2], ls=ls, c=c, zorder=zorder, alpha=alpha)
			# if len(index_split) > 1:
			# else:
			# 	ax.plot(r.T[0], r.T[1], r.T[2], ls=ls, c=c, zorder=zorder)
		else:
			b = point_in_view(r, cam_vec)
			ax.plot(r[b].T[0], r[b].T[1], r[b].T[2], marker=m, ms=ms, c=c, ls='', zorder=zorder, alpha=alpha)

	ax.plot(edge.T[0], edge.T[1], edge.T[2], 'k')
	set_aspect_equal(ax)
	# remove_axes(fig, ax)
	remove_axes_ax(ax)
	ax.view_init(azim=RA, elev=DEC)

	ax.set_xlim(target[0]-L,target[0]+L)
	ax.set_ylim(target[1]-L,target[1]+L)
	ax.set_zlim(target[2]-L,target[2]+L)

	# fig.show()
	# return ax
	