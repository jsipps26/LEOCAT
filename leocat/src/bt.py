

import numpy as np
import os,sys

from leocat.utils.const import *
from leocat.utils.math import unit, mag, dot, wrap, rad, deg, angle_in_region, interp
from leocat.utils.orbit import get_GMST, M2nu, nu2M, convert_ECI_ECF
from leocat.utils.geodesy import cart_to_RADEC, RADEC_to_cart
from leocat.utils.index import hash_index, unique_index
from leocat.utils.general import pause

from pandas import DataFrame

import warnings
from copy import deepcopy

from numba import njit, types, typed


class AnalyticCoverage:
	"""
	To do's
	By default, eliminate lat points outside swath extents

	"""
	def __init__(self, orb, swath, lon, lat, JD1, JD2):
		self.orb = orb
		self.swath = swath
		self.lon = lon
		self.lat = lat
		self.JD1 = JD1
		self.JD2 = JD2

	def get_access(self, verbose=1, GL=None, accuracy=2):
		orb = self.orb
		swath = self.swath
		lon, lat = self.lon, self.lat
		JD1, JD2 = self.JD1, self.JD2
		GL_flag = 1
		if GL is None:
			GL_flag = 0

		propagator = orb.propagator
		if not (propagator == 'kepler' or propagator == 'SPE+frozen'):
			import warnings
			warnings.warn('AnalyticCoverage only accurate for kepler or SPE+frozen propagators')

		t_access_init_total, index_total = \
			BT_coverage_init(orb, swath, lon, lat, JD1, JD2, verbose=verbose)
		#
		t_BT, q_BT = BT_coverage(t_access_init_total, index_total, \
								orb, lon, lat, JD1, JD2, accuracy=accuracy, GL=GL)
		#
		if GL_flag:
			return t_BT, q_BT
		else:
			return t_BT



def BT_coverage_init(orb, swath, lon, lat, JD1, JD2, verbose=0):

	# sidereal_days = (JD2-JD1)*86400/86164
	# k0_range = np.arange(-1,int(np.ceil(sidereal_days))+1)
	Dn = orb.get_nodal_day()
	max_days = int(np.ceil((JD2-JD1)*86400/Dn))
	k0_range = np.arange(-1,max_days+1)

	LAN_dot, omega_dot, M_dot = orb.get_LAN_dot(), orb.get_omega_dot(), orb.get_M_dot()
	Tn = orb.get_period('nodal')
	# Dn = orb.get_nodal_day()
	dlon = -2*np.pi*Tn/Dn * 180/np.pi
	t_pole_N, _ = calc_t_u(np.pi/2,orb)
	t_pole_S, _ = calc_t_u(3*np.pi/2,orb)

	# lat_GT_max = np.degrees(inc)
	# if lat_GT_max > 90:
	# 	lat_GT_max = 180-lat_GT_max
	# #
	# dist_lat = (np.abs(lat)-lat_GT_max)*np.pi/180 * R_earth
	# b_lat = dist_lat < (swath*1.2)/2 # valid latitudes, i.e. within swath

	df = DataFrame({'lat': lat})
	lat_data = df.groupby('lat', sort=False).indices
	keys = list(lat_data.keys())
	# for j,lat0 in enumerate(lat_data):

	t_access_init_total = []
	index_total = []

	iterator = range(len(keys))
	if verbose:
		from tqdm import tqdm
		iterator = tqdm(iterator)

	# time1 = time.perf_counter()
	# t_BT = {}
	# for j in tqdm(range(len(keys))):
	for j in iterator:
		# if not b_lat[j]:
		# 	continue
		lat0 = keys[j]
		idx = lat_data[lat0]
		lon_vec = lon[idx]
		# p_vec = p[idx]

		lons0, us0, ts0, split, invalid_left, invalid_right, lat_in_bounds, pole_in_view = \
			get_swath_params(orb, lat0, swath, JD1, verbose=0)
		#
		bridge_class1, bridge_class2, lons0, us0, ts0 = \
			classify_bridges(lons0, us0, ts0, orb, lat0, split, invalid_left, invalid_right, lat_in_bounds)
		#
		t_r1, t_l1, t_r2, t_l2 = ts0['r1'], ts0['l1'], ts0['r2'], ts0['l2']
		lon_r1, lon_l1, lon_r2, lon_l2 = lons0['r1'], lons0['l1'], lons0['r2'], lons0['l2']
		# time2 = time.perf_counter()

		for jj in range(len(lon_vec)):
			k_access1, k_access2 = get_k_access(lon_vec[jj], lons0, k0_range, dlon, bridge_class1, bridge_class2)
			t_access_init = get_init_t_access(t_r1, t_l1, t_r2, t_l2, lon_r1, lon_l1, lon_r2, lon_l2, invalid_left, \
							invalid_right, lat_in_bounds, pole_in_view, bridge_class1, bridge_class2, lon_vec[jj], \
							lat0, orb.inc, dlon, Tn, k_access1, k_access2, JD1, t_pole_N, t_pole_S, orb.LAN, orb.omega, orb.M0, \
							LAN_dot, omega_dot, M_dot)
			#
			# t_access_init = get_init_t_access_numba(t_r1, t_l1, t_r2, t_l2, lon_r1, lon_l1, lon_r2, lon_l2, invalid_left, \
			# 				invalid_right, lat_in_bounds, pole_in_view, bridge_class1, bridge_class2, lon_vec[jj], \
			# 				lat0, orb.inc, dlon, Tn, k_access1, k_access2, JD1, t_pole_N, t_pole_S, orb.LAN, orb.omega, orb.M0, \
			# 				LAN_dot, omega_dot, M_dot)
			# #
			if len(t_access_init) == 0:
				continue

			t_access_init_total.append(t_access_init)
			index_total.append(np.full(len(t_access_init),idx[jj]))
			# pause()

		# pause()
		# time3 = time.perf_counter()
		# print(time3-time2)


	# time4 = time.perf_counter()
	# print(time4-time1)
	return t_access_init_total, index_total




def BT_coverage(t_access_init_total, index_total, orb, lon, lat, JD1, JD2, accuracy=0, GL=None):

	t_access_init_total = np.concatenate(t_access_init_total)
	index_total = np.concatenate(index_total)

	GL_flag = 1
	if GL is None:
		GL_flag = 0

	if accuracy >= 1:
		dt_sc = 60.0
		num = int((JD2-JD1)*86400/dt_sc) + 1
		t_space = np.linspace(0,(JD2-JD1)*86400,num)
		r_eci_sc, v_eci_sc = orb.propagate(t_space)
		r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD1 + t_space/86400, r_eci_sc, v_eci_sc)
		p = R_earth * RADEC_to_cart(lon, lat)

		# r0, v0 = orb.propagate(t_access_init_total)
		# r0, v0 = convert_ECI_ECF(JD1 + t_access_init_total/86400, r0, v0)
		r0 = interp(t_access_init_total, t_space, r_ecf_sc)
		v0 = interp(t_access_init_total, t_space, v_ecf_sc)
		dt = dot(v0,p[index_total]-r0) / mag(v0)**2
		t_access_init_total = t_access_init_total + dt

		if accuracy == 2:
			J = J_NR(t_access_init_total, orb, np.radians(lon[index_total]), np.radians(lat[index_total]), JD1)
			H = H_NR(t_access_init_total, orb, np.radians(lon[index_total]), np.radians(lat[index_total]), JD1)
			t_access_init_total = t_access_init_total - J/H


	b = (t_access_init_total >= 0.0) & (t_access_init_total < (0.0 + (JD2-JD1)*86400))
	t_access_init_total = t_access_init_total[b]
	index_total = index_total[b]

	q_total = None
	if GL_flag:
		JD = JD1 + t_access_init_total/86400
		q_total = GL.get_quality(lon[index_total], lat[index_total], JD)

	t_BT = {}
	q_BT = {}
	if b.sum() > 0:
		df = DataFrame({'index': index_total})
		index_indices = df.groupby('index').indices
		for key in index_indices:
			idx = index_indices[key]
			t_key = t_access_init_total[idx]
			j_sort = np.argsort(t_key)
			idx = idx[j_sort]
			t_BT[key] = t_access_init_total[idx]
			# t_BT[key] = np.sort(t_access_init_total[idx])
			if GL_flag:
				q_BT[key] = q_total[idx]

	return t_BT, q_BT
	# if not GL_flag:
	# 	return t_BT
	# else:



def J_NR(t, orb, lam, phi, JD1):
	inc = orb.inc
	M = orb.M0 + orb.get_M_dot()*(t-orb.t0)
	nu = M2nu(M,orb.e)
	omega = orb.omega + orb.get_omega_dot()*(t-orb.t0)
	u = omega + nu
	LAN = orb.LAN + orb.get_LAN_dot()*(t-orb.t0)
	Lam = lam + rad(get_GMST(t/86400 + JD1)) - LAN
	f = np.sin(u)*np.cos(Lam) - np.cos(u)*np.cos(inc)*np.sin(Lam) - np.cos(u)*np.sin(inc)*np.tan(phi)
	return f

def H_NR(t, orb, lam, phi, JD1):
	Lam_dot = W_EARTH - orb.get_LAN_dot()
	M_dot = orb.get_M_dot()
	omega_dot = orb.get_omega_dot()
	inc = orb.inc
	e = orb.e

	M = orb.M0 + orb.get_M_dot()*(t-orb.t0)
	nu = M2nu(M,orb.e)
	omega = orb.omega + orb.get_omega_dot()*(t-orb.t0)
	u = omega + nu
	LAN = orb.LAN + orb.get_LAN_dot()*(t-orb.t0)
	Lam = lam + rad(get_GMST(t/86400 + JD1)) - LAN

	E = 2*np.arctan(np.tan(nu/2)/(np.sqrt((1+e)/(1-e))))

	arg1 = M_dot/(1-e*np.cos(E))
	arg2 = (np.cos(nu/2)/np.cos(E/2))**2
	arg3 = np.sqrt((1+e)/(1-e))
	nu_dot = arg1 * arg2 * arg3
	u_dot = omega_dot + nu_dot

	f_dot = u_dot*np.cos(u)*np.cos(Lam) - Lam_dot*np.sin(u)*np.sin(Lam) - \
		np.cos(inc)*(-u_dot*np.sin(u)*np.sin(Lam) + Lam_dot*np.cos(u)*np.cos(Lam)) + \
		u_dot*np.sin(inc)*np.tan(phi)*np.sin(u)
	#
	return f_dot



def angle_intp(x0, x1, x2, t1, t2):
	x0 = x0 % 360
	x1 = x1 % 360
	x2 = x2 % 360
	x2[x2 < x1] = x2[x2 < x1] + 360
	x0[x0 < x1] = x0[x0 < x1] + 360
	t_intp = (t2-t1)/(x2-x1) * (x0 - x1) + t1
	return t_intp

@njit
def angle_intp_numba(x0, x1, x2, t1, t2):
	x0 = x0 % 360
	x1 = x1 % 360
	x2 = x2 % 360
	x2[x2 < x1] = x2[x2 < x1] + 360
	x0[x0 < x1] = x0[x0 < x1] + 360
	t_intp = (t2-t1)/(x2-x1) * (x0 - x1) + t1
	return t_intp

@njit
def get_GMST_numba(JD, angle=True):
	"""
	Vallado v4, algorithm 15
		JD technically based on UT1
		for the ex, same as our def.
	Due to precision,
		time accurate to 0.1 sec
		angle accurate to 1e-4 deg.
	returns either hrs or deg
		relative to vernal equinox
	"""
	T = (JD - 2451545.0)/36525.0
	h = 876600.0 * 3600
	GMST = 67310.54841 + (h + 8640184.812866)*T + 0.093104*T**2 - 6.2e-6*T**3
	GMST = GMST % 86400
	if angle:
		return GMST / 240 # deg
	else:
		return GMST / 3600 # hrs


@njit
def get_init_t_access_numba(t_r1, t_l1, t_r2, t_l2, lon_r1, lon_l1, lon_r2, lon_l2, invalid_left, invalid_right, \
						lat_in_bounds, pole_in_view, bridge_class1, bridge_class2, lon, lat, inc, dlon, Tn, \
						k_access1, k_access2, JD1, t_pole_N, t_pole_S, LAN, omega, M0, LAN_dot, omega_dot, M_dot):
	#

	# t_r1, t_l1, t_r2, t_l2 = ts0['r1'], ts0['l1'], ts0['r2'], ts0['l2']
	# lon_r1, lon_l1, lon_r2, lon_l2 = lons0['r1'], lons0['l1'], lons0['r2'], lons0['l2']
	# t_access = np.array([], dtype=types.float64)
	if invalid_left and invalid_right:
		if lat_in_bounds:
			# always accessed
			#	then what is the time of access..?
			if pole_in_view:
				# pole
				if lat > 0:
					# north
					# t_pole, _ = calc_t_u(np.pi/2,orb)
					t_pole = t_pole_N
				else:
					# south
					# t_pole, _ = calc_t_u(3*np.pi/2,orb)
					t_pole = t_pole_S
				t_access = t_pole + k_access1*Tn

			else:
				lam = np.radians(lon)
				# equator
				if inc <= np.pi/2:
					# prograde
					theta_g0 = np.radians(get_GMST_numba(JD1))
					angle = (lam + (theta_g0-LAN) - (omega + M0)) % (2*np.pi) - np.radians(dlon)*k_access1
					angle_rate = (omega_dot + M_dot) - (W_EARTH - LAN_dot)
					t_access = angle/angle_rate + k_access1*Tn

				else:
					# retrograde
					theta_g0 = np.radians(get_GMST_numba(JD1))
					angle = (lam + (theta_g0-LAN) + (omega + M0)) % (2*np.pi) - np.radians(dlon)*k_access1
					angle_rate = -(omega_dot + M_dot) - (W_EARTH - LAN_dot)
					t_access = angle/angle_rate + k_access1*Tn


		else:
			# never accessed
			t_access = np.empty((0,), dtype=np.float64)

		# continue

	elif invalid_left or invalid_right:
		# accessed on either asc or desc
		t1 = t_l1 + k_access1*Tn
		t2 = t_r1 + k_access1*Tn
		x1 = lon_l1 + k_access1*dlon
		x2 = lon_r1 + k_access1*dlon
		# if not left_to_right1:
		if bridge_class1 == 2:
			# right-to-left
			x1, x2 = x2, x1
			t1, t2 = t2, t1
		t_access = angle_intp_numba(np.full(x1.shape, lon), x1, x2, t1, t2)

	else:
		# accessed on both asc and desc

		t1 = t_l1 + k_access1*Tn
		t2 = t_r1 + k_access1*Tn
		x1 = lon_l1 + k_access1*dlon
		x2 = lon_r1 + k_access1*dlon
		# if not left_to_right1:
		if bridge_class1 == 2:
			# right-to-left
			x1, x2 = x2, x1
			t1, t2 = t2, t1
		t_access1 = angle_intp_numba(np.full(x1.shape, lon), x1, x2, t1, t2)

		t1 = t_l2 + k_access2*Tn
		t2 = t_r2 + k_access2*Tn
		x1 = lon_l2 + k_access2*dlon
		x2 = lon_r2 + k_access2*dlon
		# if not left_to_right2:
		if bridge_class2 == 2:
			# right-to-left
			x1, x2 = x2, x1
			t1, t2 = t2, t1
		t_access2 = angle_intp_numba(np.full(x1.shape, lon), x1, x2, t1, t2)

		# t_access = np.concatenate((t_access1, t_access2))
		t_access = np.zeros(len(t_access1) + len(t_access2), dtype=np.float64)
		t_access[0:len(t_access1)] = t_access1
		t_access[len(t_access1):] = t_access2

	return t_access




def get_init_t_access(t_r1, t_l1, t_r2, t_l2, lon_r1, lon_l1, lon_r2, lon_l2, invalid_left, invalid_right, \
						lat_in_bounds, pole_in_view, bridge_class1, bridge_class2, lon, lat, inc, dlon, Tn, \
						k_access1, k_access2, JD1, t_pole_N, t_pole_S, LAN, omega, M0, LAN_dot, omega_dot, M_dot):
	"""
	ts0, lons0
	invalid_left, invalid_right
	lat_in_bounds, pole_in_view
	lat
	orb
	lam
	dlon, Tn
	k_access1, k_access2
	JD1

	"""
	lam = rad(lon)

	# t_r1, t_l1, t_r2, t_l2 = ts0['r1'], ts0['l1'], ts0['r2'], ts0['l2']
	# lon_r1, lon_l1, lon_r2, lon_l2 = lons0['r1'], lons0['l1'], lons0['r2'], lons0['l2']
	t_access = np.array([])
	if invalid_left and invalid_right:
		if lat_in_bounds:
			# always accessed
			#	then what is the time of access..?
			if pole_in_view:
				# pole
				if lat > 0:
					# north
					# t_pole, _ = calc_t_u(np.pi/2,orb)
					t_pole = t_pole_N
				else:
					# south
					# t_pole, _ = calc_t_u(3*np.pi/2,orb)
					t_pole = t_pole_S
				t_access = t_pole + k_access1*Tn

			else:
				# equator
				if inc <= np.pi/2:
					# prograde
					theta_g0 = rad(get_GMST(JD1))
					# angle = (lam + (theta_g0-orb.LAN) - (orb.omega + orb.M0)) % (2*np.pi) - rad(dlon)*k_access1
					# angle_rate = (orb.get_omega_dot() + orb.get_M_dot()) - (W_EARTH - orb.get_LAN_dot())
					angle = (lam + (theta_g0-LAN) - (omega + M0)) % (2*np.pi) - np.radians(dlon)*k_access1
					angle_rate = (omega_dot + M_dot) - (W_EARTH - LAN_dot)
					t_access = angle/angle_rate + k_access1*Tn

				else:
					# retrograde
					theta_g0 = rad(get_GMST(JD1))
					# angle = (lam + (theta_g0-orb.LAN) + (orb.omega + orb.M0)) % (2*np.pi) - rad(dlon)*k_access1
					# angle_rate = -(orb.get_omega_dot() + orb.get_M_dot()) - (W_EARTH - orb.get_LAN_dot())
					angle = (lam + (theta_g0-LAN) + (omega + M0)) % (2*np.pi) - np.radians(dlon)*k_access1
					angle_rate = -(omega_dot + M_dot) - (W_EARTH - LAN_dot)
					t_access = angle/angle_rate + k_access1*Tn


		# else:
		# 	# never accessed
		# 	t_access = np.array([])

		# continue

	elif invalid_left or invalid_right:
		# accessed on either asc or desc
		t1 = t_l1 + k_access1*Tn
		t2 = t_r1 + k_access1*Tn
		x1 = lon_l1 + k_access1*dlon
		x2 = lon_r1 + k_access1*dlon
		# if not left_to_right1:
		if bridge_class1 == 2:
			# right-to-left
			x1, x2 = x2, x1
			t1, t2 = t2, t1
		t_access = angle_intp_numba(np.full(x1.shape, lon), x1, x2, t1, t2)

	else:
		# accessed on both asc and desc

		t1 = t_l1 + k_access1*Tn
		t2 = t_r1 + k_access1*Tn
		x1 = lon_l1 + k_access1*dlon
		x2 = lon_r1 + k_access1*dlon
		# if not left_to_right1:
		if bridge_class1 == 2:
			# right-to-left
			x1, x2 = x2, x1
			t1, t2 = t2, t1
		t_access1 = angle_intp_numba(np.full(x1.shape, lon), x1, x2, t1, t2)

		t1 = t_l2 + k_access2*Tn
		t2 = t_r2 + k_access2*Tn
		x1 = lon_l2 + k_access2*dlon
		x2 = lon_r2 + k_access2*dlon
		# if not left_to_right2:
		if bridge_class2 == 2:
			# right-to-left
			x1, x2 = x2, x1
			t1, t2 = t2, t1
		t_access2 = angle_intp_numba(np.full(x1.shape, lon), x1, x2, t1, t2)

		t_access = np.concatenate((t_access1, t_access2))

	return t_access


def get_swath_params(orb, lat, swath, JD1, verbose=0):
	"""

	"""
	phi = rad(lat)
	inc = orb.inc
	u_init = (orb.omega + orb.nu) % (2*np.pi)

	Tn = orb.get_period('nodal')
	Dn = orb.get_nodal_day()
	dlon = -2*np.pi*Tn/Dn * 180/np.pi
	split = get_split_flag(orb, swath, lat)
	dpsi = swath/(2*R_earth)
	lons0, us0, ts0 = get_init_swath_params(orb, phi, dpsi, JD1)
	if split:
		lons0, us0, ts0 = fix_split(inc, u_init, dlon, Tn, lons0, us0, ts0, verbose=0)

	lons0, us0, ts0, lat_in_bounds, invalid_left, invalid_right, pole_in_view = \
		fix_invalid(lons0, us0, ts0, lat, inc, swath)
	#

	if verbose:
		print('invalid_left', invalid_left)
		print('invalid_right', invalid_right)
		if lat > 0.0:
			print('lat > 0.0')
		else:
			print('lat < 0.0')
		if (0 <= u_init < np.pi/2) or (3*np.pi/2 < u_init < 2*np.pi):
			print('u_init 1st or 4th quadrant')
		else:
			print('u_init 2nd or 3rd quadrant')

		print('split', split)
		print('lat_in_bounds', lat_in_bounds)
		print('pole_in_view', pole_in_view)
		# print(k_access)

	return lons0, us0, ts0, split, invalid_left, invalid_right, lat_in_bounds, pole_in_view



def plot_debug_figure(lon, lat, orb, swath, JD1, lons0, ts0, us0, Tn, dlon, k_access,
						err, debug):
	
	import matplotlib.pyplot as plt
	from leocat.utils.plot import make_fig, plot_sim_subplot, pro_plot, split_lon, \
								set_axes_equal, set_aspect_equal, draw_vector
	pro_plot()
	from leocat.utils.general import pause

	"""
	lat
	lon
	orb
	dpsi
	JD1

	lons0, ts0, us0

	Tn
	dlon

	k_access

	err, debug

	"""
	dpsi = swath/(2*R_earth)
	phi = np.radians(lat)
	inc = orb.inc

	if err or debug:

		phi_GT_l = np.nan
		arg = (np.sin(phi) - np.sin(dpsi)*np.cos(inc))/np.cos(dpsi)
		if np.abs(arg) <= 1.0:
			phi_GT_l = np.arcsin(arg)

		phi_GT_r = np.nan
		arg = (np.sin(phi) - np.sin(-dpsi)*np.cos(inc))/np.cos(-dpsi)
		if np.abs(arg) <= 1.0:
			phi_GT_r = np.arcsin(arg)

		lat_GT_l, lat_GT_r = np.degrees((phi_GT_l,phi_GT_r))

		circle_lat_GT_l = circle_at_lat(phi_GT_l)
		circle_lat_GT_r = circle_at_lat(phi_GT_r)

		if k_access is None:
			# k_access = np.arange(1000)[::100]
			# k_access = np.arange(16)
			k_access = np.array([0])
		else:
			# k_access = np.arange(np.min(k_access),np.max(k_access)+1)
			k_access = np.hstack((0, k_access))
			# k_access = np.array([0])


		for k in k_access:

			title = 'inc = %.2f, lat = %.2f' % (deg(inc), lat) + '\n' + \
					'k = %d' % k
			#

			dt = k*Tn
			t = np.linspace(orb.t0, orb.t0 + Tn, 10000) + dt
			# r_eci, v_eci = orb.propagate(t)
			# lon_GT, lat_GT = get_sc_lonlat(orb, t)
			r_eci, v_eci = orb.propagate(t)
			r_eci_gt = unit(r_eci)*R_earth
			JD = JD1 + t/86400

			r_ecf = convert_ECI_ECF(JD,r_eci)
			lon_GT, lat_GT = cart_to_RADEC(r_ecf) # spherical
			lon_GT = wrap(lon_GT)

			r_l, r_r = create_swath(unit(r_eci),unit(v_eci),swath=swath)
			r_l_ecf = convert_ECI_ECF(JD,r_l)
			r_r_ecf = convert_ECI_ECF(JD,r_r)
			lon_l, lat_l = cart_to_RADEC(r_l_ecf)
			lon_l = wrap(lon_l)
			lon_r, lat_r = cart_to_RADEC(r_r_ecf)
			lon_r = wrap(lon_r)
			index_GT = split_lon(lon_GT)
			index_l = split_lon(lon_l)
			index_r = split_lon(lon_r)

			lon_l1_0 = wrap(lons0['l1'] + k*dlon)
			lon_r1_0 = wrap(lons0['r1'] + k*dlon)
			lon_l2_0 = wrap(lons0['l2'] + k*dlon)
			lon_r2_0 = wrap(lons0['r2'] + k*dlon)

			CT_geo_beg = create_CT_geo(orb, 0+dt, swath=swath)
			CT_geo_end = create_CT_geo(orb, Tn+dt, swath=swath)
			err_l1 = CT_geo_beg[0] - r_l[0]
			err_r1 = CT_geo_beg[-1] - r_r[0]
			err_l2 = CT_geo_end[0] - r_l[-1]
			err_r2 = CT_geo_end[-1] - r_r[-1]

			if 0:
				err_list = [err_l1, err_r1, err_l2, err_l2]
				for err0 in err_list:
					print(np.linalg.norm(err0))

			lon_CT_beg, lat_CT_beg = CT_geo_to_lonlat(CT_geo_beg, 0+dt, JD1)
			lon_CT_end, lat_CT_end = CT_geo_to_lonlat(CT_geo_end, Tn+dt, JD1)			


			# err_l1 = np.array([lon_CT_beg[0]-lon_l[0], lat_CT_beg[0]-lat_l[0]])
			# err_r1 = np.array([lon_CT_beg[-1]-lon_r[0], lat_CT_beg[-1]-lat_r[0]])
			# err_l2 = np.array([lon_CT_end[0]-lon_l[-1], lat_CT_end[0]-lat_l[-1]])
			# err_r2 = np.array([lon_CT_end[-1]-lon_r[-1], lat_CT_end[-1]-lat_r[-1]])
			# if 1:
			# 	err_list = [err_l1, err_r1, err_l2, err_l2]
			# 	for err0 in err_list:
			# 		print(np.linalg.norm(err0))

			# sys.exit()

			CT_geo_l1 = None
			CT_geo_r1 = None
			CT_geo_l2 = None
			CT_geo_r2 = None
			if ~np.isnan(ts0['l1']):
				CT_geo_l1 = create_CT_geo(orb, ts0['l1']+dt)
			if ~np.isnan(ts0['r1']):
				CT_geo_r1 = create_CT_geo(orb, ts0['r1']+dt)
			if ~np.isnan(ts0['l2']):
				CT_geo_l2 = create_CT_geo(orb, ts0['l2']+dt)
			if ~np.isnan(ts0['r2']):
				CT_geo_r2 = create_CT_geo(orb, ts0['r2']+dt)

			# fig, ax = make_fig()
			# # ax.plot(t, unwrap(lon_GT))
			# ax.plot(t, unwrap(lon_l))
			# ax.plot(t, unwrap(lon_r))
			# ax.grid()
			# fig.show()

			# fig, ax = make_fig()
			# ax.plot(t, lat_GT)
			# ax.plot(t, lat_l)
			# ax.plot(t, lat_r)
			# fig.show()

			if 1:
				# fig, ax = make_fig()
				fig = plt.figure(figsize=(10,4))
				# fig = plt.figure(figsize=(14,6))
				ax = fig.add_subplot(1,2,1)

				for idx in index_GT:
					ax.plot(lon_GT[idx], lat_GT[idx], c='C0')
				for idx in index_l:
					ax.plot(lon_l[idx], lat_l[idx], '--', c='C0')
				for idx in index_r:
					ax.plot(lon_r[idx], lat_r[idx], '--', c='C0')


				if 0:
					lon_bridge = []
					lat_bridge = []
					idx_bridge = []
					bridge_count = 10
					t_bridge = np.linspace(0,Tn,bridge_count+2)[1:-1] + dt
					for j in range(bridge_count):
						# t_bridge0 = 0+dt+j*Tn/(bridge_count)
						t_bridge0 = t_bridge[j]
						CT_geo_bridge0 = create_CT_geo(orb, t_bridge0, swath=swath, N=10)
						lon_bridge0, lat_bridge0 = CT_geo_to_lonlat(CT_geo_bridge0, t_bridge0, JD1)
						idx_bridge.append(split_lon(lon_bridge0))
						lon_bridge.append(lon_bridge0)
						lat_bridge.append(lat_bridge0)

					for j in range(bridge_count):
						lon_bridge0, lat_bridge0 = lon_bridge[j], lat_bridge[j]
						# idx_bridge0 = idx_bridge[j]
						# for idx_bridge0 in idx_bridge[j]:
							# i1, i2 = idx_bridge0[0], idx_bridge0[-1]+1
							# ax.plot(lon_bridge0[i1:i2], lat_bridge0[i1:i2], 'k')
						ax.plot(lon_bridge0, lat_bridge0, 'k')

				if 0:
					t_CT, lon_CT, lat_CT, CT_geo = continuous_space(orb, t, JD1, return_CT_geo=True, debug=1)
					CT_geo_list = CT_geo
					t_geo_list = t_CT
					ax.plot(lon_CT, lat_CT, 'x')
					plot_CT_geodesic(ax, CT_geo_list, t_geo_list, JD1)

				if 0:
					# plot CT geodesic at instant it exists
					CT_geo_list = [CT_geo_l1, CT_geo_r1, CT_geo_l2, CT_geo_r2]
					t_geo_list = [ts0['l1']+dt, ts0['r1']+dt, ts0['l2']+dt, ts0['r2']+dt]
					plot_CT_geodesic(ax, CT_geo_list, t_geo_list, JD1)
					# for j in range(len(t_geo_list)):
					# 	CT_geo0 = CT_geo_list[j]
					# 	if CT_geo0 is None:
					# 		continue
					# 	t_geo0 = t_geo_list[j]
					# 	lon_CT_geo, lat_CT_geo = CT_geo_to_lonlat(CT_geo0, t_geo0)
					# 	index_CT = split_lon(lon_CT_geo)
					# 	for idx in index_CT:
					# 		ax.plot(lon_CT_geo[idx], lat_CT_geo[idx], c='C3')



				if 1:
					CT_geo_list = [CT_geo_beg, CT_geo_end]
					t_geo_list = [0+dt, Tn+dt]
					plot_CT_geodesic(ax, CT_geo_list, t_geo_list, JD1, colors=['g','r'])


				ax.plot([-180,180],[lat,lat],'k',alpha=0.25)
				ax.plot(lon, lat, '.')

				if ~np.isnan(lon_l1_0):
					ax.plot(lon_l1_0, lat, 'g.')
					ax.text(lon_l1_0, lat, '  lon_l1', fontsize=8)
				if ~np.isnan(lon_r1_0):
					ax.plot(lon_r1_0, lat, 'r.')
					ax.text(lon_r1_0, lat, '  lon_r1', fontsize=8)
				if ~np.isnan(lon_l2_0):
					ax.plot(lon_l2_0, lat, 'g.')
					ax.text(lon_l2_0, lat, '  lon_l2', fontsize=8)
				if ~np.isnan(lon_r2_0):
					ax.plot(lon_r2_0, lat, 'r.')
					ax.text(lon_r2_0, lat, '  lon_r2', fontsize=8)

				ax.set_xlim(-180,180)
				ax.set_ylim(-90,90)

				ax.grid()
				ax.set_title(title)
				ax.set_xlabel('Longitude (deg)')
				ax.set_ylabel('Latitude (deg)')


				#

				AT_l1, AT_GT1, AT_r1, v_l1, v_GT1, v_r1 = get_AT(orb, ts0['l1']+dt, ts0['GT1']+dt, ts0['r1']+dt)
				AT_l2, AT_GT2, AT_r2, v_l2, v_GT2, v_r2 = get_AT(orb, ts0['l2']+dt, ts0['GT2']+dt, ts0['r2']+dt)

				CT_l1, CT_GT1, CT_r1 = get_CT(ts0['l1']+dt, ts0['GT1']+dt, ts0['r1']+dt, \
											lons0['l1']+k*dlon, lons0['GT1']+k*dlon, lons0['r1']+k*dlon, JD1, lat)
				#
				CT_l2, CT_GT2, CT_r2 = get_CT(ts0['l2']+dt, ts0['GT2']+dt, ts0['r2']+dt, \
											lons0['l2']+k*dlon, lons0['GT2']+k*dlon, lons0['r2']+k*dlon, JD1, lat)
				#


				circle_lat = circle_at_lat(phi)
				circle_lat0 = circle_at_lat(0.0)

				# RA, DEC = 30, 45

				with warnings.catch_warnings():
					warnings.simplefilter('ignore', RuntimeWarning)
					AT_mid = np.nanmean([AT_GT1,AT_GT2,AT_l1,AT_l2,AT_r1,AT_r2],axis=0)
				# AT_mid = (AT_GT1 + AT_GT2)/2
				if (~np.isnan(AT_mid)).all():
					RA, DEC = cart_to_RADEC(AT_mid)
				else:
					RA, DEC = 25, 30
				# RA -= 10
				# DEC -= 2
				if DEC > 60:
					DEC = 60
				if DEC < -60:
					DEC = -60

				zoom = 1.25
				target = np.zeros(3)

				ax = fig.add_subplot(1,2,2, projection='3d', proj_type='ortho')


				fig_3d, ax_3d = make_fig('3d')
				lines = []
				im = ax_3d.plot(r_eci_gt.T[0], r_eci_gt.T[1], r_eci_gt.T[2])
				lines.append(im)
				lines.append(ax_3d.plot(r_l.T[0], r_l.T[1], r_l.T[2], '--', c=im[0].get_c()))
				lines.append(ax_3d.plot(r_r.T[0], r_r.T[1], r_r.T[2], '--', c=im[0].get_c()))
				lines.append(ax_3d.plot(circle_lat.T[0], circle_lat.T[1], circle_lat.T[2], 'k', alpha=0.25))
				lines.append(ax_3d.plot(circle_lat0.T[0], circle_lat0.T[1], circle_lat0.T[2], 'k', alpha=0.25))
				lines.append(ax_3d.plot(circle_lat_GT_l.T[0], circle_lat_GT_l.T[1], circle_lat_GT_l.T[2], 'k', alpha=0.05))
				lines.append(ax_3d.plot(circle_lat_GT_r.T[0], circle_lat_GT_r.T[1], circle_lat_GT_r.T[2], 'k', alpha=0.05))

				if CT_geo_l1 is not None:
					lines.append(ax_3d.plot(CT_geo_l1.T[0], CT_geo_l1.T[1], CT_geo_l1.T[2]))
				if CT_geo_r1 is not None:
					lines.append(ax_3d.plot(CT_geo_r1.T[0], CT_geo_r1.T[1], CT_geo_r1.T[2]))

				if CT_geo_l2 is not None:
					lines.append(ax_3d.plot(CT_geo_l2.T[0], CT_geo_l2.T[1], CT_geo_l2.T[2]))
				if CT_geo_r2 is not None:
					lines.append(ax_3d.plot(CT_geo_r2.T[0], CT_geo_r2.T[1], CT_geo_r2.T[2]))

				plot_sim_subplot(ax, lines, RA=RA, DEC=DEC, zoom=zoom, target=target)

				labels = ['AT_l1', 'AT_GT1', 'AT_r1', 'CT_l1', 'CT_GT1', 'CT_r1']
				vals = [AT_l1, AT_GT1, AT_r1, CT_l1, CT_GT1, CT_r1]
				plot_intersections(ax, vals, labels, fontsize=7)
				labels = ['AT_l2', 'AT_GT2', 'AT_r2', 'CT_l2', 'CT_GT2', 'CT_r2']
				vals = [AT_l2, AT_GT2, AT_r2, CT_l2, CT_GT2, CT_r2]
				plot_intersections(ax, vals, labels, fontsize=7)

				draw_vector(ax, AT_GT1, AT_GT1 + unit(v_GT1)*R_earth/4, 'r')
				draw_vector(ax, AT_GT2, AT_GT2 + unit(v_GT2)*R_earth/4, 'r')
				if 0:
					# Plot ECEF frame in ECI
					# 	doesn't really make sense b/c ECEF moves during orbit
					theta = rad(get_GMST((t[0]+dt+Tn/2)/86400 + JD1))
					R_axes = R3(theta) * R_earth
					# R_axes = np.eye(3)*R_earth
					draw_vector(ax,[0,0,0],R_axes[0],'r',zorder=0)
					draw_vector(ax,[0,0,0],R_axes[1],[0,0.9,0],zorder=0)
					draw_vector(ax,[0,0,0],R_axes[2],'b',zorder=0)

				ax.set_title(title)
				fig.tight_layout()
				fig.show()

			if len(k_access) > 1:
				pause()
				plt.close('all')

		pause()
		plt.close('all')





def classify_bridges(lons0, us0, ts0, orb, lat, split, invalid_left, invalid_right, lat_in_bounds):
	#

	inc = orb.inc
	u_init = (orb.omega + orb.nu) % (2*np.pi)
	Tn = orb.get_period('nodal')
	Dn = orb.get_nodal_day()
	dlon = -2*np.pi*Tn/Dn * 180/np.pi

	"""
	lons0, us0, ts0
	inc
	dlon
	Tn
	u_init
	lat
	split
	invalid_left, invalid_right, lat_in_bounds

	"""
	# k_access1 = None
	# k_access2 = None
	# t_access = None

	"""
	classifications
	0 - not classified
	1 - left-to-right
	2 - right-to-left
	3 - all covered
	4 - none covered
	"""

	bridge_class1 = 0
	bridge_class2 = 0

	# left_to_right1 = False
	# left_to_right2 = False
	# dlon_wrap = np.nan
	# err = True
	if invalid_left and invalid_right:
		# access on all or no revs
		# err = False
		if lat_in_bounds:
			# ratio = 180/np.pi * -2*np.pi/dlon # = Dn/Tn
			# num_revs = int((k0_range[-1]+1)*ratio)
			# k_access1 = np.arange(num_revs)
			bridge_class1 = 3
		else:
			# k_access1 = np.array([])
			bridge_class1 = 4

	elif invalid_left or invalid_right:
		if lat > 0:
			if (0 <= u_init < np.pi/2) or (3*np.pi/2 < u_init < 2*np.pi):
				# 1st/4th
				# k_access1, dlon_wrap, err = vs_left_to_right(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
				# left_to_right1 = True
				bridge_class1 = 1
			else:
				# 2nd/3rd
				if split:
					if inc <= np.pi/2:
						# prograde
						# print('special 1: lat > 0, prograde')
						lons0['r1'] = wrap(lons0['r1'] - dlon)
						ts0['r1'] -= Tn
					else:
						# retrograde
						# print('special 2: lat > 0, retrograde')
						lons0['l1'] = wrap(lons0['l1'] - dlon)
						ts0['l1'] -= Tn

					# k_access1, dlon_wrap, err = vs_right_to_left(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
					bridge_class1 = 2
				else:
					# k_access1, dlon_wrap, err = vs_left_to_right(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
					# left_to_right1 = True
					bridge_class1 = 1

		elif lat < 0:
			if (0 <= u_init < np.pi/2) or (3*np.pi/2 < u_init < 2*np.pi):
				# 1st/4th
				if split:
					if inc <= np.pi/2:
						# prograde
						# print('special 3: lat < 0, prograde')
						lons0['l1'] = wrap(lons0['l1'] - dlon)
						ts0['l1'] -= Tn
					else:
						# retrograde
						# print('special 4: lat < 0, retrograde')
						lons0['r1'] = wrap(lons0['r1'] - dlon)
						ts0['r1'] -= Tn

					# k_access1, dlon_wrap, err = vs_left_to_right(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
					# left_to_right1 = True
					bridge_class1 = 1
				else:
					# k_access1, dlon_wrap, err = vs_right_to_left(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
					bridge_class1 = 2
			else:
				# 2nd/3rd
				# k_access1, dlon_wrap, err = vs_right_to_left(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
				bridge_class1 = 2

		# else:
		# 	# shouldn't be able to have invalid_left OR invalid_right and lat=0
		# 	# if lat=0, then both are invalid or swath=0
		# 	import warnings
		# 	warnings.warn('invalid_left OR invalid_right and lat=0')

	else:
		"""
		Would u_init not imply the left-to-right vs. right-to-left option?
			instead of us0['u_r1']/2 check?
		Maybe. u_r/l/1/2 cannot be at +/-pi/2 I don't think, unless swath
		is zero... idk. Hm. Or it's extremely unlikely, since the swath edge
		would have to intersect the line of latitude exactly. It makes sense
		to consider u_init as a discriminator in splits because the split
		depends on the location of the start/end, whereas k_access depends only
		on u_r/l/1/2. It might be more robust to use them instead of u_init..

		Is it possible to be ascending in 1 and 2? Or descending in 1 and 2?
			One implies the other?
		"""

		# valid both sides
		if np.pi/2 < us0['r1'] < 3*np.pi/2:
			# descending
			# if not (np.pi/2 < us0['l1'] < 3*np.pi/2):
			# 	print('warning: u_l1 not in same quadrant as u_r1 (1)')
			# k_access1, dlon_wrap, err1 = vs_right_to_left(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
			bridge_class1 = 2

		else:
			# ascending
			# if (np.pi/2 <= us0['l1'] <= 3*np.pi/2):
			# 	print('warning: u_l1 not in same quadrant as u_r1 (2)')
			# k_access1, dlon_wrap, err1 = vs_left_to_right(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
			# left_to_right1 = True
			bridge_class1 = 1

		if np.pi/2 < us0['r2'] < 3*np.pi/2:
			# descending
			# if not (np.pi/2 < us0['l2'] < 3*np.pi/2):
			# 	print('warning: u_l2 not in same quadrant as u_r2 (1)')
			# k_access2, dlon_wrap, err2 = vs_right_to_left(lon, lons0['l2'], lons0['r2'], k0_range, dlon, debug=0)
			bridge_class2 = 2

		else:
			# ascending
			# if (np.pi/2 <= us0['l2'] <= 3*np.pi/2):
			# 	print('warning: u_l2 not in same quadrant as u_r2 (2)')
			# k_access2, dlon_wrap, err2 = vs_left_to_right(lon, lons0['l2'], lons0['r2'], k0_range, dlon, debug=0)
			# left_to_right2 = True
			bridge_class2 = 1

		# err = err1 and err2
		# k_access = np.sort(np.concatenate((k_access1,k_access2)))
		# err = err2
		# k_access = k_access2

	# return k_access1, k_access2, left_to_right1, left_to_right2, dlon_wrap, lons0, us0, ts0, err
	return bridge_class1, bridge_class2, lons0, us0, ts0



def get_k_access(lon, lons0, k0_range, dlon, bridge_class1, bridge_class2):

	"""
	classifications
		0 - not classified
		1 - left-to-right
		2 - right-to-left
		3 - all covered
		4 - none covered
	"""
	k_access1 = np.array([])
	k_access2 = np.array([])
	if bridge_class1 == 1:
		k_access1 = vs_left_to_right(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
	elif bridge_class1 == 2:
		k_access1 = vs_right_to_left(lon, lons0['l1'], lons0['r1'], k0_range, dlon, debug=0)
	elif bridge_class1 == 3:
		ratio = 180/np.pi * -2*np.pi/dlon # = Dn/Tn
		num_revs = int((k0_range[-1]+1)*ratio)
		k_access1 = np.arange(num_revs)
	# elif bridge_class1 == 4:
	# 	k_access1 = np.array([])

	if bridge_class2 == 1:
		k_access2 = vs_left_to_right(lon, lons0['l2'], lons0['r2'], k0_range, dlon, debug=0)
	elif bridge_class2 == 2:
		k_access2 = vs_right_to_left(lon, lons0['l2'], lons0['r2'], k0_range, dlon, debug=0)

	return k_access1, k_access2


def fix_invalid(lons0, us0, ts0, lat, inc, swath):
	lat_in_bounds = True
	invalid_left, invalid_right = False, False
	pole_in_view = False
	lat_GT_max = np.degrees(inc)
	if lat_GT_max > 90:
		lat_GT_max = 180-lat_GT_max
	#
	lat_GT_min = -lat_GT_max
	dist_pole = (90.0-lat_GT_max)*np.pi/180 * R_earth
	if dist_pole <= swath/2:
		pole_in_view = True

	# if np.isnan([us0['l1'], us0['l2'], us0['GT1'], us0['GT2'], us0['r1'], us0['r2']]).any():
	if np.isnan([us0['l1'], us0['l2'], us0['r1'], us0['r2']]).any():

		# GP at poles or equator between envelope edges
		# sin_u = (np.sin(phi) - np.cos(inc)*np.sin(dpsi))/(np.sin(inc)*np.cos(dpsi))
		# if |sin_u| > 1:
		if np.isnan(us0['l1']) or np.isnan(us0['l2']):
			# l is nan
			invalid_left = True
		if np.isnan(us0['r1']) or np.isnan(us0['r2']):
			# r is nan
			invalid_right = True

		if invalid_left and invalid_right:
			# both nan
			# test if GP is outside latitude coverage
			# lat_GT_max = np.degrees(inc)
			# if lat_GT_max > 90:
			# 	lat_GT_max = 180-lat_GT_max
			# #
			dist_lat = (np.abs(lat)-lat_GT_max)*np.pi/180 * R_earth
			if dist_lat > swath/2:
				lat_in_bounds = False
				# access on no revs

			# if lat_in_bounds:
			# 	# either at pole or equator
			# 	# access on all revs
			# 	if pole_in_view:
			# 		# pole
			# 		pass
			# 	else:
			# 		# equator
			# 		pass
			# 	pass

		elif invalid_left:
			# if lat > 0.0:
			us0['l1'], us0['r2'], ts0['l1'], ts0['r2'], lons0['l1'], lons0['r2'] = \
				span_from_right(us0['l1'], us0['r2'], ts0['l1'], ts0['r2'], lons0['l1'], lons0['r2'])
			#

		elif invalid_right:
			us0['r1'], us0['l2'], ts0['r1'], ts0['l2'], lons0['r1'], lons0['l2'] = \
				span_from_left(us0['r1'], us0['l2'], ts0['r1'], ts0['l2'], lons0['r1'], lons0['l2'])
			#

		else:
			# both are valid
			pass

	return lons0, us0, ts0, lat_in_bounds, invalid_left, invalid_right, pole_in_view


def get_init_swath_params(orb, phi, dpsi, JD1):
	inc = orb.inc
	# calc_u_swath should be exact, no matter case
	#	true to spherical latitude
	u_l1, u_l2 = calc_u_swath(phi, inc, dpsi)
	# u_GT1, u_GT2 = calc_u_swath(phi, inc, 0.0)
	u_r1, u_r2 = calc_u_swath(phi, inc, -dpsi)

	# calc_t_u exact to J2, even if omega_dot != 0
	t_l1, _ = calc_t_u(u_l1,orb)
	t_l2, _ = calc_t_u(u_l2,orb)
	# t_GT1, _ = calc_t_u(u_GT1,orb)
	# t_GT2, _ = calc_t_u(u_GT2,orb)
	t_r1, _ = calc_t_u(u_r1,orb)
	t_r2, _ = calc_t_u(u_r2,orb)
	t_l1, t_l2, t_r1, t_r2

	# if t_GT2 < t_GT1:
	# 	t_GT1, t_GT2 = t_GT2, t_GT1
	# 	u_GT1, u_GT2 = u_GT2, u_GT1
	if t_l2 < t_l1:
		t_l1, t_l2 = t_l2, t_l1
		u_l1, u_l2 = u_l2, u_l1
	if t_r2 < t_r1:
		t_r1, t_r2 = t_r2, t_r1
		u_r1, u_r2 = u_r2, u_r1

	# calc_lam exact no matter case
	#	true to geographic or geodetic?
	lam_l1, lam_l2 = calc_lam(t_l1,u_l1,orb,dpsi,JD1), calc_lam(t_l2,u_l2,orb,dpsi,JD1)
	# lam_GT1, lam_GT2 = calc_lam(t_GT1,u_GT1,orb,0.0,JD1), calc_lam(t_GT2,u_GT2,orb,0.0,JD1)
	lam_r1, lam_r2 = calc_lam(t_r1,u_r1,orb,-dpsi,JD1), calc_lam(t_r2,u_r2,orb,-dpsi,JD1)

	lon_l1, lon_l2 = np.degrees((lam_l1, lam_l2))
	# lon_GT1, lon_GT2 = np.degrees((lam_GT1, lam_GT2))
	lon_r1, lon_r2 = np.degrees((lam_r1, lam_r2))
	lon_l1, lon_l2 = wrap(np.array([lon_l1,lon_l2]))
	# lon_GT1, lon_GT2 = wrap(np.array([lon_GT1,lon_GT2]))
	lon_r1, lon_r2 = wrap(np.array([lon_r1,lon_r2]))

	lons0 = {'r1': lon_r1, 'l1': lon_l1,
		'r2': lon_r2, 'l2': lon_l2}
	#
	us0 = {'r1': u_r1, 'l1': u_l1,
			'r2': u_r2, 'l2': u_l2}
	#
	ts0 = {'r1': t_r1, 'l1': t_l1,
			'r2': t_r2, 'l2': t_l2}
	#

	# lons0 = {'r1': lon_r1, 'l1': lon_l1, 'GT1': lon_GT1,
	# 	'r2': lon_r2, 'l2': lon_l2, 'GT2': lon_GT2}
	# #
	# us0 = {'r1': u_r1, 'l1': u_l1, 'GT1': u_GT1,
	# 		'r2': u_r2, 'l2': u_l2, 'GT2': u_GT2}
	# #
	# ts0 = {'r1': t_r1, 'l1': t_l1, 'GT1': t_GT1,
	# 		'r2': t_r2, 'l2': t_l2, 'GT2': t_GT2}
	# #
	return lons0, us0, ts0


def get_split_flag(orb, swath, lat):
	CT_geo_beg0 = get_CT_edges(orb, 0, swath)
	# CT_geo_end0 = get_CT_edges(orb, Tn+dt, swath)

	_, lat_beg_l = cart_to_RADEC(CT_geo_beg0[0])
	_, lat_beg_r = cart_to_RADEC(CT_geo_beg0[1])
	# _, lat_end_l = cart_to_RADEC(CT_geo_end0[0])
	# _, lat_end_r = cart_to_RADEC(CT_geo_end0[1])

	split = 0
	lat_lower, lat_upper = lat_beg_l, lat_beg_r
	if lat_upper < lat_lower:
		lat_lower, lat_upper = lat_upper, lat_lower
	if (lat_lower < lat < lat_upper):
		split = 1
	return split


def calc_u_swath(phi, inc, dpsi):
	if np.sin(inc) == 0.0:
		return np.nan, np.nan
	sin_u = (np.sin(phi) - np.cos(inc)*np.sin(dpsi))/(np.sin(inc)*np.cos(dpsi))
	u1, u2 = np.nan, np.nan
	if np.abs(sin_u) <= 1:
		sol = np.arcsin(sin_u)
		u1 = (sol) % (2*np.pi)
		u2 = (np.pi - sol) % (2*np.pi)
		if u1 > u2:
			u1, u2 = u2, u1
	return u1, u2

def calc_lam(t, u, orb, dpsi, JD1):
	if np.isnan(t) or np.isnan(u):
		return np.nan
	LAN_dot = orb.get_LAN_dot()
	LAN = orb.LAN + LAN_dot*(t-orb.t0)
	c_LAN = np.cos(LAN)
	s_LAN = np.sin(LAN)
	c_u = np.cos(u)
	s_u = np.sin(u)
	c_i = np.cos(orb.inc)
	s_i = np.sin(orb.inc)
	c_dpsi = np.cos(dpsi)
	s_dpsi = np.sin(dpsi)

	RHS_c = (c_LAN*c_u - s_LAN*c_i*s_u)*c_dpsi + s_LAN*s_i*s_dpsi
	RHS_s = (s_LAN*c_u + c_LAN*c_i*s_u)*c_dpsi - c_LAN*s_i*s_dpsi

	theta_g = get_GMST(JD1 + t/86400) * np.pi/180
	lam = np.arctan2(RHS_s,RHS_c) - theta_g

	return lam


def create_swath(r_eci_gt, v_eci_gt, swath):
	from leocat.utils.math import R1
	x_hat_gt = unit(v_eci_gt)
	z_hat_gt = unit(r_eci_gt)
	y_hat_gt = unit(np.cross(z_hat_gt,x_hat_gt))
	R_GT = np.transpose([x_hat_gt,y_hat_gt,z_hat_gt], (1,2,0))

	central_angle = (swath/2)/R_earth # rad
	r_l_loc = R1(-central_angle) @ np.array([0,0,R_earth]) # in local frame
	r_r_loc = R1(central_angle) @ np.array([0,0,R_earth])
	r_l = R_GT @ r_l_loc # in eci
	r_r = R_GT @ r_r_loc

	return r_l, r_r

def circle_at_lat(phi,N=1000):
	radius_lat = R_earth * np.cos(phi)
	angle = np.linspace(0,2*np.pi,N)
	z_lat = R_earth * np.sin(phi)
	circle_lat = radius_lat * np.transpose([np.cos(angle), np.sin(angle), np.full(angle.shape, z_lat/radius_lat)])
	return circle_lat


def create_CT_geo(orb, t, N=250, swath=None):
	r0, v0 = orb.propagate(t)
	h_hat0 = unit(np.cross(r0,v0))
	r_hat0 = unit(r0)
	if swath is None:
		psi_range = np.flip(np.linspace(0,2*np.pi,N))
	else:
		psi_range = np.flip(np.linspace(-swath/(2*R_earth), swath/(2*R_earth), N))

	CT_geo_x = r_hat0[0]*np.cos(psi_range) + h_hat0[0]*np.sin(psi_range)
	CT_geo_y = r_hat0[1]*np.cos(psi_range) + h_hat0[1]*np.sin(psi_range)
	CT_geo_z = r_hat0[2]*np.cos(psi_range) + h_hat0[2]*np.sin(psi_range)
	CT_geo = np.transpose([CT_geo_x,CT_geo_y,CT_geo_z]) * R_earth
	return CT_geo

def get_CT_edges(orb, t, swath):
	CT_geo = create_CT_geo(orb, t, swath=swath, N=2)
	return CT_geo


def calc_Lam(u):
	arg_c = np.cos(u_l1)/np.cos(phi)
	arg_s = np.sin(u_l1)*np.cos(inc)/np.cos(phi)
	Lam_t = np.arctan2(arg_s, arg_c)
	return Lam_t



@njit
def vector_search_numba(lon, lon_l1, lon_r1, k0_range, dlon):

	dlon_abs = np.abs(dlon)

	problem_type = 0
	# dlon_wrap = (lon_r1-lon_l1) % 360.0
	# dlon_wrap2 = np.fmod(lon_r1-lon_l1,360.0)
	dlon_wrap = (lon_r1-lon_l1) % 360.0
	# dlon_wrap = wrap(lon_r1-lon_l1, two_pi=True)

	# err = np.abs(dlon_wrap2 - dlon_wrap)
	# # if err > 1e-12:
	# if err > 0:
	# 	from leocat.utils.general import pause
	# 	print(lon_r1-lon_l1, err, dlon_wrap, dlon_wrap2)
	# 	# print(dlon_wrap)
	# 	pause()

	if dlon_wrap < dlon_abs:
		problem_type = 1
	elif dlon_wrap < dlon_abs*2:
		problem_type = 2
	else:
		problem_type = 3

	lon_l1_unwrap = lon_l1
	lon_r1_unwrap = lon_l1 + dlon_wrap

	# # k_l = hash_index(lon - 360*k0_range, lon_l1_unwrap, dlon_abs) # hash origin on LHS
	# # k_r = hash_index(lon - 360*k0_range, lon_r1_unwrap-dlon_abs, dlon_abs) # hash origin on RHS

	# np.floor((val-val0)/res).astype(int)
	val = lon - 360*k0_range
	val0 = lon_l1_unwrap
	res = dlon_abs
	k_l = np.floor((val-val0)/res).astype(types.int64)
	# k_l = np.floor((val-val0)/res).astype(int)

	# k_access = np.array([0])
	# return k_access

	val = lon - 360*k0_range
	val0 = lon_r1_unwrap-dlon_abs
	# res = dlon_abs
	k_r = np.floor((val-val0)/res).astype(types.int64)
	# k_r = np.floor((val-val0)/res).astype(int)

	# k_access = np.array([0])
	# return k_access

	if problem_type == 1:
		# no overlap between successive tracks
		#	lon cov region within L AND R
		k_access = k_l[k_l==k_r]

	elif problem_type == 2:
		# overlap between successive tracks (two tracks)
		#	lon cov region within L OR R
		k_access = np.unique(np.hstack((k_l,k_r)))
		# vec = np.hstack((k_l,k_r))
		# idx = unique_index(np.hstack((k_l,k_r)))
		# k_access = vec[idx]

	elif problem_type == 3:
		# overlap between multiple successive tracks
		#	lon cov region within L, R, and revs in-between
		# skips possible, make bridges

		# k1 = np.empty(len(k_l), dtype=types.int64)
		# k2 = np.empty(len(k_r), dtype=types.int64)
		s = 0
		for j in range(len(k_l)):
			num_elem = k_l[j] - k_r[j] + 1
			s = s + num_elem

		# print('problem_type = 3')
		k_access = np.zeros(s, dtype=types.int64)
		# k_access = np.zeros(s, dtype=int)
		offset = 0
		for j in range(len(k_l)):
			vec = np.flip(np.arange(k_r[j],k_l[j]+1))
			k_access[offset:offset + len(vec)] = vec
			offset = offset + len(vec)

		# k_access = np.zeros(s, dtype=types.int64)
		# for j in range(len(k_l)):
		# k_access = []
		# for j in range(len(k_l)):
		# 	k_access.append(np.flip(np.arange(k_r[j],k_l[j]+1)))
		# k_access = np.concatenate(k_access)

	return -k_access



def vector_search(lon, lon_l1, lon_r1, k0_range, dlon, debug=0):

	if np.isnan(lon_l1) or np.isnan(lon_r1):
		return np.array([])

	"""
	Depends on l1 being less than r1 (in unwrapped angle)

	"""

	# lon_l1 = float(np.copy(lon_l1))
	# lon_r1 = float(np.copy(lon_r1))
	dlon_abs = np.abs(dlon)
	# if lon_l1 > lon_r1:
	# 	lon_l1, lon_r1 = lon_r1, lon_l1

	problem_type = 0
	dlon_wrap = wrap(lon_r1-lon_l1, two_pi=True)
	# print('dlon_wrap', dlon_wrap)
	if dlon_wrap < dlon_abs:
		problem_type = 1
	elif dlon_wrap < dlon_abs*2:
		problem_type = 2
	else:
		problem_type = 3

	lon_l1_unwrap = lon_l1
	lon_r1_unwrap = lon_l1 + dlon_wrap


	# k0_range = np.arange(0,10)
	# k_l = hash_index(lon - 360*k0_range, lon_l1, dlon_abs) # hash origin on LHS
	# k_r = hash_index(lon - 360*k0_range, lon_r1-dlon_abs, dlon_abs) # hash origin on RHS
	k_l = hash_index(lon - 360*k0_range, lon_l1_unwrap, dlon_abs) # hash origin on LHS
	k_r = hash_index(lon - 360*k0_range, lon_r1_unwrap-dlon_abs, dlon_abs) # hash origin on RHS
	# print('k_l', k_l)
	# print('k_r', k_r)
	# print('lon_l1', lon_l1)
	# print('lon_r1', lon_r1)
	# pause()

	if problem_type == 1:
		# no overlap between successive tracks
		#	lon cov region within L AND R
		k_access = k_l[k_l==k_r]

	elif problem_type == 2:
		# overlap between successive tracks (two tracks)
		#	lon cov region within L OR R
		k_access = np.unique(np.hstack((k_l,k_r)))
		# Can apply same algorithm as in 3,
		# but unique maybe faster?

	elif problem_type == 3:
		# overlap between multiple successive tracks
		#	lon cov region within L, R, and revs in-between
		# skips possible, make bridges
		k_access = []
		for j in range(len(k_l)):
			k_access.append(np.flip(np.arange(k_r[j],k_l[j]+1)))
		k_access = np.concatenate(k_access)

	if debug:

		from leocat.utils.plot import make_fig

		print('problem_type', problem_type)

		lon_l1_vec = []
		lon_r1_vec = []
		k_range = np.arange(np.min(k_access),np.max(k_access)+1)
		# k_range = np.arange(np.min([np.min(k_l),np.max(k_r)]),np.max([np.max(k_l),np.max(k_r)])+1)
		# k_range = np.arange(-10,130)
		# k_range = np.arange(0,7)
		# k0_range = np.arange(2)

		lon_l1_vec = lon_l1_unwrap + k_range*dlon_abs
		lon_r1_vec = lon_r1_unwrap + k_range*dlon_abs
		lon_vec = lon - k0_range*360

		level = int(np.ceil(np.abs(dlon_wrap)/dlon_abs))
		if level < 2:
			level = 2
		levels = np.arange(level)/8

		fig, ax = make_fig()
		for i in range(len(k_range)):
			y = levels[i % level]
			k = k_range[i]
			ax.plot([lon_l1_vec[i], lon_r1_vec[i]], [y,y], c='k')
			ax.plot(lon_l1_vec[i], y, 'g.')
			ax.plot(lon_r1_vec[i], y, 'r.')
			# if (k in k_l) and (k in k_r):
			if 1: #k in k_access:
				dy = 0.05
				vert1 = np.array([lon_l1_vec[i], lon_l1_vec[i] + dlon_abs])
				poly1 = np.array([[vert1[0],y-dy],[vert1[1],y-dy],
									[vert1[1],y+dy],[vert1[0],y+dy],
									[vert1[0],y-dy]])
				#
				vert2 = np.array([lon_r1_vec[i], lon_r1_vec[i] + dlon_abs]) - dlon_abs
				poly2 = np.array([[vert2[0],y-dy],[vert2[1],y-dy],
									[vert2[1],y+dy],[vert2[0],y+dy],
									[vert2[0],y-dy]])
				#
				ax.plot(poly1.T[0], poly1.T[1], 'g--')
				ax.plot(poly2.T[0], poly2.T[1], 'r--')

				color = 'k'
				if k in k_access:
					color = 'r'
				ax.text(vert1[0], y-dy, '%d' % (k_range[i]), va='bottom', c=color)


		# ax.set_ylim(-1,1)
		ax.set_xlabel('Unwrapped Longitude (deg)')
		ax.set_title('Vector Search Diagram')
		# ax.set_xlim(-180,180)
		ylim = ax.get_ylim()
		for lon0 in lon_vec:
			ax.plot([lon0,lon0], ylim, 'k--')


		# ax.set_xlim(-123.6262454998369, 120.13180894318236)
		# ax.set_ylim(-0.1266804274365475, 0.36750239830635206)

		fig.show()
		# pause()
		# plt.close(fig)

	return -k_access #, dlon_wrap
	# return np.sort(-k_access)



def vs_left_to_right(lon, lon_l1, lon_r1, k0_range, dlon, debug=0):
	# print('vs_left_to_right')
	# k_access = vector_search(lon, lon_l1, lon_r1, k0_range, dlon, debug=debug)
	k_access = vector_search_numba(lon, lon_l1, lon_r1, k0_range, dlon)

	# LHS = rad(lon_l1 + k_access*dlon)
	# RHS = rad(lon_r1 + k_access*dlon)
	# b = np.zeros(k_access.shape).astype(bool)
	# for k in range(len(k_access)):
	# 	b[k] = angle_in_region(rad(lon), LHS[k], RHS[k])
	# err = not b.all()
	# err = False

	# b_l1 = wrap(lon_l1 + k_access*dlon) < lon
	# b_r1 = lon < wrap(lon_r1 + k_access*dlon)
	# err = ~(b_l1 & b_r1).all()
	return k_access #, dlon_wrap, err

def vs_right_to_left(lon, lon_l1, lon_r1, k0_range, dlon, debug=0):
	# print('vs_right_to_left')
	# k_access = vector_search(lon, lon_r1, lon_l1, k0_range, dlon, debug=debug)
	k_access = vector_search_numba(lon, lon_r1, lon_l1, k0_range, dlon)

	# LHS = rad(lon_r1 + k_access*dlon)
	# RHS = rad(lon_l1 + k_access*dlon)
	# b = np.zeros(k_access.shape).astype(bool)
	# for k in range(len(k_access)):
	# 	b[k] = angle_in_region(rad(lon), LHS[k], RHS[k])
	# err = not b.all()
	# err = False

	# b_l1 = wrap(lon_r1 + k_access*dlon) < lon
	# b_r1 = lon < wrap(lon_l1 + k_access*dlon)
	# err = ~(b_l1 & b_r1).all()
	return k_access #, dlon_wrap, err

def check_vec_nan(vec):
	if np.isnan(vec).any():
		return 0
	return 1

def plot_intersections(ax, vals, labels, fontsize=9, space='  '):
	for i in range(len(vals)):
		# AT_l1, AT_GT1, AT_r1, CT_l1, CT_GT1, CT_r1
		vec = vals[i]
		if check_vec_nan(vec):
			ax.plot(vec[0], vec[1], vec[2], 'k.')
			if i != 4:
				ax.text(vec[0], vec[1], vec[2], space + labels[i], fontsize=fontsize)
			else:
				ax.text(vec[0], vec[1], vec[2], space + labels[i], fontsize=fontsize, va='top')


def get_AT(orb, t_l1, t_GT1, t_r1):
	"""
	AT_l1/2 and AT_r1/2 correspond to u_l1/2 and u_r1/2
		position along-track s.t. CT geodesic
		intersects GP latitude at half swath
		width
	"""
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', category=UserWarning)
		AT_l1, v_l1 = orb.propagate(t_l1)
		AT_GT1, v_GT1 = orb.propagate(t_GT1)
		AT_r1, v_r1 = orb.propagate(t_r1)

	v_l1, v_GT1, v_r1 = v_l1*R_earth/mag(AT_l1), v_GT1*R_earth/mag(AT_GT1), v_r1*R_earth/mag(AT_r1)
	AT_l1 = unit(AT_l1)*R_earth
	AT_GT1 = unit(AT_GT1)*R_earth
	AT_r1 = unit(AT_r1)*R_earth

	return AT_l1, AT_GT1, AT_r1, v_l1, v_GT1, v_r1


def get_CT(t_l1, t_GT1, t_r1, lon_l1, lon_GT1, lon_r1, JD1, lat):
	"""
	CT_l1/2 and CT_r1/2 correspond to lon_l1/2, lon_r1/2
		intersected position of GP latitude and
		CT geodesics at t_l1/2 and t_r1/2, at half
		swath width
		then if lon_l1/2 < lon < lon_r1/2, lon is 
		accessed around t_GT1/2

	CT_GT1/2 == AT_GT1/2

	"""
	CT_l1 = np.array([np.nan,np.nan,np.nan])
	CT_GT1 = np.array([np.nan,np.nan,np.nan])
	CT_r1 = np.array([np.nan,np.nan,np.nan])
	if not np.isnan(t_l1):
		CT_l1 = RADEC_to_cart(lon_l1 + get_GMST(JD1 + t_l1/86400), lat) * R_earth
	if not np.isnan(t_GT1):
		CT_GT1 = RADEC_to_cart(lon_GT1 + get_GMST(JD1 + t_GT1/86400), lat) * R_earth
	if not np.isnan(t_r1):
		CT_r1 = RADEC_to_cart(lon_r1 + get_GMST(JD1 + t_r1/86400), lat) * R_earth
	return CT_l1, CT_GT1, CT_r1

def CT_geo_to_lonlat(CT_geo0, t_geo0, JD1):
	CT_geo_ecf = convert_ECI_ECF(np.full(len(CT_geo0), t_geo0/86400 + JD1), CT_geo0)
	lon_CT_geo, lat_CT_geo = cart_to_RADEC(CT_geo_ecf)
	lon_CT_geo = wrap(lon_CT_geo)
	return lon_CT_geo, lat_CT_geo

def get_sc_lonlat(orb, t, JD1):
	r_eci, v_eci = orb.propagate(t)
	r_eci_gt = unit(r_eci)*R_earth
	JD = JD1 + t/86400
	r_ecf = convert_ECI_ECF(JD,r_eci)
	lon_GT, lat_GT = cart_to_RADEC(r_ecf) # spherical
	lon_GT = wrap(lon_GT)
	return lon_GT, lat_GT


# def continuous_space(orb, t_CT_space, JD1, return_CT_geo=False, debug=0):

# 	M = orb.M0 + orb.get_M_dot()*(t_CT_space-orb.t0)
# 	nu = M2nu(M,orb.e)
# 	omega = orb.omega + orb.get_omega_dot()*(t_CT_space-orb.t0)
# 	u = omega + nu
# 	LAN = orb.LAN + orb.get_LAN_dot()*(t_CT_space-orb.t0)
# 	Lam = lam + rad(get_GMST(t_CT_space/86400 + JD1)) - LAN
# 	f = np.tan(u)*np.cos(Lam) - np.cos(inc)*np.sin(Lam) - np.sin(inc)*np.tan(phi)
# 	f[np.abs(f) > 20] = np.nan

# 	idx_sol = np.where(((f[1:] > 0) & (f[:-1] < 0)) | ((f[1:] < 0) & (f[:-1] > 0)))
# 	# tn = newton(func_prime, tn0)

# 	if debug:
# 		fig, ax = make_fig()
# 		ax.plot(t_CT_space, f)
# 		ax.plot(t_CT_space[idx_sol], f[idx_sol], 'r.')
# 		xlim = ax.get_xlim()
# 		ax.plot(xlim, [0,0], 'k--')
# 		fig.show()

# 	t_CT = t_CT_space[idx_sol]
# 	lon_CT, lat_CT = get_sc_lonlat(orb, t_CT, JD1)

# 	if not return_CT_geo:
# 		return t_CT, lon_CT, lat_CT
# 	else:
# 		CT_geo = []
# 		for j in range(len(t_CT)):
# 			CT_geo.append( create_CT_geo(orb, t_CT[j]) )
# 		return t_CT, lon_CT, lat_CT, CT_geo


def plot_CT_geodesic(ax, CT_geo_list, t_geo_list, JD1, colors=None, linestyles=None):
	# CT_geo_list = [CT_geo_l1, CT_geo_r1, CT_geo_l2, CT_geo_r2]
	# t_geo_list = [t_l1+dt, t_r1+dt, t_l2+dt, t_r2+dt]

	from leocat.utils.plot import split_lon

	if colors is None:
		colors = ['C3' for i in range(len(t_geo_list))]
	if linestyles is None:
		linestyles = ['-' for i in range(len(t_geo_list))]

	for j in range(len(t_geo_list)):
		CT_geo0 = CT_geo_list[j]
		if CT_geo0 is None:
			continue
		t_geo0 = t_geo_list[j]
		lon_CT_geo, lat_CT_geo = CT_geo_to_lonlat(CT_geo0, t_geo0, JD1)
		index_CT = split_lon(lon_CT_geo)
		for idx in index_CT:
			ax.plot(lon_CT_geo[idx], lat_CT_geo[idx], c=colors[j], linestyle=linestyles[j])


def apply_split(lons, dlon, shift, swap):
	lons = deepcopy(lons)
	lons[shift] = wrap(lons[shift] - dlon)
	lons[swap[0]], lons[swap[1]] = lons[swap[1]], lons[swap[0]]
	return lons


#####################################################

def _calc_t_u(u_phi, orb, max_iter=10, debug=0):

	if np.isnan(u_phi):
		return np.nan, 0

	omega_dot = orb.get_omega_dot()
	M_dot = orb.get_M_dot()
	omega_k = orb.omega

	e = orb.e
	M0 = orb.M0
	t0 = orb.t0
	err = 0
	dM_prior = None

	k = 0
	while k < max_iter:
		nu_phi = u_phi - omega_k
		M_phi = nu2M(nu_phi,e)
		M_dot = orb.get_M_dot()
		dM = (M_phi - M0) % (2*np.pi)
		if debug:
			print(k, 'dM', dM, max_iter)
		if dM_prior is not None:
			if np.abs(dM-dM_prior) > np.pi/2:
				err = 1
				break
		t_phi = dM/M_dot
		if omega_dot == 0.0:
			break
		omega_k = orb.omega + omega_dot*(t_phi-t0)
		k += 1
		if dM_prior is None:
			dM_prior = dM

	return t_phi, err


def calc_t_u(u_phi, orb, max_iter=10, debug=0):
	if np.isnan(u_phi):
		return np.nan, 0
	t_phi, err = _calc_t_u(u_phi, orb, max_iter=max_iter, debug=debug)
	if err:
		u1 = u_phi + np.pi
		u2 = u_phi

		orb_copy = deepcopy(orb)
		t_phi1, err1 = _calc_t_u(u1, orb_copy, max_iter=max_iter, debug=debug)
		if debug:
			print('t_phi1', t_phi1)
		orb_copy.propagate_epoch(t_phi1, reset_epoch=True)

		t_phi2, err2 = _calc_t_u(u2, orb_copy, max_iter=max_iter, debug=debug)
		if debug:
			print('t_phi2', t_phi2)
		t_phi = t_phi1 + t_phi2

		err = err1 + err2

	return t_phi, err


def get_u(orb, dt):
	OE = orb.propagate(dt, return_OEs=True)
	omega = OE[-2]
	nu = OE[-1]
	u = omega + nu
	return u % (2*np.pi)

def get_du(u_est, u_true):
	du = (u_est - u_true) % (2*np.pi)
	if du > np.pi:
		du = 2*np.pi - du
	return du


def fix_split(inc, u_init, dlon, Tn, lons0, us0, ts0, verbose=0):

	"""
	Problems
	1. splits
	2. If on equator, solns 1 and 2 get flipped
	3. inc_lat > |lat|
	4. equator
	5. variable Tn

	"""

	# t_r1, t_l1, t_GT1 = ts0['r1'], ts0['l1'], ts0['GT1']
	# t_r2, t_l2, t_GT2 = ts0['r2'], ts0['l2'], ts0['GT2']
	# u_r1, u_l1, u_GT1 = us0['r1'], us0['l1'], us0['GT1']
	# u_r2, u_l2, u_GT2 = us0['r2'], us0['l2'], us0['GT2']
	t_r1, t_l1 = ts0['r1'], ts0['l1'],
	t_r2, t_l2 = ts0['r2'], ts0['l2']
	u_r1, u_l1 = us0['r1'], us0['l1']
	u_r2, u_l2 = us0['r2'], us0['l2']

	if verbose:
		print('split')
	if inc <= np.pi/2:
		# prograde
		"""
		If inc=90 deg:
		Occurs when u_init=0/180, when inc=90, and lat=0
			and at u_init=180, round-off prevents it
		Assume current track case
		"""
		if verbose:
			print('prograde')
		if (0.0 <= u_init < np.pi/2) or (3*np.pi/2 < u_init < 2*np.pi):
			# sol1
			if verbose:
				print('sol1')
			shift = 'l2'
			swap = ['l1','l2']
			lons1 = apply_split(lons0, dlon, shift, swap)
			# dlon11 = wrap(lons1['l1'] - lons1['r1'])
			# dlon12 = wrap(lons1['l2'] - lons1['r2'])

			# lon_r1, lon_l1, lon_GT1 = lons1['r1'], lons1['l1'], lons1['GT1']
			# lon_r2, lon_l2, lon_GT2 = lons1['r2'], lons1['l2'], lons1['GT2']
			lon_r1, lon_l1 = lons1['r1'], lons1['l1']
			lon_r2, lon_l2 = lons1['r2'], lons1['l2']
			t_l2 -= Tn
			t_l1, t_l2 = t_l2, t_l1
			u_l1, u_l2 = u_l2, u_l1
		
		# elif (np.pi/2 < u_init < 3*np.pi/2):
		else:
			# sol2
			if verbose:
				print('sol2')
			shift = 'r2'
			swap = ['r1','r2']
			lons2 = apply_split(lons0, dlon, shift, swap)
			# dlon21 = wrap(lons2['l1'] - lons2['r1'])
			# dlon22 = wrap(lons2['l2'] - lons2['r2'])

			# lon_r1, lon_l1, lon_GT1 = lons2['r1'], lons2['l1'], lons2['GT1']
			# lon_r2, lon_l2, lon_GT2 = lons2['r2'], lons2['l2'], lons2['GT2']
			lon_r1, lon_l1 = lons2['r1'], lons2['l1']
			lon_r2, lon_l2 = lons2['r2'], lons2['l2']
			t_r2 -= Tn
			t_r1, t_r2 = t_r2, t_r1
			u_r1, u_r2 = u_r2, u_r1

		# else:
		# 	# u on +/-np.pi/2
		# 	import warnings
		# 	warnings.warn('u_init at +/-pi/2 (1)')

	elif inc > np.pi/2:
		# retrograde
		if verbose:
			print('retrograde')
		if (0.0 <= u_init < np.pi/2) or (3*np.pi/2 < u_init < 2*np.pi):
			# sol1
			if verbose:
				print('sol1')
			shift = 'r2'
			swap = ['r1','r2']
			lons2 = apply_split(lons0, dlon, shift, swap)
			# dlon21 = wrap(lons2['l1'] - lons2['r1'])
			# dlon22 = wrap(lons2['l2'] - lons2['r2'])

			# lon_r1, lon_l1, lon_GT1 = lons2['r1'], lons2['l1'], lons2['GT1']
			# lon_r2, lon_l2, lon_GT2 = lons2['r2'], lons2['l2'], lons2['GT2']
			lon_r1, lon_l1 = lons2['r1'], lons2['l1']
			lon_r2, lon_l2 = lons2['r2'], lons2['l2']
			t_r2 -= Tn
			t_r1, t_r2 = t_r2, t_r1
			u_r1, u_r2 = u_r2, u_r1

		# elif (np.pi/2 < u_init < 3*np.pi/2):
		else:
			# sol2
			if verbose:
				print('sol2')
			shift = 'l2'
			swap = ['l1','l2']
			lons1 = apply_split(lons0, dlon, shift, swap)
			# dlon11 = wrap(lons1['l1'] - lons1['r1'])
			# dlon12 = wrap(lons1['l2'] - lons1['r2'])

			# lon_r1, lon_l1, lon_GT1 = lons1['r1'], lons1['l1'], lons1['GT1']
			# lon_r2, lon_l2, lon_GT2 = lons1['r2'], lons1['l2'], lons1['GT2']
			lon_r1, lon_l1 = lons1['r1'], lons1['l1']
			lon_r2, lon_l2 = lons1['r2'], lons1['l2']
			t_l2 -= Tn
			t_l1, t_l2 = t_l2, t_l1
			u_l1, u_l2 = u_l2, u_l1

		# else:
		# 	# u on +/-np.pi/2
		# 	import warnings
		# 	warnings.warn('u_init at +/-pi/2 (2)')

	# else:
		
	# 	# import warnings
	# 	# warnings.warn('inc at 90 deg')


	lons0 = {'r1': lon_r1, 'l1': lon_l1,
			'r2': lon_r2, 'l2': lon_l2}
	#
	us0 = {'r1': u_r1, 'l1': u_l1,
			'r2': u_r2, 'l2': u_l2}
	#
	ts0 = {'r1': t_r1, 'l1': t_l1,
			'r2': t_r2, 'l2': t_l2}
	#
	# lons0 = {'r1': lon_r1, 'l1': lon_l1, 'GT1': lon_GT1,
	# 		'r2': lon_r2, 'l2': lon_l2, 'GT2': lon_GT2}
	# #
	# us0 = {'r1': u_r1, 'l1': u_l1, 'GT1': u_GT1,
	# 		'r2': u_r2, 'l2': u_l2, 'GT2': u_GT2}
	# #
	# ts0 = {'r1': t_r1, 'l1': t_l1, 'GT1': t_GT1,
	# 		'r2': t_r2, 'l2': t_l2, 'GT2': t_GT2}
	# #

	return lons0, us0, ts0


def span_from_left(u_r1, u_l2, t_r1, t_l2, lon_r1, lon_l2):
	# l is valid, r is nan
	# l1 and l2 are defined
	# set l2 to r1, then l2 and r2 to nan
	u_r1 = u_l2
	u_l2 = np.nan
	t_r1 = t_l2
	t_l2 = np.nan
	lon_r1 = lon_l2
	lon_l2 = np.nan
	return u_r1, u_l2, t_r1, t_l2, lon_r1, lon_l2

def span_from_right(u_l1, u_r2, t_l1, t_r2, lon_l1, lon_r2):
	# l is nan, r is valid
	# r1 and r2 are defined
	# set r2 to l1, then r2 and l2 to nan
	u_l1 = u_r2
	u_r2 = np.nan
	t_l1 = t_r2
	t_r2 = np.nan
	lon_l1 = lon_r2
	lon_r2 = np.nan
	return u_l1, u_r2, t_l1, t_r2, lon_l1, lon_r2

