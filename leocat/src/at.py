
import numpy as np
from leocat.utils.const import *
from leocat.utils.math import arcsin, unit
from leocat.utils.geodesy import RADEC_to_cart
from leocat.utils.orbit import get_GMST
from leocat.utils.cov import FOV_to_swath, swath_to_FOV
from leocat.utils.index import hash_index

from numba import njit, types

"""
Analytic time of access of the orbit, not the spacecraft
	s/c time of access is a subset of the orbit

Implemented "viewing cone" based on source:
C. Han, X. Gao, and X. Sun, “Rapid satellite-to-site visibility determination 
based on self-adaptive interpolation technique,” Sci. China Technol. Sci., 
vol. 60, no. 2, pp. 264–270, Feb. 2017, doi: 10.1007/s11431-016-0513-8.
	In this code we have further adapted to compensate for J2,
	e.g., the secular motion of LAN

This is the "minimum elevation" method, which finds intervals
of time when the orbit is visible from a point on Earth. At worst, it
is an overestimation of access time windows. At best, it is exact to
true access time windows. Generally, larger swath/FOVs are exact,
while it overestimates small swath access times. Even so, this 
module heavily reduces the overall search space for the sat-to-site
visibility problem.

"""


def get_access_bounds(orb, swath, lon, lat, JD1, JD2):
	FOV = swath_to_FOV(swath, orb.get_alt())
	theta0 = get_theta0(orb, FOV)
	gamma0 = get_gamma(orb, theta0)
	t_int0, angle_rate = get_t_int0(orb, lon, lat, gamma0, JD1)
	lat_in_bounds = get_lat_in_bounds(orb, lat, FOV)
	t_seg = get_t_seg(t_int0, angle_rate, lat_in_bounds, JD1, JD2)
	return t_seg

# def t_seg_to_idx(t_seg, dt, t0=0.0):
# 	cols = np.array([])
# 	if len(t_seg) > 0:
# 		c1 = hash_index(t_seg.T[0], t0, dt)
# 		c2 = hash_index(t_seg.T[1], t0, dt)
# 		cols = []
# 		for i in range(len(t_seg)):
# 			cols0 = np.arange(c1[i],c2[i]+1)
# 			cols.append(cols0)
# 		cols = np.concatenate(cols)

# 	return cols


def t_seg_to_idx(t_seg, dt, t0=0.0):
	cols = np.array([])
	if len(t_seg) > 0:
		cols = t_seg_to_idx_numba(t_seg, dt, t0=t0)
	return cols

@njit
def t_seg_to_idx_numba(t_seg, dt, t0=0.0):
	c1 = np.floor((t_seg.T[0]-t0)/dt).astype(types.int64)
	c2 = np.floor((t_seg.T[1]-t0)/dt).astype(types.int64)
	num = np.sum(c2-c1) + len(c1)
	cols = np.zeros(num,dtype=types.int64)
	j1 = 0
	for i in range(len(t_seg)):
		cols0 = np.arange(c1[i],c2[i]+1)
		j2 = j1 + len(cols0)
		cols[j1:j2] = cols0
		j1 = j2
	return cols

def t_seg_to_t_test(t_seg, dt, t0=0.0):
	t_test = np.array([])
	if len(t_seg) > 0:
		c1 = hash_index(t_seg.T[0], t0, dt)
		c2 = hash_index(t_seg.T[1], t0, dt)
		t_test = []
		for i in range(len(t_seg)):
			cols = np.arange(c1[i],c2[i]+1)
			tau = t0 + cols*dt # + dt/2
			t_test.append(tau)
		t_test = np.concatenate(t_test)

	return t_test

def get_theta0(orb, FOV):
	alt = orb.get_alt()
	eta = np.radians(FOV/2)
	alpha0 = np.arcsin((R_earth+alt)/R_earth * np.sin(eta)) - eta
	theta0 = np.pi/2 - alpha0 - eta
	return theta0

def get_gamma(orb, theta0):
	ra = orb.a * (1+orb.e)
	q_mag = ra
	arg = R_earth * np.sin(np.pi/2 + theta0) / q_mag
	gamma1, gamma2 = arcsin(arg, offset=-theta0)
	gamma0 = gamma1
	return gamma0

def get_t_int0(orb, lon, lat, gamma0, JD1):

	p_unit = RADEC_to_cart(lon, lat)
	px, py, pz = p_unit

	inc = orb.inc
	ux = -py*np.sin(inc)
	uy = px*np.sin(inc)
	c = pz*np.cos(inc)
	u = np.sqrt(ux**2 + uy**2)
	alpha = np.arctan2(ux,uy)

	arg = (np.cos(gamma0) - c) / -u
	angle11, angle12 = arcsin(arg, offset=-alpha)
	arg = (-np.cos(gamma0) - c) / -u
	angle21, angle22 = arcsin(arg, offset=-alpha)

	theta_g0_init = np.radians(get_GMST(JD1))
	LAN_init = orb.LAN
	LAN_dot = orb.get_LAN_dot()
	angle_init = -(LAN_init - theta_g0_init)
	angle_rate = -(LAN_dot - W_EARTH)

	angle_int = np.array([angle11,angle12,angle21,angle22])
	t_int0 = np.sort( (angle_int - angle_init) / angle_rate )

	return t_int0, angle_rate

def get_lat_in_bounds(orb, lat, FOV):
	inc = orb.inc
	swath = FOV_to_swath(FOV, orb.get_alt())
	lat_in_bounds = True
	lat_GT_max = np.degrees(inc)
	if lat_GT_max > 90:
		lat_GT_max = 180-lat_GT_max
	#
	dist_lat = (np.abs(lat)-lat_GT_max)*np.pi/180 * R_earth
	if dist_lat > swath/2:
		lat_in_bounds = False
	return lat_in_bounds



@njit
def get_t_seg_n0(t_int0, angle_rate, num_days):
	t_seg1 = np.array([t_int0[0], t_int0[1]])
	t_seg2 = np.array([t_int0[2], t_int0[3]])
	t_seg = np.zeros((2*num_days,2),dtype=types.float64)
	for j in range(num_days):
		t_seg[2*j,:] = t_seg1 + 2*np.pi*j/angle_rate
		t_seg[2*j+1,:] = t_seg2 + 2*np.pi*j/angle_rate
	return t_seg

# def get_t_seg_n0(t_int0, angle_rate, num_days):
# 	t_seg1 = np.array([t_int0[0], t_int0[1]])
# 	t_seg2 = np.array([t_int0[2], t_int0[3]])
# 	t_seg = []
# 	for j in range(num_days):
# 		t_seg.append(t_seg1 + 2*np.pi*j/angle_rate)
# 		t_seg.append(t_seg2 + 2*np.pi*j/angle_rate)
# 	t_seg = np.vstack(t_seg)
# 	return t_seg

# # @njit
# # def get_t_seg_n2(t_int0, angle_rate, num_days):
# # 	b = ~np.isnan(t_int0)
# # 	# t_seg1 = np.sort(t_int0[b])
# # 	t_seg1 = t_int0[b]
# # 	t_seg = np.zeros((2*num_days,2),dtype=types.float64)
# # 	for j in range(num_days):
# # 		t_seg[j,:] = t_seg1 + 2*np.pi*j/angle_rate
# # 	return t_seg

def get_t_seg_n2(t_int0, angle_rate, num_days):
	b = ~np.isnan(t_int0)
	t_seg1 = np.sort(t_int0[b])
	t_seg = []
	for j in range(num_days):
		t_seg.append(t_seg1 + 2*np.pi*j/angle_rate)
	t_seg = np.vstack(t_seg)
	return t_seg


def get_t_seg(t_int0, angle_rate, lat_in_bounds, JD1, JD2):

	num_days = int(np.ceil(JD2-JD1)) + 1

	sim_period = (JD2-JD1)*86400
	num_nan = np.isnan(t_int0).sum()
	b = ~np.isnan(t_int0)

	t_seg = np.array([])
	if num_nan == 0:
		t_seg = get_t_seg_n0(t_int0, angle_rate, num_days)

	elif num_nan == 2:
		t_seg = get_t_seg_n2(t_int0, angle_rate, num_days)

	elif num_nan == 4:
		# either always covered or never
		if lat_in_bounds:
			# always accessed
			t_seg = np.array([[0.0, sim_period]])
		# else:
		# 	# never accessed
		# 	t_seg = []

	else:
		raise Exception('num_nan is odd')

	if len(t_seg) > 0:
		B = (0.0 <= t_seg) & (t_seg < sim_period)
		b = np.any(B,axis=1)
		t_seg = t_seg[b]

	return t_seg


# def get_t_seg(t_int0, angle_rate, lat_in_bounds, JD1, JD2):

# 	num_days = int(np.ceil(JD2-JD1)) + 1

# 	sim_period = (JD2-JD1)*86400
# 	num_nan = np.isnan(t_int0).sum()
# 	b = ~np.isnan(t_int0)

# 	t_seg = np.array([])
# 	if num_nan == 0:
# 		t_seg1 = np.array([t_int0[0], t_int0[1]])
# 		t_seg2 = np.array([t_int0[2], t_int0[3]])
# 		t_seg = []
# 		for j in range(num_days):
# 			t_seg.append(t_seg1 + 2*np.pi*j/angle_rate)
# 			t_seg.append(t_seg2 + 2*np.pi*j/angle_rate)
# 		t_seg = np.vstack(t_seg)

# 	elif num_nan == 2:
# 		# upper/lower latitudes
# 		b = ~np.isnan(t_int0)
# 		t_seg1 = np.sort(t_int0[b])
# 		t_seg = []
# 		for j in range(num_days):
# 			t_seg.append(t_seg1 + 2*np.pi*j/angle_rate)
# 		t_seg = np.vstack(t_seg)

# 	elif num_nan == 4:
# 		# either always covered or never
# 		if lat_in_bounds:
# 			# always accessed
# 			t_seg = np.array([[0.0, sim_period]])
# 		# else:
# 		# 	# never accessed
# 		# 	t_seg = []

# 	else:
# 		raise Exception('num_nan is odd')

# 	if len(t_seg) > 0:
# 		B = (0.0 <= t_seg) & (t_seg < sim_period)
# 		b = np.any(B,axis=1)
# 		t_seg = t_seg[b]

# 	return t_seg



