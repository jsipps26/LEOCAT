
import numpy as np
import os, sys

from leocat.utils.const import *
from copy import deepcopy
from leocat.utils.math import rad, wrap
from leocat.utils.index import hash_cr_DGG
from leocat.cov import t_access_to_vector, vector_to_t_access

"""
To do
Shifts cause error in swath of ~res/2
If approx=1 (from cst), could compute coverage for swath+res/2,
apply shifts, then trim analytically via
	cos(phi)*cos(Lam) = cos(u)*cos(dpsi)

"""


def trim_swath(orb, w_new, lon, lat, JD1, t_access, dt_sc=60.0):

	from leocat.utils.math import interp, unit, dot
	from leocat.utils.orbit import convert_ECI_ECF
	from leocat.utils.geodesy import RADEC_to_cart

	t_total, index_total = t_access_to_vector(t_access)
	t1, t2 = np.min(t_total), np.max(t_total)
	num = int((t2-t1)/dt_sc) + 1
	t_space = np.linspace(t1,t2,num)

	r_eci_sc, v_eci_sc = orb.propagate(t_space)
	r_ecf_sc = convert_ECI_ECF(JD1 + t_space/86400, r_eci_sc)
	r_ecf_sc_intp = interp(t_total, t_space, r_ecf_sc)

	phi = np.radians(lat)
	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
	lat_c = np.degrees(phi_c)

	p_hat = RADEC_to_cart(lon, lat_c)[index_total]
	r_hat = unit(r_ecf_sc_intp)
	proj = dot(r_hat,p_hat)
	dpsi = np.arccos(proj)
	dist = R_earth*dpsi
	b = dist < w_new/2

	t_total_w = t_total[b]
	index_total_w = index_total[b]
	t_access_w = vector_to_t_access(t_total_w, index_total_w)

	return t_access_w


def shift_MLST(lon, lat, MLST_shift, DGG=None, orb=None):
	"""
	Wrapper to shift the sun-synchronous orbit (SSO) equatorial
	crossing time (Mean local solar time of equatorial crossing,
	or MLST), by a given MLST shift. It is equivalent to a shift
	in LAN, so this function calls shift_LAN with the proper 
	adjustment from hour to angle for the given MLST.

	"""
	LAN_shift = MLST_shift/24 * 360
	return shift_LAN(LAN_shift, lon, lat, DGG=DGG, orb=orb)


def shift_LAN_orbit(LAN_shift, orb):
	LAN_shift_rad = rad(LAN_shift)
	orb_shift = deepcopy(orb)
	LAN_new = orb_shift.LAN + LAN_shift_rad
	metadata = orb_shift.metadata
	if 'JD' in metadata:
		"""
		adjust MLST, if any
			~10 sec error from LAN_to_MLST conversion
			I think b/c difference in W_EARTH vs. GMST
			in SSO definition vs. MLST conversions
		"""
		from leocat.utils.orbit import LAN_to_MLST

		# print(metadata)
		LTAN = metadata['LTAN']
		LTDN = metadata['LTDN']
		JD_MLST = metadata['JD']
		direction = metadata['direction']
		LTAN_new = LAN_to_MLST(LAN_new, JD_MLST)
		LTDN_new = LAN_to_MLST(LAN_new + np.pi, JD_MLST)
		# if direction == 'ascending':
		# 	# JD is for AN
		# 	LTAN_new = LAN_to_MLST(LAN_new, JD_MLST)
		# 	LTDN_new = LAN_to_MLST(LAN_new + np.pi, JD_MLST)
		# elif direction == 'descending':
		# 	# JD is for DN
		# 	LTAN_new = LAN_to_MLST(LAN_new + np.pi, JD_MLST)
		# 	LTDN_new = LAN_to_MLST(LAN_new, JD_MLST)

		# LTAN = metadata['LTAN']
		# JD_AN = metadata['JD_AN']
		# LTAN_new = LAN_to_MLST(LAN_new, JD_AN)
		# LTDN_new = (LTAN_new + 12) % 24
		metadata['LTAN'] = LTAN_new
		metadata['LTDN'] = LTDN_new
		# print(metadata)

	orb_shift.set_OEs(orb_shift.a, orb_shift.e, orb_shift.inc, LAN_new, orb_shift.omega, orb_shift.nu)
	return orb_shift


def shift_LAN(LAN_shift, lon, lat, DGG=None, orb=None):
	"""
	Utility function to shift primarily coverage grid
	by a change in orbit LAN. This is useful if simulating
	multiple satellites, since each satellites' access times
	are equivalent, but the change in LAN is the only difference.
	Then one can determine coverage for a single satellite,
	then copy it to other with a shift in LAN.

	"""
	if DGG is None:
		lon_shift = LAN_shift
		lon = wrap(lon + lon_shift)
	else:
		# If need to snap to grid
		lon_shift0 = LAN_shift
		dlon = DGG.dlon # vector
		cols, rows = hash_cr_DGG(lon, lat, DGG)
		# cols, rows = cr_track.T
		r_min_offset = DGG.r_min_offset
		dc_shift = np.round(lon_shift0 / dlon[rows-r_min_offset]).astype(int)
		lon_shift = dc_shift*dlon[rows-r_min_offset]
		lon = wrap(lon + lon_shift)

	if orb is not None:
		orb_shift = shift_LAN_orbit(LAN_shift, orb)
		return lon, orb_shift

	return lon


def trim_time(t_access, t1, t2, time_shift=0.0):
	t_total, index = t_access_to_vector(t_access)
	b = (t1 <= t_total) & (t_total < t2)
	t_access_trim = vector_to_t_access(t_total[b] + time_shift, index[b])
	return t_access_trim

def shift_time(t_access, time_shift):
	t_access_shift = {}
	for key in t_access:
		t_access_shift[key] = t_access[key] + time_shift
	return t_access_shift


def get_shift_dt_nu(nu_shift, orb):

	from leocat.utils.orbit import nu2M

	if np.abs(nu_shift) > 360:
		raise Exception('nu can only be shifted +/-360 deg.')

	dnu = np.radians(nu_shift)
	def get_TOF(orb, dnu):
		nu = orb.nu # nu at orb.t0, not sure what happens if orb.t0 != 0
		# OE_init = orb.propagate(0.0, return_OEs=True) # this might fix if orb.t0 != 0
		# nu = OE_init[-1]
		nu_new = nu + dnu
		M_new = nu2M(nu_new,orb.e)
		t_new = orb.t0 + (M_new-orb.M0)/orb.get_M_dot()
		dt_nu = t_new - orb.t0
		return dt_nu

	dt_nu = get_TOF(orb, dnu)
	omega_dot = orb.get_omega_dot()
	if omega_dot != 0.0:
		omega_new = omega_dot*dt_nu + orb.omega
		domega = omega_new - orb.omega
		dnu = dnu - domega
		dt_nu_new = get_TOF(orb, dnu)
		dt_nu = dt_nu_new

	# actual shift in nu given changes in omega
	# nu_shift = np.degrees(dnu)

	return dt_nu



def shift_nu(nu_shift, lon, lat, t_access, orb, JD1, JD2, JD1_buffer, DGG=None, LAN_shift=0.0,
			dt_sc=60.0, w_true=None):

	"""
	The idea of the "nu_shift" is to shift the starting position of a satellite in its
	orbit without changing any other variables (omega, LAN, time, etc.). It's possible 
	to advance nu just by setting the orbital elements, but the coverage grid has no
	analytic representation, so we must make an effective time-shift and related shifts
	in LAN to simulate a shift in orbit starting position.

	Can assume that if this function is used, t_access is 
	reckoned to one keplerian period prior
		i.e., t=0 -> minus 1 keplerian period (or -orb.get_period('kepler'))

	"""
	dt_nu = 0.0
	if nu_shift != 0.0:
		dt_nu = get_shift_dt_nu(nu_shift, orb)

	orb_shift = deepcopy(orb)
	nu_new = (orb.nu + rad(nu_shift)) % (2*np.pi)
	orb_shift.set_OEs(orb.a, orb.e, orb.inc, orb.LAN, orb.omega, nu_new)
	# orb_shift = deepcopy(orb)
	# t_new = orb_shift.t0 + dt_nu
	# orb_shift.propagate_epoch(t_new, reset_epoch=True)
	# LAN_new = orb_shift.LAN - orb.get_LAN_dot()*dt_nu
	# orb_shift.set_OEs(orb_shift.a, orb_shift.e, orb_shift.inc, LAN_new, orb_shift.omega, orb_shift.nu)
	if LAN_shift != 0.0:
		# shift_LAN_orbit compensates for changes in MLST, if any
		orb_shift = shift_LAN_orbit(LAN_shift, orb_shift)

	"""
	Shifting coverage:
	The "coverage" is quantified by lon, lat, and t_access. We can either
	1. propagate new coverage to shift forward by nu, or
	2. use a buffer region and trim coverage already computed by time

	The latter is more efficient, but it requires a buffer region, which
	makes it more complicated to preprocess. But the former might require
	new space to be made, which could be very complicated.. I think
	trimming is just simpler, and it's for sure more efficient than 
	propagating new coverage.

	We then "propagate" coverage by trimming by time based on dt_nu,
	which is the time it takes to cover nu_shift, minus any secular motion
	from omega (assuming get_shift_dt_nu is working properly). Then time
	and location along-track are taken care of. But since time has advanced,
	we must compensate for the change in LAN from orbital precession
	and the rotation of the Earth on the coverage grid. This is effectively
	accomplished by a shift in LAN in total:
		-LAN_dot*dt_nu + W_EARTH*dt_nu + LAN_shift
	where LAN_shift is any other additional LAN_shift from an external
	shift in LAN, aside from nu.

	"""
	# shift times by dt_nu, trim by JD1/2
	t_total, index = t_access_to_vector(t_access)
	dJD_nu = dt_nu/86400.0
	JD = JD1_buffer + t_total/86400.0
	b_buffer = ((JD1+dJD_nu) <= JD) & (JD < (JD2+dJD_nu))
	t_total = t_total[b_buffer] - dt_nu # trim by dt_nu
	index = index[b_buffer]
	dt_buffer = (JD1-JD1_buffer)*86400 # shift back to JD1 epoch
	t_total = t_total - dt_buffer
	# t_access_shift = vector_to_t_access(t_total, index)

	"""
	t1, t2 = np.min(t_total), np.max(t_total)
	num = int((t2-t1)/dt_sc) + 1
	t_space = np.linspace(t1,t2,num)

	r_eci_sc, v_eci_sc = orb.propagate(t_space)
	r_ecf_sc = convert_ECI_ECF(JD1 + t_space/86400, r_eci_sc)
	r_ecf_sc_intp = interp(t_total, t_space, r_ecf_sc)

	phi = np.radians(lat)
	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
	lat_c = np.degrees(phi_c)
	
	p_hat = RADEC_to_cart(lon, lat_c)[index_total]
	r_hat = unit(r_ecf_sc_intp)
	proj = dot(r_hat,p_hat)
	dpsi = np.arccos(proj)
	dist = R_earth*dpsi
	b = dist < w_new/2

	t_total_w = t_total[b]
	index_total_w = index_total[b]
	t_access_w = vector_to_t_access(t_total_w, index_total_w)
	"""

	# keys = np.array(list(t_access_shift.keys()))
	# lon, lat = lon[keys], lat[keys]

	LAN_shift_nu_rad = -orb.get_LAN_dot()*dt_nu + W_EARTH*dt_nu # rad
	LAN_shift_nu = np.degrees(LAN_shift_nu_rad)
	lon = shift_LAN(LAN_shift + LAN_shift_nu, lon, lat, DGG=DGG, orb=None)
		
	fix_swath = True
	if w_true is None:
		fix_swath = False

	if fix_swath:
		from leocat.utils.orbit import convert_ECI_ECF
		from leocat.utils.math import interp, unit, dot
		from leocat.utils.geodesy import RADEC_to_cart

		t1, t2 = np.min(t_total), np.max(t_total)
		num = int((t2-t1)/dt_sc) + 1
		t_space = np.linspace(t1,t2,num)

		r_eci_sc, v_eci_sc = orb_shift.propagate(t_space)
		r_ecf_sc = convert_ECI_ECF(JD1 + t_space/86400, r_eci_sc)
		r_ecf_sc_intp = interp(t_total, t_space, r_ecf_sc)

		phi = np.radians(lat)
		phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
		lat_c = np.degrees(phi_c)
		
		p_hat = RADEC_to_cart(lon, lat_c)[index]
		r_hat = unit(r_ecf_sc_intp)
		proj = dot(r_hat,p_hat)
		dpsi = np.arccos(proj)
		dist = R_earth*dpsi
		b = dist < w_true/2

		t_access_shift = vector_to_t_access(t_total[b], index[b])
		# from leocat.src.bt import J_NR, H_NR
		# t_total, index = t_total[b], index[b]
		# J = J_NR(t_total, orb_shift, np.radians(lon[index]), np.radians(lat[index]), JD1)
		# H = H_NR(t_total, orb_shift, np.radians(lon[index]), np.radians(lat[index]), JD1)
		# t_total = t_total - J/H
		# t_access_shift = vector_to_t_access(t_total, index)

	else:
		t_access_shift = vector_to_t_access(t_total, index)


	return lon, lat, t_access_shift, orb_shift



# def shift_nu(nu_shift, lon, lat, t_access, orb, JD1, JD2, JD1_buffer, DGG=None, LAN_shift=0.0):

# 	"""
# 	The idea of the "nu_shift" is to shift the starting position of a satellite in its
# 	orbit without changing any other variables (omega, LAN, time, etc.). It's possible 
# 	to advance nu just by setting the orbital elements, but the coverage grid has no
# 	analytic representation, so we must make an effective time-shift and related shifts
# 	in LAN to simulate a shift in orbit starting position.

# 	Can assume that if this function is used, t_access is 
# 	reckoned to one keplerian period prior
# 		i.e., t=0 -> minus 1 keplerian period (or -orb.get_period('kepler'))

# 	"""
# 	dt_nu = 0.0
# 	if nu_shift != 0.0:
# 		dt_nu = get_shift_dt_nu(nu_shift, orb)

# 	orb_shift = deepcopy(orb)
# 	nu_new = (orb.nu + rad(nu_shift)) % (2*np.pi)
# 	orb_shift.set_OEs(orb.a, orb.e, orb.inc, orb.LAN, orb.omega, nu_new)
# 	# orb_shift = deepcopy(orb)
# 	# t_new = orb_shift.t0 + dt_nu
# 	# orb_shift.propagate_epoch(t_new, reset_epoch=True)
# 	# LAN_new = orb_shift.LAN - orb.get_LAN_dot()*dt_nu
# 	# orb_shift.set_OEs(orb_shift.a, orb_shift.e, orb_shift.inc, LAN_new, orb_shift.omega, orb_shift.nu)
# 	if LAN_shift != 0.0:
# 		# shift_LAN_orbit compensates for changes in MLST, if any
# 		orb_shift = shift_LAN_orbit(LAN_shift, orb_shift)

# 	"""
# 	Shifting coverage:
# 	The "coverage" is quantified by lon, lat, and t_access. We can either
# 	1. propagate new coverage to shift forward by nu, or
# 	2. use a buffer region and trim coverage already computed by time

# 	The latter is more efficient, but it requires a buffer region, which
# 	makes it more complicated to preprocess. But the former might require
# 	new space to be made, which could be very complicated.. I think
# 	trimming is just simpler, and it's for sure more efficient than 
# 	propagating new coverage.

# 	We then "propagate" coverage by trimming by time based on dt_nu,
# 	which is the time it takes to cover nu_shift, minus any secular motion
# 	from omega (assuming get_shift_dt_nu is working properly). Then time
# 	and location along-track are taken care of. But since time has advanced,
# 	we must compensate for the change in LAN from orbital precession
# 	and the rotation of the Earth on the coverage grid. This is effectively
# 	accomplished by a shift in LAN in total:
# 		-LAN_dot*dt_nu + W_EARTH*dt_nu + LAN_shift
# 	where LAN_shift is any other additional LAN_shift from an external
# 	shift in LAN, aside from nu.

# 	"""
# 	# shift times by dt_nu, trim by JD1/2
# 	t_total, index = t_access_to_vector(t_access)
# 	dJD_nu = dt_nu/86400.0
# 	JD = JD1_buffer + t_total/86400.0
# 	b_buffer = ((JD1+dJD_nu) <= JD) & (JD < (JD2+dJD_nu))
# 	t_total = t_total[b_buffer] - dt_nu # trim by dt_nu
# 	index = index[b_buffer]
# 	dt_buffer = (JD1-JD1_buffer)*86400 # shift back to JD1 epoch
# 	t_access_shift = vector_to_t_access(t_total - dt_buffer, index)

# 	"""
# 	t1, t2 = np.min(t_total), np.max(t_total)
# 	num = int((t2-t1)/dt_sc) + 1
# 	t_space = np.linspace(t1,t2,num)

# 	r_eci_sc, v_eci_sc = orb.propagate(t_space)
# 	r_ecf_sc = convert_ECI_ECF(JD1 + t_space/86400, r_eci_sc)
# 	r_ecf_sc_intp = interp(t_total, t_space, r_ecf_sc)

# 	phi = np.radians(lat)
# 	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
# 	lat_c = np.degrees(phi_c)
	
# 	p_hat = RADEC_to_cart(lon, lat_c)[index_total]
# 	r_hat = unit(r_ecf_sc_intp)
# 	proj = dot(r_hat,p_hat)
# 	dpsi = np.arccos(proj)
# 	dist = R_earth*dpsi
# 	b = dist < w_new/2

# 	t_total_w = t_total[b]
# 	index_total_w = index_total[b]
# 	t_access_w = vector_to_t_access(t_total_w, index_total_w)
# 	"""

# 	# keys = np.array(list(t_access_shift.keys()))
# 	# lon, lat = lon[keys], lat[keys]

# 	LAN_shift_nu_rad = -orb.get_LAN_dot()*dt_nu + W_EARTH*dt_nu # rad
# 	LAN_shift_nu = np.degrees(LAN_shift_nu_rad)
# 	lon = shift_LAN(LAN_shift + LAN_shift_nu, lon, lat, DGG=DGG, orb=None)
	
# 	return lon, lat, t_access_shift, orb_shift



