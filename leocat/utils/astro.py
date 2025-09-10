

import numpy as np
from leocat.utils.const import *
from leocat.utils.geodesy import lla_to_ecf, RADEC_to_cart
from leocat.utils.math import matmul
from leocat.utils.orbit import get_R_ECI_ECF_GMST


def solar_elev(lon, lat, JD, R_ECI_ECF=None, positive=False, mean_sun=False, spherical=False):

	# single_value = 0
	# if not (type(lon) is np.ndarray):
	# 	single_value = 1
	# 	lon = np.array([lon])
	# 	lat = np.array([lat])
	# 	JD = np.array([JD])
	# 	if R_ECI_ECF is not None:
	# 		R_ECI_ECF = np.array([R_ECI_ECF])

	if spherical:
		r_ecf = RADEC_to_cart(lon, lat) # km
	else:
		r_ecf = lla_to_ecf(lon, lat, np.zeros(lon.shape)) # km
		
	if R_ECI_ECF is None:
		R_ECI_ECF = get_R_ECI_ECF_GMST(JD)

	R_ECF_ECI = np.transpose(R_ECI_ECF, axes=(0,2,1))

	r_eci = matmul(R_ECF_ECI, r_ecf) # km
	r_sun = solar_pos_approx(JD) # km

	if mean_sun:
		s_eci = r_sun
		s_ecf_xy = matmul(R_ECI_ECF, s_eci)
		s_ecf_xy.T[2] = 0.0
		r_sun_xy = (s_ecf_xy.T / np.linalg.norm(s_ecf_xy,axis=1)).T * R_earth
		r_sun = matmul(R_ECF_ECI, r_sun_xy)

	z_hat = (r_eci.T / np.linalg.norm(r_eci, axis=1)).T # up, geocentric
	s_hat = (r_sun.T / np.linalg.norm(r_sun, axis=1)).T # towards sun

	proj_sun = np.einsum('ij,ij->i', z_hat, s_hat) # ut.dot
	proj_sun[proj_sun > 1.0] = 1.0
	proj_sun[proj_sun < -1.0] = -1.0
	angle = np.arccos(proj_sun) * 180/np.pi
	elev = 90.0 - angle
	if positive:
		elev[proj_sun <= 0] = 0.0 # invalid angle, below horizon

	# if single_value:
	# 	elev = elev[0]

	return elev


def solar_pos_approx(JD, return_lon_eclip=False):

	# position in km from Earth at JD
	# sun position
	# 	<1% relative error in pos
	# Fundamentals of Astrodynamics and Applications by David A. Vallado
	# 	section 5.1, algorithm 29

	AU = 149597870700 # meters, wikipedia

	UT1 = (JD - 2451545.0) / 36525
	lam = 280.460 + 36000.771*UT1 # deg
	TDB = UT1 # approx
	M = 357.5291092 + 35999.05034*TDB # deg
	lam_eclip = lam + 1.914666471*np.sin(np.radians(M)) + 0.019994643*np.sin(np.radians(2*M)) # deg
	if return_lon_eclip:
		return wrap(np.radians(lam_eclip), radians=True, two_pi=True)
	r_mag = 1.000140612 - 0.016708617*np.cos(np.radians(M)) - 0.000139589*np.cos(np.radians(2*M)) # AU
	eps = 23.439291 - 0.013004*2*TDB # deg

	rx = r_mag*np.cos(np.radians(lam_eclip))
	ry = r_mag*np.cos(np.radians(eps))*np.sin(np.radians(lam_eclip))
	rz = r_mag*np.sin(np.radians(eps))*np.sin(np.radians(lam_eclip))

	r = np.transpose([rx, ry, rz]) * AU / 1e3

	return r


def sun_lon_eclip(JD):
	# JD in days
	# outputs in radians

	r_sun = solar_pos_approx(JD) # relative to Earth
	r_sun_unit = (r_sun.T / np.linalg.norm(r_sun,axis=1)).T

	x_hat = np.array([1,0,0]) # ECI
	y_hat = np.array([0,1,0])

	proj_x = np.dot(r_sun_unit, x_hat)
	proj_x[proj_x > 1] = 1
	proj_x[proj_x < -1] = -1
	angle = np.arccos(proj_x) * 180/np.pi

	proj_y = np.dot(r_sun_unit, y_hat)
	proj_y[proj_y > 1] = 1
	proj_y[proj_y < -1] = -1
	angle[proj_y < 0] = -angle[proj_y < 0] + 360

	LE = angle

	return np.radians(LE)


def lunar_elev(lon, lat, JD, R_ECI_ECF=None, positive=False, relative=True):

	# single_value = 0
	# if not type(lon) is np.ndarray:
	# 	single_value = 1
	# 	lon = np.array([lon])
	# 	lat = np.array([lat])
	# 	JD = np.array([JD])
	# 	if R_ECI_ECF is not None:
	# 		R_ECI_ECF = np.array([R_ECI_ECF])

	r_ecf = lla_to_ecf(lon, lat, np.zeros(lon.shape)) # km
	if R_ECI_ECF is None:
		R_ECI_ECF = get_R_ECI_ECF_GMST(JD) # not validated

	R_ECF_ECI = np.transpose(R_ECI_ECF, axes=(0,2,1))

	r_eci = matmul(R_ECF_ECI, r_ecf) # km
	# r_sun = solar_pos_approx(JD) # km
	r_moon = lunar_pos_approx(JD) # km, in ECI

	# Need relative pos of moon wrt given lon/lat (pt)
	#	not for solar_elev b/c sun is so far
	#	elevation varies by ~1 deg
	if relative:
		r_moon = r_moon - r_eci

	# if mean_sun:
	# 	s_eci = r_sun
	# 	s_ecf_xy = matmul(R_ECI_ECF, s_eci)
	# 	s_ecf_xy.T[2] = 0.0
	# 	r_sun_xy = (s_ecf_xy.T / np.linalg.norm(s_ecf_xy,axis=1)).T * R_earth
	# 	r_sun = matmul(R_ECF_ECI, r_sun_xy)

	z_hat = (r_eci.T / np.linalg.norm(r_eci, axis=1)).T # up, geocentric
	m_hat = (r_moon.T / np.linalg.norm(r_moon, axis=1)).T # towards moon

	proj_moon = np.einsum('ij,ij->i', z_hat, m_hat) # ut.dot
	proj_moon[proj_moon > 1.0] = 1.0
	proj_moon[proj_moon < -1.0] = -1.0
	angle = np.arccos(proj_moon) * 180/np.pi
	elev = 90.0 - angle
	if positive:
		elev[proj_moon <= 0] = 0.0 # invalid angle, below horizon

	# if single_value:
	# 	elev = elev[0]

	return elev



def lunar_pos_approx(JD, return_lon_eclip=False, R_earth=6378.137):

	"""
	<1% relative error

	"""
	def sin_deg(angle):
		return np.sin(np.radians(angle))

	def cos_deg(angle):
		return np.cos(np.radians(angle))

	# AU = 149597870700 # meters, wikipedia
	# R_earth = 6378.137 # slight difference from Vallado

	JD_TDB = JD # approx, use algorithm 16 Vallado to find TDB
	T_TDB = (JD_TDB - 2451545.0) / 36525.0

	# all in degrees
	lam_eclip = 218.32 + 481267.8813*T_TDB + \
				6.29*sin_deg(134.9 + 477198.85*T_TDB) - \
				1.27*sin_deg(259.2 - 413335.38*T_TDB) + \
				0.66*sin_deg(235.7 + 890534.23*T_TDB) + \
				0.21*sin_deg(269.9 + 954397.70*T_TDB) - \
				0.19*sin_deg(357.5 + 35999.05*T_TDB) - \
				0.11*sin_deg(186.6 + 966404.05*T_TDB)
	#
	if return_lon_eclip:
		return wrap(np.radians(lam_eclip), radians=True, two_pi=True)

	phi_eclip = 5.13*sin_deg(93.3 + 483202.03*T_TDB) + \
				0.28*sin_deg(228.2 + 960400.87*T_TDB) - \
				0.28*sin_deg(318.3 + 6003.18*T_TDB) - \
				0.17*sin_deg(217.6 - 407332.20*T_TDB)
	#
	p = 0.9508 + 0.0518*cos_deg(134.9 + 477198.85*T_TDB) + \
		0.0095*cos_deg(259.2 - 413335.38*T_TDB) + \
		0.0078*cos_deg(235.7 + 890534.23*T_TDB) + \
		0.0028*cos_deg(269.9 + 954397.70*T_TDB)
	#
	eps = 23.439291 - 0.0130042*T_TDB - 1.64e-7*T_TDB**2 + 5.04e-7*T_TDB**3
	dist_moon = R_earth/sin_deg(p)

	rx = dist_moon * (cos_deg(phi_eclip)*cos_deg(lam_eclip))
	ry = dist_moon * (cos_deg(eps)*cos_deg(phi_eclip)*sin_deg(lam_eclip) - sin_deg(eps)*sin_deg(phi_eclip))
	rz = dist_moon * (sin_deg(eps)*cos_deg(phi_eclip)*sin_deg(lam_eclip) + cos_deg(eps)*sin_deg(phi_eclip))

	if type(rx) is np.ndarray:
		r_moon = np.transpose([rx,ry,rz])
	else:
		r_moon = np.array([rx,ry,rz])

	return r_moon


def get_lunar_p(JD):
	def sin_deg(angle):
		return np.sin(np.radians(angle))
	def cos_deg(angle):
		return np.cos(np.radians(angle))

	# from lunar_pos_approx
	JD_TDB = JD # approx, use algorithm 16 Vallado to find TDB
	T_TDB = (JD_TDB - 2451545.0) / 36525.0
	p = 0.9508 + 0.0518*cos_deg(134.9 + 477198.85*T_TDB) + \
		0.0095*cos_deg(259.2 - 413335.38*T_TDB) + \
		0.0078*cos_deg(235.7 + 890534.23*T_TDB) + \
		0.0028*cos_deg(269.9 + 954397.70*T_TDB) # deg
	#
	return p

def lunar_phase(JD):
	# returns lunar phase in deg (rel to geocentric)
	# Astronomical Algorithms 1992 and Vallado 5.3.4
	# 	Astr. Alg. eq. 48.2
	# 	Astr. Alg. eq. 48.3
	from leocat.utils.geodesy import cart_to_RADEC

	r_sun = solar_pos_approx(JD)
	RA_sun, DEC_sun = cart_to_RADEC(r_sun) # deg
	r_moon = lunar_pos_approx(JD)
	RA_moon, DEC_moon = cart_to_RADEC(r_moon) # deg

	def sin_deg(angle):
		return np.sin(np.radians(angle))
	def cos_deg(angle):
		return np.cos(np.radians(angle))

	arg1 = sin_deg(DEC_sun)*sin_deg(DEC_moon)
	arg2 = cos_deg(DEC_sun)*cos_deg(DEC_moon)*cos_deg(RA_sun-RA_moon)
	E = np.arccos(arg1 + arg2) * 180/np.pi # deg, # "elongation" phi in Astr. Alg. eq. 48.2
	# phase = 180.0 - E # deg, similar to Astr. Alg. eq. 48.4.., neglects lunar latitude

	# Astr. Alg. eq. 48.3, includes lunar latitude
	phi = np.radians(E)
	r_sun_mag, r_moon_mag = mag(r_sun), mag(r_moon)
	y = r_sun_mag * np.sin(phi)
	x = r_moon_mag - r_sun_mag*np.cos(phi)
	# arg = r_sun_mag * np.sin(phi) / (r_moon_mag - r_sun_mag*np.cos(phi))
	# phase = np.arctan(arg) * 180/np.pi # deg, lunar phase
	phase = np.arctan2(y,x) * 180/np.pi # deg, lunar phase

	return phase


def lunar_illumination_fraction(phase, radians=False):
	# percent of lunar surface illuminated by sun w.r.t. center of Earth
	#	input phase as deg.
	# Astronimal Algorithms 1992 eq. 48.1
	if not radians:
		phase_rad = np.radians(phase)
	else:
		phase_rad = phase
	p_illum = (1 + np.cos(phase_rad))/2.0
	return p_illum


def surface_illumination(elev, JD, body='sun', phase=None, return_log=False):

	# Returns illumination on Earth's surface from given body
	#	body either sun or moon (considers lunar phases)
	#	illumination in lux, or luminous flux lumens/m^2
	# Vallado Ch.5 Sec. 5.3.4
	# elev in deg, should not be an absolute value

	import pandas as pd
	if body == 'sun':
		l0_coefs = [13.84, 2.70, 2.88, 2.88, 3.05, 3.74]
		l1_coefs = [262.72, 12.17, 21.81, 22.26, 13.28, 3.97]
		l2_coefs = [1447.42, -431.69, -258.11, -207.64, -45.98, -4.07]
		l3_coefs = [2797.93, -1899.83, -858.36, 1034.30, 64.33, 1.47]
		elev_thresholds = [-91, -18, -12, -5, -0.8, 5, 20, 90]

	elif body == 'moon':
		l0_coefs = [-2.79, -2.58, -1.95]
		l1_coefs = [24.27, 12.58, 4.06]
		l2_coefs = [-252.95, -42.58, -4.24]
		l3_coefs = [1321.29, 59.06, 1.56]
		elev_thresholds = [-91, -0.8, 5, 20, 90]

		def sin_deg(angle):
			return np.sin(np.radians(angle))
		def cos_deg(angle):
			return np.cos(np.radians(angle))

		# from lunar_pos_approx
		JD_TDB = JD # approx, use algorithm 16 Vallado to find TDB
		T_TDB = (JD_TDB - 2451545.0) / 36525.0
		p = 0.9508 + 0.0518*cos_deg(134.9 + 477198.85*T_TDB) + \
			0.0095*cos_deg(259.2 - 413335.38*T_TDB) + \
			0.0078*cos_deg(235.7 + 890534.23*T_TDB) + \
			0.0028*cos_deg(269.9 + 954397.70*T_TDB) # deg
		#

		if phase is None:
			phase = lunar_phase(JD)
		else:
			phase = np.full(elev.shape, phase)


	elev[elev > 90] = 90.0
	elev[elev < -90] = -90.0
	x = elev/90.0
	L = np.full(elev.shape, np.nan)
	df = pd.DataFrame({'elev': elev})
	df['bin'] = pd.cut(df['elev'], elev_thresholds)
	index_bin = df.groupby('bin',observed=False).indices
	for j,key in enumerate(index_bin):
		index = index_bin[key]
		i = j-1
		if len(index) == 0 or i < 0:
			continue

		elev_bin = elev[index]
		l0 = l0_coefs[i]
		l1 = l1_coefs[i]
		l2 = l2_coefs[i]
		l3 = l3_coefs[i]

		x_idx = x[index]
		L1 = l0 + l1*x_idx + l2*x_idx**2 + l3*x_idx**3
		if body == 'moon':
			L2 = -8.68e-3*phase[index] - 2.2e-9*phase[index]**4
			L3 = 2*np.log10(p[index]/0.951)
			L1 = L1 + L2 + L3

		L[index] = L1

	if not return_log:
		M = 10**L
		M[np.isnan(L)] = 0.0
		return M
	else:
		return L


def surface_illumination_approx(elev, phase=None, return_log=False):

	# Vallado Ch.5 Sec. 5.3.4
	# elev in deg, should not be an absolute value
	#	removed variation from JD
	#	lunar illum approx depends only on elev and phase

	eps = 1e-6
	import pandas as pd
	if phase is None:
		l0_coefs = [13.84, 2.70, 2.88, 2.88, 3.05, 3.74]
		l1_coefs = [262.72, 12.17, 21.81, 22.26, 13.28, 3.97]
		l2_coefs = [1447.42, -431.69, -258.11, -207.64, -45.98, -4.07]
		l3_coefs = [2797.93, -1899.83, -858.36, 1034.30, 64.33, 1.47]
		elev_thresholds = [-91, -18-eps, -12, -5, -0.8, 5, 20, 90]

	else:
		l0_coefs = [-2.79, -2.58, -1.95]
		l1_coefs = [24.27, 12.58, 4.06]
		l2_coefs = [-252.95, -42.58, -4.24]
		l3_coefs = [1321.29, 59.06, 1.56]
		elev_thresholds = [-91, -0.8-eps, 5, 20, 90]
		if not (phase is np.ndarray):
			phase = np.full(elev.shape, phase)


	elev[elev > 90] = 90.0
	elev[elev < -90] = -90.0
	x = elev/90.0
	L = np.full(elev.shape, np.nan)
	df = pd.DataFrame({'elev': elev})
	df['bin'] = pd.cut(df['elev'], elev_thresholds)
	index_bin = df.groupby('bin',observed=False).indices
	for j,key in enumerate(index_bin):
		index = index_bin[key]
		i = j-1
		if len(index) == 0 or i < 0:
			continue

		elev_bin = elev[index]
		l0 = l0_coefs[i]
		l1 = l1_coefs[i]
		l2 = l2_coefs[i]
		l3 = l3_coefs[i]

		x_idx = x[index]
		L1 = l0 + l1*x_idx + l2*x_idx**2 + l3*x_idx**3
		if not (phase is None):
			L2 = -8.68e-3*phase[index] - 2.2e-9*phase[index]**4
			# L3 = 2*np.log10(p[index]/0.951)
			L1 = L1 + L2 #+ L3

		L[index] = L1

	if not return_log:
		M = 10**L
		M[np.isnan(L)] = 0.0
		return M
	else:
		return L

