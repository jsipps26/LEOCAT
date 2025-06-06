

import numpy as np
from leocat.utils.const import *
from leocat.utils.cov import get_access_interval, get_t_access_avg, swath_to_FOV
from leocat.utils.cov import get_num_obs, get_revisit
from pyproj import CRS, Transformer

from leocat.utils.geodesy import lla_to_ecf, RADEC_to_cart, DiscreteGlobalGrid
from leocat.utils.orbit import convert_ECI_ECF
from leocat.utils.math import unit, dot, mag
from tqdm import tqdm

from numba import njit, types
from numba.typed import Dict


# @njit
# def access_calc_numba(p_hat, p, r_hat, r_ecf_sc, alpha_max, FOV):
# 	num = len(p)
# 	# access = {}
# 	access = Dict.empty(
# 		key_type=types.int64,
# 		value_type=types.int64[:],  # Variable-length 1D float64 arrays
# 	)
# 	# access = 0
# 	for j in range(num):

# 		p_hat0 = p_hat[j]
# 		p0 = p[j]
# 		proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
# 		dr = p0 - r_ecf_sc

# 		# dr_hat = unit(dr)
# 		# m = np.linalg.norm(dr,axis=1)

# 		m = np.sqrt(dr.T[0]**2 + dr.T[1]**2 + dr.T[2]**2)
# 		dr_hat = (dr.T/m).T
# 		# proj = dot(dr_hat,-r_hat)
# 		X, Y = dr_hat, -r_hat
# 		proj = X.T[0]*Y.T[0] + X.T[1]*Y.T[1] + X.T[2]*Y.T[2]
# 		angle = np.arccos(proj)
# 		alpha = np.arccos(proj_alpha)
# 		b = (alpha < alpha_max) & (angle < FOV/2)

# 		if b.any():
# 			idx = np.where(b)[0].astype(np.int64)
# 			access[j] = idx

# 	return access


class SimpleCoverage:
	def __init__(self, orb, swath, lon, lat, JD1, JD2):
		self.orb = orb
		self.swath = swath
		self.lon, self.lat = lon, lat
		self.JD1, self.JD2 = JD1, JD2
		self.num_pts = len(lon)
		self.dt = None

	def get_access(self, dt=None, f=0.25, verbose=1, solar_band=None, \
						warn=True, spherical=False, fast_interval=False):
		#
		# if solar_band is not None:
		t, access = self.compute_coverage(dt, f, verbose, solar_band, warn, spherical, fast_interval)
		# else:
		# t, access = self.compute_coverage_fast(dt, f, verbose, warn, spherical)
		access_interval = get_access_interval(access)
		t_access_avg = get_t_access_avg(t, access_interval)
		return t_access_avg

	def get_num_obs(self, t_access_avg, GL=None):
		num_obs = get_num_obs(t_access_avg, self.num_pts)
		return num_obs

	def get_revisit(self, t_access_avg, revisit_type='avg', GL=None):
		revisit = get_revisit(t_access_avg, self.num_pts, revisit_type=revisit_type)
		return revisit

	# def compute_coverage_fast(self, dt=None, f=0.25, verbose=1, \
	# 							warn=True, spherical=False):

	# 	"""
	# 	Calculates conical sensor access with/without solar/night illumination
	# 	constraint. 
	# 		orb - Orbit object defining spacecraft trajectory
	# 		swath - conical sensor nominal swath width at 
	# 				satellite altitude def by alt = a - R_earth
	# 		lon - grid point longitudes
	# 		lat - grid point latitudes
	# 		JD1 - Julian date of simulation start
	# 		JD2 - Julian date of simulation end

	# 		Optional inputs
	# 		dt - by default, simple_coverage finds a "good" dt
	# 			automatically based on swath size, but user can input
	# 			dt in seconds if desired
	# 		verbose - for more or less print outputs
	# 		solar_band - list in form of [lowest elevation, highest elevation],
	# 					that limits access to when the solar elevation is 
	# 					between those two values in the solar band
	# 					Leave None for None
	# 		warn - display warning messages, if any

	# 	"""

	# 	# orb, swath, lon, lat, JD1, JD2
	# 	orb = self.orb
	# 	swath = self.swath
	# 	lon, lat = self.lon, self.lat
	# 	JD1, JD2 = self.JD1, self.JD2

	# 	def _format_input_array(vec):
	# 		if not (type(vec) is np.ndarray):
	# 			vec = np.array(vec)
	# 		scalar = 0
	# 		if len(vec.shape) == 0:
	# 			vec = np.array([vec])
	# 			scalar = 1
	# 		return vec, scalar

	# 	lon, is_scalar = _format_input_array(lon)
	# 	lat, is_scalar = _format_input_array(lat)

	# 	if len(lon) != len(lat):
	# 		raise Exception('lon and lat inputs must have same length')

	# 	# Dn = orb.get_nodal_day()
	# 	# JD2 = JD1 + D*Dn/86400
	# 	# JD2 = JD1 + 2
	# 	# dt_des = 10.0 # sec
	# 	a = orb.a
	# 	dt_des = dt
	# 	if dt_des is None:
	# 		# auto-derive good dt for given footprint size
	# 		e = orb.e
	# 		if e > 0.05 and warn:
	# 			import warnings
	# 			warnings.warn(f'dt not set but orbit is eccentric (e={e}), time-step may be incorrect')
	# 		# ~circular
	# 		# f = 0.25
	# 		r_sc, v_sc = orb.r, orb.v
	# 		v_gt = mag(v_sc) * R_earth/a
	# 		t_AT = f*swath/v_gt
	# 		dt_des = t_AT

	# 	self.dt = dt_des
		
	# 	dJD_des = dt_des/86400
	# 	N = int((JD2-JD1)/dJD_des)+1
	# 	dJD = (JD2-JD1)/N
	# 	JD = np.linspace(JD1,JD2,N)
	# 	t = (JD-JD1)*86400
	# 	dt = t[1]-t[0]

	# 	# if fast_interval:
	# 	# 	from leocat.src.at import get_access_bounds, t_seg_to_idx

	# 	r_eci_sc, v_eci_sc = orb.propagate(t)
	# 	r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD, r_eci_sc, v_eci_sc)

	# 	p_lla = np.transpose([lon,lat,np.zeros(lon.shape)])
	# 	if spherical:
	# 		p = RADEC_to_cart(p_lla.T[0], p_lla.T[1]) * R_earth
	# 	else:
	# 		p = lla_to_ecf(p_lla.T[0], p_lla.T[1], p_lla.T[2])
	# 	p_hat = unit(p)

	# 	alt = a - R_earth
	# 	FOV = swath_to_FOV(swath, alt, radians=True)
	# 	r_hat = unit(r_ecf_sc)

	# 	is_solar = 0
	# 	# if solar_band is not None:
	# 	# 	is_solar = 1
	# 	# 	from leocat.utils.astro import solar_elev

	# 	alpha_max = np.arccos(R_earth/a)

	# 	# iterator = range(len(lon))
	# 	# if verbose:
	# 	# 	iterator = tqdm(iterator)
	# 	# access = {}
	# 	# for j in iterator:

	# 	# 	p_hat0 = p_hat[j]
	# 	# 	p0 = p[j]
	# 	# 	proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
	# 	# 	dr = p0 - r_ecf_sc

	# 	# 	dr_hat = unit(dr)
	# 	# 	proj = dot(dr_hat,-r_hat)
	# 	# 	angle = np.arccos(proj)
	# 	# 	alpha = np.arccos(proj_alpha)
	# 	# 	b = (alpha < alpha_max) & (angle < FOV/2)

	# 	# 	if b.any():
	# 	# 		idx = np.where(b)[0].astype(int)
	# 	# 		access[j] = idx

	# 	access = access_calc_numba(p_hat, p, r_hat, r_ecf_sc, alpha_max, FOV)

	# 	return t, access


	# # # @njit
	# # def access_calc_numba(self, p_hat, p, r_hat, r_ecf_sc, alpha_max, FOV):
	# # 	num = len(p)
	# # 	access = {}
	# # 	# access = Dict.empty(
	# # 	# 	key_type=types.int64,
	# # 	# 	value_type=types.int64[:],  # Variable-length 1D float64 arrays
	# # 	# )
	# # 	for j in tqdm(range(num)):

	# # 		p_hat0 = p_hat[j]
	# # 		p0 = p[j]
	# # 		proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
	# # 		dr = p0 - r_ecf_sc

	# # 		# dr_hat = unit(dr)
	# # 		m = np.linalg.norm(dr,axis=1)
	# # 		dr_hat = (dr.T/m).T
	# # 		# proj = dot(dr_hat,-r_hat)
	# # 		X, Y = dr_hat, -r_hat
	# # 		proj = X.T[0]*Y.T[0] + X.T[1]*Y.T[1] + X.T[2]*Y.T[2]
	# # 		angle = np.arccos(proj)
	# # 		alpha = np.arccos(proj_alpha)
	# # 		b = (alpha < alpha_max) & (angle < FOV/2)

	# # 		if b.any():
	# # 			idx = np.where(b)[0].astype(int)
	# # 			access[j] = idx

	# # 	return access

	# # # @njit
	# # def access_calc_numba(self, p_hat, p, r_hat, r_ecf_sc, alpha_max, FOV):
	# # 	access = {}
	# # 	num = len(p)
	# # 	for j in tqdm(range(num)):

	# # 		p_hat0 = p_hat[j]
	# # 		p0 = p[j]
	# # 		proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
	# # 		dr = p0 - r_ecf_sc

	# # 		dr_hat = unit(dr)
	# # 		proj = dot(dr_hat,-r_hat)
	# # 		angle = np.arccos(proj)
	# # 		alpha = np.arccos(proj_alpha)
	# # 		b = (alpha < alpha_max) & (angle < FOV/2)

	# # 		if b.any():
	# # 			idx = np.where(b)[0].astype(int)
	# # 			access[j] = idx

	# # 	return access



	def compute_coverage(self, dt=None, f=0.25, verbose=1, solar_band=None, \
						warn=True, spherical=False, fast_interval=False):

		"""
		Calculates conical sensor access with/without solar/night illumination
		constraint. 
			orb - Orbit object defining spacecraft trajectory
			swath - conical sensor nominal swath width at 
					satellite altitude def by alt = a - R_earth
			lon - grid point longitudes
			lat - grid point latitudes
			JD1 - Julian date of simulation start
			JD2 - Julian date of simulation end

			Optional inputs
			dt - by default, simple_coverage finds a "good" dt
				automatically based on swath size, but user can input
				dt in seconds if desired
			verbose - for more or less print outputs
			solar_band - list in form of [lowest elevation, highest elevation],
						that limits access to when the solar elevation is 
						between those two values in the solar band
						Leave None for None
			warn - display warning messages, if any

		"""

		# orb, swath, lon, lat, JD1, JD2
		orb = self.orb
		swath = self.swath
		lon, lat = self.lon, self.lat
		JD1, JD2 = self.JD1, self.JD2

		def _format_input_array(vec):
			if not (type(vec) is np.ndarray):
				vec = np.array(vec)
			scalar = 0
			if len(vec.shape) == 0:
				vec = np.array([vec])
				scalar = 1
			return vec, scalar

		lon, is_scalar = _format_input_array(lon)
		lat, is_scalar = _format_input_array(lat)

		if len(lon) != len(lat):
			raise Exception('lon and lat inputs must have same length')

		# Dn = orb.get_nodal_day()
		# JD2 = JD1 + D*Dn/86400
		# JD2 = JD1 + 2
		# dt_des = 10.0 # sec
		a = orb.a
		dt_des = dt
		if dt_des is None:
			# auto-derive good dt for given footprint size
			e = orb.e
			if e > 0.05 and warn:
				import warnings
				warnings.warn(f'dt not set but orbit is eccentric (e={e}), time-step may be incorrect')
			# ~circular
			# f = 0.25
			r_sc, v_sc = orb.r, orb.v
			v_gt = mag(v_sc) * R_earth/a
			t_AT = f*swath/v_gt
			dt_des = t_AT

		self.dt = dt_des
		
		dJD_des = dt_des/86400
		N = int((JD2-JD1)/dJD_des)+1
		dJD = (JD2-JD1)/N
		JD = np.linspace(JD1,JD2,N)
		t = (JD-JD1)*86400
		dt = t[1]-t[0]

		if fast_interval:
			from leocat.src.at import get_access_bounds, t_seg_to_idx

		r_eci_sc, v_eci_sc = orb.propagate(t)
		r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD, r_eci_sc, v_eci_sc)

		p_lla = np.transpose([lon,lat,np.zeros(lon.shape)])
		if spherical:
			p = RADEC_to_cart(p_lla.T[0], p_lla.T[1]) * R_earth
		else:
			p = lla_to_ecf(p_lla.T[0], p_lla.T[1], p_lla.T[2])
		p_hat = unit(p)

		alt = a - R_earth
		FOV = swath_to_FOV(swath, alt, radians=True)
		r_hat = unit(r_ecf_sc)

		is_solar = 0
		if solar_band is not None:
			is_solar = 1
			from leocat.utils.astro import solar_elev

		iterator = range(len(lon))
		if verbose:
			iterator = tqdm(iterator)

		alpha_max = np.arccos(R_earth/a)

		access = {}
		for j in iterator:

			p_hat0 = p_hat[j]
			p0 = p[j]

			# if fast_interval:
			# 	t_seg = get_access_bounds(orb, swath, lon[j], lat[j], JD1, JD2)
			# 	idx = t_seg_to_idx(t_seg, dt, t0=t[0])

			if not fast_interval:
				# t_seg = get_access_bounds(orb, swath, lon[j], lat[j], JD1, JD2)

				proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
				dr = p0 - r_ecf_sc

				dr_hat = unit(dr)
				proj = dot(dr_hat,-r_hat)
				angle = np.arccos(proj)

				alpha = np.arccos(proj_alpha)

				# q = np.ones(t.shape)
				# if is_solar:
				# 	p_lla0 = p_lla[j]
				# 	elev = solar_elev(np.full(JD.shape,p_lla0[0]), np.full(JD.shape,p_lla0[1]), JD)
				# 	b_solar = (solar_band[0] <= elev) & (elev <= solar_band[1])
				# 	q = b_solar.astype(float)
				# b = (alpha < alpha_max) & (angle < FOV/2) & (q > 0.0)
				b = (alpha < alpha_max) & (angle < FOV/2)

				if b.any():
					if is_solar:
						p_lla0 = p_lla[j]
						elev = solar_elev(np.full(b.sum(),p_lla0[0]), np.full(b.sum(),p_lla0[1]), JD[b])
						b_solar = (solar_band[0] <= elev) & (elev <= solar_band[1])
						q = b_solar.astype(float)
						if (~(q > 0.0)).all():
							continue

						b[b] = b[b] & (q > 0.0)

					idx = np.where(b)[0].astype(int)
					access[j] = idx


			else:
				t_seg = get_access_bounds(orb, swath, lon[j], lat[j], JD1, JD2)
				idx = t_seg_to_idx(t_seg, dt, t0=t[0])
				idx = np.clip(idx,0,len(r_hat)-1)
				# continue
				if len(idx) == 0:
					continue

				# print('test')

				# print(fast_interval, j, len(r_hat), len(idx))
				# r_hat_idx = r_hat[idx]

				proj_alpha = p_hat0[0]*r_hat[idx].T[0] + p_hat0[1]*r_hat[idx].T[1] + p_hat0[2]*r_hat[idx].T[2]
				dr = p0 - r_ecf_sc[idx]

				dr_hat = unit(dr)
				proj = dot(dr_hat,-r_hat[idx])
				angle = np.arccos(proj)

				alpha = np.arccos(proj_alpha)

				# q = np.ones(t.shape)
				# if is_solar:
				# 	p_lla0 = p_lla[j]
				# 	elev = solar_elev(np.full(JD.shape,p_lla0[0]), np.full(JD.shape,p_lla0[1]), JD)
				# 	b_solar = (solar_band[0] <= elev) & (elev <= solar_band[1])
				# 	q = b_solar.astype(float)
				# b = (alpha < alpha_max) & (angle < FOV/2) & (q > 0.0)
				b = (alpha < alpha_max) & (angle < FOV/2)

				if b.any():
					if is_solar:
						p_lla0 = p_lla[j]
						elev = solar_elev(np.full(b.sum(),p_lla0[0]), np.full(b.sum(),p_lla0[1]), JD[b])
						b_solar = (solar_band[0] <= elev) & (elev <= solar_band[1])
						q = b_solar.astype(float)
						if (~(q > 0.0)).all():
							continue

						b[b] = b[b] & (q > 0.0)

					# idx = np.where(b)[0].astype(int)
					# access[j] = idx
					access[j] = idx[b]


		return t, access


	# def compute_coverage_debug(self, dt=None, f=0.25, verbose=1, \
	# 							warn=True, spherical=False):

	# 	"""
	# 	Calculates conical sensor access with/without solar/night illumination
	# 	constraint. 
	# 		orb - Orbit object defining spacecraft trajectory
	# 		swath - conical sensor nominal swath width at 
	# 				satellite altitude def by alt = a - R_earth
	# 		lon - grid point longitudes
	# 		lat - grid point latitudes
	# 		JD1 - Julian date of simulation start
	# 		JD2 - Julian date of simulation end

	# 		Optional inputs
	# 		dt - by default, simple_coverage finds a "good" dt
	# 			automatically based on swath size, but user can input
	# 			dt in seconds if desired
	# 		verbose - for more or less print outputs
	# 		solar_band - list in form of [lowest elevation, highest elevation],
	# 					that limits access to when the solar elevation is 
	# 					between those two values in the solar band
	# 					Leave None for None
	# 		warn - display warning messages, if any

	# 	"""

	# 	# orb, swath, lon, lat, JD1, JD2
	# 	orb = self.orb
	# 	swath = self.swath
	# 	lon, lat = self.lon, self.lat
	# 	JD1, JD2 = self.JD1, self.JD2

	# 	def _format_input_array(vec):
	# 		if not (type(vec) is np.ndarray):
	# 			vec = np.array(vec)
	# 		scalar = 0
	# 		if len(vec.shape) == 0:
	# 			vec = np.array([vec])
	# 			scalar = 1
	# 		return vec, scalar

	# 	lon, is_scalar = _format_input_array(lon)
	# 	lat, is_scalar = _format_input_array(lat)

	# 	if len(lon) != len(lat):
	# 		raise Exception('lon and lat inputs must have same length')

	# 	# Dn = orb.get_nodal_day()
	# 	# JD2 = JD1 + D*Dn/86400
	# 	# JD2 = JD1 + 2
	# 	# dt_des = 10.0 # sec
	# 	a = orb.a
	# 	dt_des = dt
	# 	if dt_des is None:
	# 		# auto-derive good dt for given footprint size
	# 		e = orb.e
	# 		if e > 0.05 and warn:
	# 			import warnings
	# 			warnings.warn(f'dt not set but orbit is eccentric (e={e}), time-step may be incorrect')
	# 		# ~circular
	# 		# f = 0.25
	# 		r_sc, v_sc = orb.r, orb.v
	# 		v_gt = mag(v_sc) * R_earth/a
	# 		t_AT = f*swath/v_gt
	# 		dt_des = t_AT

	# 	self.dt = dt_des
		
	# 	dJD_des = dt_des/86400
	# 	N = int((JD2-JD1)/dJD_des)+1
	# 	dJD = (JD2-JD1)/N
	# 	JD = np.linspace(JD1,JD2,N)
	# 	t = (JD-JD1)*86400
	# 	dt = t[1]-t[0]

	# 	# if fast_interval:
	# 	# 	from leocat.src.at import get_access_bounds, t_seg_to_idx

	# 	r_eci_sc, v_eci_sc = orb.propagate(t)
	# 	r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD, r_eci_sc, v_eci_sc)

	# 	p_lla = np.transpose([lon,lat,np.zeros(lon.shape)])
	# 	if spherical:
	# 		p = RADEC_to_cart(p_lla.T[0], p_lla.T[1]) * R_earth
	# 	else:
	# 		p = lla_to_ecf(p_lla.T[0], p_lla.T[1], p_lla.T[2])
	# 	p_hat = unit(p)

	# 	alt = a - R_earth
	# 	FOV = swath_to_FOV(swath, alt, radians=True)
	# 	r_hat = unit(r_ecf_sc)

	# 	is_solar = 0
	# 	# if solar_band is not None:
	# 	# 	is_solar = 1
	# 	# 	from leocat.utils.astro import solar_elev

	# 	alpha_max = np.arccos(R_earth/a)

	# 	# iterator = range(len(lon))
	# 	# if verbose:
	# 	# 	iterator = tqdm(iterator)
	# 	# access = {}
	# 	# for j in iterator:

	# 	# 	p_hat0 = p_hat[j]
	# 	# 	p0 = p[j]
	# 	# 	proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
	# 	# 	dr = p0 - r_ecf_sc

	# 	# 	dr_hat = unit(dr)
	# 	# 	proj = dot(dr_hat,-r_hat)
	# 	# 	angle = np.arccos(proj)
	# 	# 	alpha = np.arccos(proj_alpha)
	# 	# 	b = (alpha < alpha_max) & (angle < FOV/2)

	# 	# 	if b.any():
	# 	# 		idx = np.where(b)[0].astype(int)
	# 	# 		access[j] = idx

	# 	return p_hat, p, r_hat, r_ecf_sc, alpha_max, FOV
