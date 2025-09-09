
import numpy as np
from leocat.utils.math import mag, unit, R1, R2, R3
from leocat.utils.index import mask_intervals_numba
from leocat.utils.astro import solar_pos_approx
from leocat.utils.cov import get_lat_GT_max

from leocat.utils.const import R_earth, TWO_PI
from tqdm import tqdm
from copy import deepcopy

class LatitudeCoverage:

	"""
	Info
	- For cicular orbit up to J2+frozen, uses BT for
	longitudinal projection of swath width. 
	- Swath width technically unbounded but should likely 
	remain within 10000 km.
	- Nadir viewing, e.g. FOR

	Error sources
	- Exact num revs can bias soln
	- Assume sun pos const over JD1/JD2 interval, increasing
	error near terminator

	Potential issue
	Poles assume 360 deg. coverage... maybe slightly off?

	"""

	def __init__(self, orb, swath, lat, JD0, dJD=1.0):
		self.orb = orb
		self.swath = swath
		self.JD1 = JD0 - dJD/2.0
		self.JD2 = JD0 + dJD/2.0
		self.dJD = dJD

		if not (type(lat) is np.ndarray):
			lat = np.array(lat)
		self.lat = lat
		self.phi = np.radians(lat)
		self._init_geometry()


	def _init_geometry(self):

		JD1, JD2 = self.JD1, self.JD2
		orb = self.orb
		swath = self.swath

		# get psi and cylindrical dist 
		# of swath from origin
		psi = (swath/2) / R_earth
		dist = get_cylindrical_dist(psi)
		self.psi = psi
		self.dist = dist

		# Get mean perifocal frame (p,h,q) and h_hat
		# over the simulation period
		t0 = 0.0
		JD_bar = (JD1+JD2)/2.0
		t_bar = (JD_bar-JD1)*86400
		LAN_bar = orb.LAN + orb.get_LAN_dot()*(t_bar-t0)

		R_LAN = R3(LAN_bar)
		R_inc = R1(orb.inc)
		R_omega = R3(orb.omega)
		R_313 = R_LAN @ (R_inc @ R_omega)

		# p_hat = R_313[:,0]
		h_hat = R_313[:,2]
		# q_hat = np.cross(h_hat,p_hat)

		self.JD_bar = JD_bar
		# self.p_hat = p_hat
		self.h_hat = h_hat
		# self.q_hat = q_hat


	def get_swath_lat(self, BT=True):

		orb = self.orb
		swath = self.swath
		phi = self.phi

		swath_lat = np.zeros(phi.shape)
		if BT:
			lat_GT_max = get_lat_GT_max(orb)
			dlat_swath = swath/R_earth * 180/np.pi # both sides, deg
			lat_peak = lat_GT_max - dlat_swath/2

			lat = self.lat
			for i,lat0 in enumerate(lat):
				swath_lat0 = get_swath_lat(orb, lat0, swath)
				swath_lat0 = swath_lat0 * 2 # assume both asc/desc
				if (lat0 < -lat_peak) or (lat0 > lat_peak):
					swath_lat0 = swath_lat0 / 2.0 # no asc/desc, above lat_GT_max
				swath_lat[i] = swath_lat0

		else:
			dist = self.dist
			psi = self.psi
			phi = self.phi
			h_hat = self.h_hat

			for i,phi0 in enumerate(phi):
				bounds_w = get_swath_bounds(orb, swath, phi0, h_hat, psi, dist)
				radius = R_earth*np.cos(phi0)
				# swath_lat0 = 0.0
				if len(bounds_w) > 0:
					dlam_total = np.sum(np.diff(bounds_w,axis=1))
					swath_lat0 = radius*dlam_total
					swath_lat[i] = swath_lat0

		return swath_lat


	def _get_num_obs(self, BT=True, swath_lat=None):
		orb = self.orb
		lat = self.lat
		JD1, JD2 = self.JD1, self.JD2

		num_days = JD2-JD1
		Tn = orb.get_period('nodal')
		Dn = orb.get_nodal_day()

		if swath_lat is None:
			swath_lat = self.get_swath_lat(BT=BT)
		radius_perp = R_earth * np.cos(np.radians(lat))
		C = 2*np.pi*radius_perp
		# Q_solar = 86400/Tn * 86400/Dn # revs in 1 solar day
		Q_solar = Dn/Tn # revs in 1 solar day
		num_obs_lat_est = Q_solar*num_days * swath_lat/C

		return num_obs_lat_est


	def _get_frac_lat_solar(self, elev=0.0, day=True):

		dist = self.dist
		psi = self.psi
		phi = self.phi
		h_hat = self.h_hat
		orb = self.orb
		swath = self.swath

		JD_bar = self.JD_bar
		s_bar = solar_pos_approx(JD_bar)
		s_hat = unit(s_bar)
		EL0 = np.radians(elev)

		frac = np.zeros(phi.shape)
		# ss_vec = np.zeros(phi.shape, dtype=bool)
		for i,phi0 in enumerate(phi):
			r_int_s, valid_s = get_int_r_s(phi0, s_hat, EL0)
			split_solar = False
			if valid_s:
				dist_s = np.abs(np.dot(r_int_s,h_hat))
				split_solar = (dist_s < dist).any()
				# ss_vec[i] = split_solar

			# if not split_solar:
			# 	continue

			# if split_solar, continue to find boundary for solar
			bounds_s_day, bounds_s_night = \
				get_solar_bounds(r_int_s, valid_s, phi0, s_hat, EL0)
			#
			bounds_s_mask = bounds_s_night
			if not day:
				bounds_s_mask = bounds_s_day
			bounds_w = get_swath_bounds(orb, swath, phi0, h_hat, psi, dist)

			# if split_solar, continue to intersect swath and solar boundaries
			if len(bounds_w) > 0:
				bounds_w2 = mask_intervals_numba(bounds_w, bounds_s_mask)
			else:
				bounds_w2 = bounds_w

			if len(bounds_w) > 0 and len(bounds_w2) > 0:
				num = np.sum(np.diff(bounds_w2))
				den = np.sum(np.diff(bounds_w))
				frac0 = num / den
				frac[i] = frac0

		return frac #, ss_vec


	def get_num_obs(self, elev=None, day=True, swath_lat=None):
		num_obs = self._get_num_obs(BT=True, swath_lat=swath_lat)
		if elev is None:
			return num_obs
		frac = self._get_frac_lat_solar(elev=elev, day=day)
		return frac*num_obs


	def get_series(self, JD1, JD2, elev=None, day=True, verbose=1):
		dJD = self.dJD
		lat = self.lat
		orb = self.orb
		swath = self.swath

		N = int((JD2-JD1) / dJD)
		dJD_true = (JD2-JD1)/N
		JD = JD1 + np.arange(N)*dJD_true + dJD_true/2.0
		factor = dJD/dJD_true

		iterator = range(len(JD))
		if verbose:
			iterator = tqdm(iterator)

		xx, yy = np.meshgrid(JD,lat)
		xx, yy = xx.T, yy.T
		zz = []
		orb_prop = deepcopy(orb)
		swath_lat = self.get_swath_lat(BT=True)
		for i in iterator:
			orb_prop.propagate_epoch(dJD_true*86400, reset_epoch=True)
			LC = LatitudeCoverage(orb_prop, swath, lat, JD[i])
			num_obs = LC.get_num_obs(elev=elev, day=day, swath_lat=swath_lat)
			zz.append(factor*num_obs)
		zz = np.array(zz)

		return xx, yy, zz


	def extend_meshgrid(self, lon, num_obs):
		lat = self.lat
		xx, yy = np.meshgrid(lon, lat)
		xx, yy = xx.T, yy.T
		z = np.tile(num_obs,len(lon))
		zz = z.reshape(xx.shape)
		return xx, yy, zz

	def interp_lat(self, lat, num_obs):
		lat_input = self.lat
		z = np.interp(lat, lat_input, num_obs)
		lat_min, lat_max = np.min(lat_input), np.max(lat_input)
		b = (lat_min < lat) & (lat < lat_max)
		z[~b] = 0.0
		return z




################################################################


def get_cylindrical_dist(psi0):
	# dist of cylindrical swath from origin
	# r0 = R_earth*np.sin(psi0)*h_hat
	# dist = np.abs(np.dot(r0,h_hat))
	dist = R_earth*np.abs(np.sin(psi0))
	return dist

def get_swath_lat(orb, lat0, swath):
	from leocat.src.bt import get_swath_params, classify_bridges, get_dlon_lons
	from leocat.utils.time import date_to_jd

	JD1 = date_to_jd(2021,1,1)
	lons0, us0, ts0, split, invalid_left, invalid_right, lat_in_bounds, pole_in_view = \
		get_swath_params(orb, lat0, swath, JD1, verbose=0)
	#
	bridge_class1, bridge_class2, lons0, us0, ts0 = \
		classify_bridges(lons0, us0, ts0, orb, lat0, split, invalid_left, invalid_right, lat_in_bounds)
	#
	dlon_lat = get_dlon_lons(orb, lons0, bridge_class1, pole_in_view)
	radius_perp = R_earth * np.cos(np.radians(lat0))
	swath_app = radius_perp*np.radians(dlon_lat)
	return swath_app


def intersect_circles(L1, L2):
	n_hat1, theta1 = L1
	n_hat2, theta2 = L2

	r1 = R_earth*np.sin(theta1)*n_hat1
	r2 = R_earth*np.sin(theta2)*n_hat2
	c1 = np.dot(n_hat1,r1)
	c2 = np.dot(n_hat2,r2)

	H = np.array([n_hat1,n_hat2])
	p0 = np.linalg.pinv(H) @ np.array([c1,c2])
	det = R_earth**2 - mag(p0)**2
	proj = np.dot(n_hat1,n_hat2)

	# print('test', det, np.abs(proj), np.abs(proj) != 1.0)
	valid = False
	if det >= 0.0 and np.abs(proj) != 1.0:
		valid = True
		l_hat = unit(np.cross(n_hat1,n_hat2))
		r_int1 = p0 + l_hat*np.sqrt(det)
		r_int2 = p0 - l_hat*np.sqrt(det)
	else:
		r_int1 = np.full(3,np.nan)
		r_int2 = np.full(3,np.nan)

	return r_int1, r_int2, valid


def r_int_to_theta(r_int):
	x, y = r_int[0], r_int[1]
	theta = np.arctan2(y,x) % (2*np.pi)
	return theta



################################################################

def get_int_r_w(phi, h_hat, psi0):
	z_hat = np.array([0,0,1])
	r_l_int1, r_l_int2, valid_l = intersect_circles([z_hat,phi],[h_hat,-psi0])
	r_r_int1, r_r_int2, valid_r = intersect_circles([z_hat,phi],[h_hat,psi0])
	r_int_w = np.array([[r_l_int1, r_l_int2],
						[r_r_int1, r_r_int2]])
	#
	return r_int_w, valid_l, valid_r

def get_int_r_s(phi, s_hat, EL0):
	z_hat = np.array([0,0,1])
	r_s_int1, r_s_int2, valid_s = intersect_circles([z_hat,phi],[s_hat,EL0])
	r_int_s = np.array([r_s_int1, r_s_int2])
	return r_int_s, valid_s

def get_int_theta_w(r_int_w, valid_l, valid_r):
	if valid_l:
		r_l_int1, r_l_int2 = r_int_w[0]
		theta_l1, theta_l2 = r_int_to_theta(r_l_int1), r_int_to_theta(r_l_int2)
	if valid_r:
		r_r_int1, r_r_int2 = r_int_w[1]
		theta_r1, theta_r2 = r_int_to_theta(r_r_int1), r_int_to_theta(r_r_int2)

	if valid_l and valid_r:
		reg_w = np.sort([theta_l1, theta_l2, theta_r1, theta_r2])
	elif valid_l:
		reg_w = np.sort([theta_l1, theta_l2])
	elif valid_r:
		reg_w = np.sort([theta_r1, theta_r2])
	else:
		reg_w = np.array([])

	return reg_w

def get_int_theta_s(r_int_s, valid_s):
	if valid_s:
		r_s_int1, r_s_int2 = r_int_s
		theta_s1, theta_s2 = r_int_to_theta(r_s_int1), r_int_to_theta(r_s_int2)
		reg_s = np.sort([theta_s1, theta_s2])
	else:
		reg_s = np.array([])
	return reg_s

def get_theta_w_test(reg_w):
	dreg_w = reg_w[1]-reg_w[0]
	theta_w_test = reg_w[0] + dreg_w/2.0
	return theta_w_test

def get_theta_s_test(reg_s):
	dreg_s = reg_s[1]-reg_s[0]
	theta_s_test = reg_s[0] + dreg_s/2.0
	return theta_s_test


def get_pt_w_test(phi, theta_w_test):
	radius = R_earth*np.cos(phi)
	z_offset = R_earth*np.sin(phi)
	pt_w_test = np.array([radius*np.cos(theta_w_test), 
							radius*np.sin(theta_w_test), 
							z_offset])
	#
	return pt_w_test


def get_pt_s_test(phi, theta_s_test):
	radius = R_earth*np.cos(phi)
	z_offset = R_earth*np.sin(phi)
	pt_s_test = np.array([radius*np.cos(theta_s_test), 
							radius*np.sin(theta_s_test), 
							z_offset])
	#
	return pt_s_test


def get_pt_test(phi, theta_w_test, theta_s_test):
	radius = R_earth*np.cos(phi)
	z_offset = R_earth*np.sin(phi)

	pt_w_test = np.array([radius*np.cos(theta_w_test), 
							radius*np.sin(theta_w_test), 
							z_offset])
	#
	pt_s_test = np.array([radius*np.cos(theta_s_test), 
							radius*np.sin(theta_s_test), 
							z_offset])
	#
	return pt_w_test, pt_s_test


def get_bounds_solar(reg_s, s_hat, EL0, pt_s_test):
	day_test = np.dot(s_hat,pt_s_test) > R_earth*np.sin(EL0)
	if day_test:
		bounds_s_day = np.array([[reg_s[0],reg_s[1]]])
	else:
		bounds_s_day = np.array([[0.0,reg_s[0]],[reg_s[1],TWO_PI]])
	# bounds_s_night = mask_intervals_numba(np.array([[0.0,TWO_PI]]), bounds_s_day)
	# night_test = np.dot(s_hat,pt_s_test) < 0.0
	night_test = ~day_test
	if night_test:
		bounds_s_night = np.array([[reg_s[0],reg_s[1]]])
	else:
		bounds_s_night = np.array([[0.0,reg_s[0]],[reg_s[1],TWO_PI]])

	return bounds_s_day, bounds_s_night


def get_bounds_swath(reg_w, h_hat, pt_w_test, dist):
	"""
	classification near 0/2pi is analytic
		known from other region classifications
		check whether pts on lat band are between offset planes
	alternating classification known for solar and swath
		only need to test a single pt each

	bounds_w classification for nominal Case (a)
	Can assume bounds_w classification alternates
	as True/False for every region. bounds_w is a 
	superset of reg_w, where bounds_w includes 0
	and 2pi. We must extend reg_w when necessary
	to include 0 and 2pi as edges_w, then create
	bounds_w.

	If reg_w nominal 
		reg_w[0] != 0.0 and reg_w[-1] != TWO_PI
	then b_w init as alternating booleans False, True, etc.
	swath_test is from test pt which is at the
	-second- bin in bounds_w based on nominal case.
	We know the boolean alternates so we flip the
	boolean of swath_test to check whether the first
	bin in bounds_w is accessed. If so, b_w init as
	False so we flip the b_w sequence. Otherwise,
	leave b_w as-is, starts as False.

	If reg_w is not nominal, i.e. starts/ends with
	0 or 2pi, then there's subtle changes in each
	case compensating for not needing to extend 
	reg_w with edges_w as much.

	After robustly classifying bounds_w where the
	swath exists on the lat band, we check if pts
	from terminator cross within the swath. We can
	check if terminator pts are between the cylindrical
	swath as before. If any, then terminator intersects
	swath envelope, and we must mask bounds_w by 
	bounds_s.

	bounds_s is -not- required unless terminator
	intersects the swath envelope.
		Also, if no intersection, bounds_w should
		have equal intervals corresponding to 
		asc/desc pass, so only one interval is
		really necessary.

	"""

	# assuming nadir viewing, find if pt_w_test is within
	# cylindrical swath via dist from origin
	# r0 = R_earth*np.sin(psi0)*h_hat
	# dist = np.abs(np.dot(r0,h_hat))
	dist_w = np.abs(np.dot(pt_w_test,h_hat))
	swath_test = dist_w < dist


	if reg_w[0] != 0.0 and reg_w[-1] != TWO_PI:
		swath_test = ~swath_test # first segment
		b_w = (np.arange(len(reg_w)+1) % 2).astype(bool)
		if swath_test:
			b_w = ~b_w
		edges_w = np.hstack((0.0, reg_w, TWO_PI))

	elif reg_w[0] == 0.0 and reg_w[-1] != TWO_PI:
		b_w = (np.arange(len(reg_w)+0) % 2).astype(bool)
		if swath_test:
			b_w = ~b_w
		edges_w = np.hstack((reg_w, TWO_PI))

	elif reg_w[0] != 0.0 and reg_w[-1] == TWO_PI:
		swath_test = ~swath_test # first segment
		b_w = (np.arange(len(reg_w)+0) % 2).astype(bool)
		if swath_test:
			b_w = ~b_w
		edges_w = np.hstack((0.0, reg_w))

	else:
		# reg_w[0] == 0.0 and reg_w[-1] == TWO_PI
		b_w = (np.arange(len(reg_w)-1) % 2).astype(bool)
		if swath_test:
			b_w = ~b_w
		edges_w = np.copy(reg_w)

	bounds_w = np.transpose([edges_w[:-1], edges_w[1:]])[b_w]

	return bounds_w


def get_split_solar(phi, r_int_s, h_hat, dist):
	dist_s = np.abs(np.dot(r_int_s,h_hat))
	split_solar = (dist_s < dist).any()
	return split_solar




################################################################


def get_swath_bounds(orb, swath, phi0, h_hat, psi0, dist):
	r_int_w, valid_l, valid_r = get_int_r_w(phi0, h_hat, psi0)
	if valid_l or valid_r:
		# Case a or b
		reg_w = get_int_theta_w(r_int_w, valid_l, valid_r)
		theta_w_test = get_theta_w_test(reg_w)
		pt_w_test = get_pt_w_test(phi0, theta_w_test)
		bounds_w = get_bounds_swath(reg_w, h_hat, pt_w_test, dist)

	else:
		lat_in_bounds = True
		lat_GT_max = np.degrees(orb.inc)
		if lat_GT_max > 90:
			lat_GT_max = 180-lat_GT_max
		#
		lat0 = np.degrees(phi0)
		dist_lat = (np.abs(lat0)-lat_GT_max)*np.pi/180 * R_earth
		if dist_lat > swath/2:
			lat_in_bounds = False

		pole_in_view = False
		dist_pole = (90.0-lat_GT_max)*np.pi/180 * R_earth
		if dist_pole <= swath/2:
			pole_in_view = True

		if not lat_in_bounds:
			# Case c
			bounds_w = np.array([])

		elif pole_in_view:
			# Case d
			bounds_w = np.array([[0.0,TWO_PI]])

		else:
			# Case e
			bounds_w = np.array([[0.0,TWO_PI]])

	return bounds_w



def get_solar_bounds(r_int_s, valid_s, phi, s_hat, EL0):
	if valid_s:
		# lat band int with terminator
		reg_s = get_int_theta_s(r_int_s, valid_s)
		theta_s_test = get_theta_s_test(reg_s)
		pt_s_test = get_pt_s_test(phi, theta_s_test)
		bounds_s_day, bounds_s_night = get_bounds_solar(reg_s, s_hat, EL0, pt_s_test)

	else:
		"""
		no intersection with lat band
			all day or all night

		"""
		z_hat = np.array([0,0,1])
		s_offset = R_earth*np.sin(EL0)
		pt_s = s_offset*s_hat
		n_hat = z_hat
		p0 = R_earth*np.sin(phi) * z_hat
		dp = p0-pt_s
		proj = np.dot(dp,s_hat)
		if proj > 0.0:
			# day
			bounds_s_day = np.array([[0.0,TWO_PI]])
			bounds_s_night = np.array([])
		else:
			# night
			bounds_s_day = np.array([])
			bounds_s_night = np.array([[0.0,TWO_PI]])

	return bounds_s_day, bounds_s_night





