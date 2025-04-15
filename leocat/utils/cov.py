

import numpy as np
from leocat.utils.const import *
from leocat.utils.math import R1, R2, R3
# from leocat.utils.math import newton_raphson
from leocat.utils.math import unit, mag, dot, matmul

from leocat.utils.geodesy import ecf_to_lla, lla_to_ecf
from leocat.utils.geodesy import ev_direct, ev_inverse
# from leocat.utils.plot import make_fig, draw_vector, set_axes_equal, set_aspect_equal

from numba import njit

def get_lat_GT_max(orb):
	inc = np.degrees(orb.inc)
	lat_GT_max = inc
	if lat_GT_max > 90:
		lat_GT_max = 180 - lat_GT_max
	return lat_GT_max


def get_num_obs_lat(orb, swath, num_days, lat):

	"""
	Analytic avg. number of obs given orb/swath and number
	of days of simulation, for a given latitude. 

	"""
	from leocat.src.bt import get_dlon_lons, get_swath_params, classify_bridges
	from leocat.utils.time import date_to_jd

	scalar = 0
	if not isinstance(lat, np.ndarray):
		lat = np.array(lat)
		if len(lat.shape) == 0:
			# scalar
			scalar = 1
			lat = np.array([lat])

	Tn = orb.get_period('nodal')
	Dn = orb.get_nodal_day()
	dlon = 360*Tn/Dn

	lat_GT_max = get_lat_GT_max(orb)
	dlat_swath = swath/R_earth * 180/np.pi # both sides, deg
	lat_peak = lat_GT_max - dlat_swath/2

	JD1 = date_to_jd(2021,1,1)

	period = num_days
	num_obs_lat = np.zeros(lat.shape)
	for i,lat0 in enumerate(lat):
		lons0, us0, ts0, split, invalid_left, invalid_right, lat_in_bounds, pole_in_view = \
			get_swath_params(orb, lat0, swath, JD1, verbose=0)
		#
		bridge_class1, bridge_class2, lons0, us0, ts0 = \
			classify_bridges(lons0, us0, ts0, orb, lat0, split, invalid_left, invalid_right, lat_in_bounds)
		#
		dlon_lat = get_dlon_lons(lons0, bridge_class1)
		# if np.abs(lat0) < 2.0:
		# 	print(lat0, lons0, bridge_class1, dlon_lat)
		# 	pause()

		radius_perp = R_earth * np.cos(np.radians(lat0))
		swath_app = radius_perp*np.radians(dlon_lat)
		C = 2*np.pi*radius_perp
		Q_solar = 86400/Tn * 86400/Dn # revs in 1 solar day
		# Q_solar = Dn/Tn
		num_obs_lat0 = Q_solar*period * swath_app/C * 2 # *2 for asc/desc
		if bridge_class1 == 3 and not pole_in_view:
			# factor to adjust equator, only approx
			# 	single revs do not completely cover equator
			num_obs_lat0 = num_obs_lat0 / (1+dlon*np.cos(orb.inc)/360.0)

		if (lat0 < -lat_peak) or (lat0 > lat_peak):
			num_obs_lat0 = num_obs_lat0 / 2.0 # no asc/desc, above lat_GT_max

		num_obs_lat[i] = num_obs_lat0

	if scalar:
		return num_obs_lat[0]
	else:
		return num_obs_lat


def get_wall_to_wall_swath(orb):
	# Swath s.t. equator exactly covered
	# Circular orbits only
	# 	source [7]
	Dn = orb.get_nodal_day()
	Tn = orb.get_period('nodal')
	dlon = 360*Tn/Dn
	swath = np.radians(dlon)*R_earth
	Q = Dn/Tn # R/D
	arg = np.abs(np.sin(orb.inc) / (np.cos(orb.inc) - 1/Q))
	inc_app = np.arctan(arg)
	swath_w2w = swath * np.sin(inc_app)

	return swath_w2w


def get_apparent_swath(orb, swath, inverse=True):
	"""
	Apparent swath at equator
	Circular orbits only
		source [7]
	If inverse=True, returns swath that sat should have
	s.t. input swath is met given the sat inclination
	"""
	Dn = orb.get_nodal_day()
	Tn = orb.get_period('nodal')
	Q = Dn/Tn # R/D
	arg = np.abs(np.sin(orb.inc) / (np.cos(orb.inc) - 1/Q))
	inc_app = np.arctan(arg)
	if not inverse:
		swath_app = swath / np.sin(inc_app)
	else:
		swath_app = swath * np.sin(inc_app)

	# Q = Dn/Tn # R/D
	# arg = np.abs(np.sin(np.radians(inc)) / (np.cos(np.radians(inc)) - 1/Q))
	# inc_app = np.arctan(arg)
	# w = w*np.sin(inc_app)
	return swath_app


def get_num_obs_eq(orb, period, swath):
	"""
	Estimate N at equator
		wall-to-wall coverage
		period in solar days
		divide by 2 for day- or night-only?
	This might need to be mult. by nodal day/solar day
	or vise-versa
		but that's only off by like 2%

	This is nearly equivalent to
		np.mean(num_obs[at equator])
	Somehow it works even far below repeat cycle

	Can probably find day/night by dividing by 2
	but only if SSO

	"""
	C = 2*np.pi*R_earth
	Dn = orb.get_nodal_day()
	Tn = orb.get_period('nodal')
	# Q_solar = Dn/Tn # R/D, revs per nodal day
	# Q_solar = 86400 / Tn # revs in 1 solar day
	Q_solar = 86400 / Tn * 86400/Dn # revs in 1 solar day
	swath_app = get_apparent_swath(orb, swath, inverse=False)
	num_obs_eq = Q_solar*period * swath_app/C * 2 # *2 for day/night
	return num_obs_eq


def get_revisit(t_access_avg, num_pts, revisit_type='avg'):
	revisit = np.full(num_pts,np.nan)
	for key in t_access_avg:
		t_access = t_access_avg[key]
		if len(t_access) < 2:
			continue

		# revisit exists
		if revisit_type == 'avg':
			revisit[key] = np.mean(np.diff(t_access))
		elif revisit_type == 'std':
			revisit[key] = np.std(np.diff(t_access))
		elif revisit_type == 'max':
			revisit[key] = np.max(np.diff(t_access))
		elif revisit_type == 'min':
			revisit[key] = np.min(np.diff(t_access))
		elif revisit_type == 'count':
			revisit[key] = len(np.diff(t_access))

	return revisit
	
def get_num_obs(t_access_avg, num_pts):
	num_obs = np.zeros(num_pts)
	for key in t_access_avg:
		num_obs[key] = num_obs[key] + len(t_access_avg[key])
	return num_obs


def get_cst_dt_max_pt(t_vec, threshold=None, use_numba=False):
	"""
	Function dedicated just to determining dt_max or if
	dt_max is greater than a given threshold, for a constellation.

	If use_numba is False, just does standard numpy sort/concat,
	then depends on option:
	1. If calc dt_max
		does standard max(diff(t))
	2. If thresholding
		iterates over t[k]-t[k-1] until threshold fails
		at best, exits immediately
		at worst, iterates over all revisits

	If use_numba is True, tries a smart sort leveraging the
	fact that each sat time-series is sorted
	1. If calc dt_max
		computes max(dt) sequentially
	2. If thresholding
		computes max(dt) until threshold fails
		at best, exits immediately
		at worst, iterates over all revisits

	Runtime differences
	If you're running this only once, use_numba=False is much
	faster because numba has to compile which takes time.

	However, if you're running this in a loop many times:
	In general, use_numba=True is better for small time-series.
	Upper-bound num obs for LEO sat w/16 days of coverage, which
	meets typical revisit saturation, is ~56 avg. obs. globally.
	For some reason, numba-based functions have worse complexity
	than numpy but better overhead. So for small time-series, 
	use_numba=True is better.

	Input
	List of access times for each satellite, for a given grid point.

	Output
	if threshold = None:
		returns dt_max
	else:
		returns -1, 0, or 1, indicating
		-1: max revisit doesn't exist, or num. obs. < 2
		0: max revisit does not meet threshold
		1: max revisit meets threshold

	"""

	if len(t_vec) == 1:
		t = t_vec[0]
		if threshold is None:
			dt_max = np.max(np.diff(t))
			return dt_max
		else:
			within_threshold = query_within_threshold_direct_numba(t, threshold)
			if within_threshold:
				return 1
			else:
				return 0

	num = 0
	for t_vec0 in t_vec:
		num += len(t_vec0)
	if num < 2:
		if threshold is None:
			return np.nan
		else:
			return -1

	if not use_numba:
		t = np.sort(np.concatenate(t_vec))
		if threshold is None:
			dt_max = np.max(np.diff(t))
			return dt_max
		else:
			within_threshold = query_within_threshold_direct_numba(t, threshold)
			if within_threshold:
				return 1
			else:
				return 0

	else:
		if threshold is None:
			return get_cst_dt_max_pt_numba(*t_vec)
		else:
			within_threshold = get_cst_dt_max_pt_threshold_numba(threshold, *t_vec)
			if within_threshold:
				return 1
			else:
				return 0

@njit
def query_within_threshold_direct_numba(t, threshold):
	within_threshold = True
	for k in range(1,len(t)):
		if t[k]-t[k-1] > threshold:
			within_threshold = False
			break
	return within_threshold


@njit
def get_cst_dt_max_pt_numba(*arrays):
	# Initialize indices for all arrays
	indices = [0] * len(arrays)
	prev = None
	max_diff = 0.0

	while True:
		# Find the array with the current smallest available value
		min_val = None
		min_idx = -1
		for i, (arr, idx) in enumerate(zip(arrays, indices)):
			if idx < len(arr):
				val = arr[idx]
				if min_val is None or val < min_val:
					min_val = val
					min_idx = i

		# All arrays are exhausted
		if min_idx == -1:
			break

		# Compute diff
		if prev is not None:
			max_diff = max(max_diff, min_val - prev)
		prev = min_val

		# Advance the index in the array that provided min_val
		indices[min_idx] += 1

	return max_diff


@njit
def get_cst_dt_max_pt_threshold_numba(threshold, *arrays):
	# Initialize indices for all arrays
	indices = [0] * len(arrays)
	prev = None
	max_diff = 0.0

	within_threshold = True

	# count = 0
	while True:
		# Find the array with the current smallest available value
		min_val = None
		min_idx = -1
		for i, (arr, idx) in enumerate(zip(arrays, indices)):
			if idx < len(arr):
				val = arr[idx]
				if min_val is None or val < min_val:
					min_val = val
					min_idx = i

		# All arrays are exhausted
		if min_idx == -1:
			break

		# Compute diff
		if prev is not None:
			max_diff = max(max_diff, min_val - prev)
			if max_diff > threshold:
				within_threshold = False
				break
		prev = min_val

		# Advance the index in the array that provided min_val
		indices[min_idx] += 1

		# count += 1

	return within_threshold #, count



@njit
def get_dt_avg_MC_numba(t, q, N_MC):
	# Applicable to WSC
	dt_avg_MC = 0.0
	count = 0
	for i in range(N_MC):
		rnd = np.random.uniform(0,1,len(q))
		b = rnd <= q
		chi = t[b]
		if len(chi) < 2:
			continue
		dt_avg_MC = dt_avg_MC + np.mean(np.diff(chi))
		count += 1
	if count > 0:
		dt_avg_MC = dt_avg_MC / count
	else:
		dt_avg_MC = np.nan

	return dt_avg_MC


@njit
def get_dt_max_MC_numba(t, q, N_MC):
	# Applicable to WSC
	dt_avg_MC = 0.0
	count = 0
	for i in range(N_MC):
		rnd = np.random.uniform(0,1,len(q))
		b = rnd <= q
		chi = t[b]
		if len(chi) < 2:
			continue
		dt_avg_MC = dt_avg_MC + np.max(np.diff(chi))
		count += 1
	if count > 0:
		dt_avg_MC = dt_avg_MC / count
	else:
		dt_avg_MC = np.nan

	return dt_avg_MC


@njit
def get_dt_min_MC_numba(t, q, N_MC):
	# Applicable to WSC
	dt_avg_MC = 0.0
	count = 0
	for i in range(N_MC):
		rnd = np.random.uniform(0,1,len(q))
		b = rnd <= q
		chi = t[b]
		if len(chi) < 2:
			continue
		dt_avg_MC = dt_avg_MC + np.min(np.diff(chi))
		count += 1
	if count > 0:
		dt_avg_MC = dt_avg_MC / count
	else:
		dt_avg_MC = np.nan

	return dt_avg_MC


def get_stats(q):
	# num_pass = len(q)
	P = 0.0
	P_one = 0.0
	N = 0.0
	N_R_vec = []
	P_R_vec = []
	P_vec = []
	N_vec = []

	for k in range(len(q)):
		P_one = P_one*(1-q[k]) + (1-P)*q[k]
		P = 1 - (1-P)*(1-q[k])
		N = N + q[k]
		if P > 0:
			N_R_vec.append((N-P)/P)
		else:
			N_R_vec.append(np.nan)
		P_R_vec.append(P-P_one)
		P_vec.append(P)
		N_vec.append(N)
	N_R_vec = np.array(N_R_vec)
	P_R_vec = np.array(P_R_vec)
	P_vec = np.array(P_vec)
	N_vec = np.array(N_vec)

	P_R = P - P_one

	N_R0 = np.nan
	N_R = np.nan
	if P > 0:
		N_R0 = (N-P)/P
	if P_R > 0:
		N_R = (N-P)/P_R

	return N, P, N_R0, N_R, P_R

	
def get_combs(t):
	idx0 = np.arange(len(t))
	idx = []
	for j in range(2,len(idx0)+1):
		for subset in itertools.combinations(idx0,j):
			idx.append(list(subset))

	dt_vec = []
	for j,idx0 in enumerate(idx):
		dt0 = func(np.diff(t[idx0]))
		dt_vec.append(dt0)

	return idx, dt_vec


def get_p_dt(q, idx, dt_vec, P_R, return_unique=False):
	num_pass = len(q)
	p_dt = []
	for j in range(len(dt_vec)):
		p0 = []
		for k in range(num_pass):
			if k in idx[j]:
				p0.append(q[k])
			else:
				p0.append(1-q[k])
		p0 = np.prod(p0)/P_R
		p_dt.append([dt_vec[j], p0])
	p_dt = np.array(p_dt)
	p_dt = sort_col(p_dt)

	if return_unique:
		df = pd.DataFrame({'dt': np.round(p_dt.T[0],8), 'p': p_dt.T[1]})
		df_uq = df.groupby('dt').agg('sum')
		df_uq = df_uq.reset_index()
		p_dt_uq = df_uq.to_numpy()
		p_dt = p_dt_uq

	return np.fliplr(p_dt)


def get_dt_true(q, idx, dt_vec, return_unique=False):
	N, P, N_R0, N_R, P_R = get_stats(q)
	p_dt = get_p_dt(q, idx, dt_vec, P_R, return_unique=return_unique)
	dt_avg = np.dot(p_dt.T[0],p_dt.T[1])
	return dt_avg, p_dt


def get_access_interval(access):
	access_interval = {}
	for key in access:
		access_interval[key] = []

	for key in access:
		idx = access[key]
		idx_change = np.where(np.diff(idx) > 1)[0].astype(int)
		if len(idx_change) == 0:
			# only one period of access
			k1, k2 = idx[0], idx[-1]
			access_interval[key] = [[k1,k2]]

		else:
			# multiple periods of access
			# idx_change corresponds to last element in each access period
			k1 = idx[0]
			for j in range(len(idx_change)):
				k2 = idx[idx_change[j]]
				access_interval[key].append([k1,k2])
				k1 = idx[idx_change[j]+1]

			access_interval[key].append([k1,idx[-1]])

	for key in access_interval:
		access_interval[key] = np.array(access_interval[key])

	return access_interval


def get_t_access_avg(t, access_interval):
	t_access_avg = {}
	for key in access_interval:
		k1, k2 = access_interval[key].T[0], access_interval[key].T[1]
		t_access_avg[key] = []
		for j in range(len(k1)):
			t_access = t[k1[j]:k2[j]+1]
			t_access_avg[key].append(np.mean(t_access))

	for key in t_access_avg:
		t_access_avg[key] = np.array(t_access_avg[key])

	return t_access_avg


def swath_to_FOV(swath, alt, radians=False):
	alpha = swath / (2*R_earth)
	arg = np.sin(alpha) / ((R_earth+alt)/R_earth - np.cos(alpha))
	eta = np.arctan(arg)
	FOV = 2*np.degrees(eta)
	if radians:
		return np.radians(FOV)
	return FOV

def FOV_to_swath(FOV, alt, radians=False):
	if radians:
		eta = FOV/2
	else:
		eta = np.radians(FOV)/2
	alpha = np.arcsin((R_earth+alt)/R_earth * np.sin(eta)) - eta
	swath = 2*alpha * R_earth
	return swath

# def swath_to_FOV(w, alt, radians=False):

# 	# stupid approach
# 	# here's a better one
# 	# Law of Sines, spherical Earth
# 	# swath_to_FOV
# 	# theta = (w/2)/R_earth
# 	# alpha = np.arctan( R_earth*np.sin(theta) / (R_earth + h - R_earth*np.cos(theta)) )
# 	# FOV2 = np.degrees(alpha*2)

# 	# Governing eqn
# 	#	sin(alpha)/R = sin(alpha+theta)/(R+h)
# 	#	alpha = FOV/2
# 	#	w = R*theta*2

# 	r0 = np.array([R_earth + alt, 0, 0])

# 	u_hat0 = -unit(r0)
# 	r_ground0 = ray_cast_vec(np.array([r0]), np.array([u_hat0]))
# 	r_ground0 = r_ground0[0]

# 	lon0, lat0, _ = ecf_to_lla(r_ground0[0], r_ground0[1], r_ground0[2])
# 	lon1, lat1, _ = ev_direct(lon0, lat0, 90.0, w/2, radians=False, unit='km')
# 	lon2, lat2, _ = ev_direct(lon0, lat0, 270.0, w/2, radians=False, unit='km')

# 	r_ground1 = lla_to_ecf(lon1, lat1, 0)
# 	r_ground2 = lla_to_ecf(lon2, lat2, 0)

# 	u_hat1 = unit(r0 - r_ground1)
# 	u_hat2 = unit(r0 - r_ground2)

# 	proj = np.dot(u_hat1,u_hat2)
# 	FOV = np.arccos(proj)*180/np.pi

# 	if radians:
# 		FOV = np.radians(FOV)

# 	return FOV
	


# def w_from_FOV_new(FOV, alt, radians=True, debug=0):
# def FOV_to_swath(FOV, alt, radians=False, debug=0):

# 	# [208]
# 	# h = alt
# 	# psi = np.radians(FOR/2)
# 	# gamma = np.pi - np.arcsin((R_earth + h)*np.sin(psi) / R_earth)
# 	# S = R_earth * np.cos(gamma) + (R_earth + h)*np.cos(psi)
# 	# s = 2*S*np.sin(psi)

# 	# stupid approaches
# 	# here's a better one
# 	# Law of Sines, spherical Earth
# 	# FOV_to_swath
# 	# alpha = np.radians(FOV)/2
# 	# theta = np.arcsin((R_earth+h)/R_earth * np.sin(alpha)) - alpha
# 	# w = R_earth * theta * 2

# 	if not radians:
# 		FOV = np.radians(FOV)

# 	# alt = 700
# 	r0 = np.array([R_earth + alt, 0, 0])

# 	# FOV = np.radians(45.0)
# 	u_hat1 = R3(-FOV/2) @ -unit(r0)
# 	u_hat2 = R3(FOV/2) @ -unit(r0)

# 	r_ground = ray_cast_vec(np.array([r0,r0]), np.array([u_hat1,u_hat2]))
# 	lon1, lat1, _ = ecf_to_lla(r_ground[0][0], r_ground[0][1], r_ground[0][2])
# 	lon2, lat2, _ = ecf_to_lla(r_ground[1][0], r_ground[1][1], r_ground[1][2])

# 	eps = 1e-4
# 	dist = ev_inverse(lon1, lat1+eps, lon2, lat2, radians=False, dist_only=True, unit='km')

# 	if debug:
# 		from leocat.utils.plot import make_fig, draw_vector, set_axes_equal, set_aspect_equal
# 		# make 2D
# 		r0 = r0[:-1]
# 		r_ground = r_ground[:,:-1]

# 		L_plot = alt*4
# 		origin = np.mean(r_ground,axis=0)

# 		theta = np.linspace(0,2*np.pi,1000)
# 		x = R_earth*np.cos(theta)
# 		y = R_earth_pole*np.sin(theta)

# 		fig, ax = make_fig()
# 		ax.plot(x, y)
# 		ax.plot(r0[0], r0[1], '.')
# 		draw_vector(ax, r0, r_ground[0], 'k')
# 		draw_vector(ax, r0, r_ground[1], 'k')
# 		set_axes_equal(ax)
# 		set_aspect_equal(ax)
# 		ax.set_xlim(origin[0]-L_plot/2, origin[0]+L_plot/2)
# 		ax.set_ylim(origin[1]-L_plot/2, origin[1]+L_plot/2)
# 		fig.show()

# 	return dist



def project_footprint(r_ecf_sc, R_FOV, fp_geom, a=6378.137, f=1/298.257223563):

	def angle_to_vector(fp_geom):
		# convert 2d angle into unit vectors
		u_hat = []
		for (gamma_x, gamma_y) in fp_geom:
			uz = 1/np.sqrt( 1 + np.tan(gamma_x)**2 + np.tan(gamma_y)**2 )
			ux = uz*np.tan(gamma_x)
			uy = uz*np.tan(gamma_y)
			u_hat0 = np.array([ux,uy,uz])
			u_hat.append(u_hat0)
		u_hat = np.array(u_hat)
		return u_hat

	u_hat = angle_to_vector(fp_geom)

	r_corner = []
	for u_hat0 in u_hat:
		# fpt_c0 = -u_hat0
		# if LVLH:
		fpt_c0 = u_hat0
		fpt_rot_c0 = R_FOV @ fpt_c0 # Nx3
		corner = ray_cast_vec(r_ecf_sc, fpt_rot_c0, a, f) # Nx3
		r_corner.append(corner)
	r_corner = np.array(r_corner)

	return r_corner


def create_swath_simple(r_eci_gt, v_eci_gt, swath):
	x_hat_gt = unit(v_eci_gt)
	z_hat_gt = unit(r_eci_gt)
	y_hat_gt = unit(np.cross(z_hat_gt,x_hat_gt))
	R_GT = np.transpose([x_hat_gt,y_hat_gt,z_hat_gt], (1,2,0))

	# swath = 2500
	central_angle = (swath/2)/R_earth # rad
	r_l_loc = R1(-central_angle) @ np.array([0,0,R_earth]) # in local frame
	r_r_loc = R1(central_angle) @ np.array([0,0,R_earth])
	r_l = R_GT @ r_l_loc # in eci
	r_r = R_GT @ r_r_loc

	return r_l, r_r


def create_swath(lon, lat, w, unit='m'):
	"""
	Approximation
		not robust to fast footprint shape changes

	lon/lat in degrees
	swath in meters
		w is entire swath

	returns
		lon_right, lat_right, lon_left, lat_left
	"""
	if unit == 'km':
		w = w*1e3 # turn km to m

	lon_rad = np.radians(lon)
	lat_rad = np.radians(lat)

	num_iter = 5
	_, azim_start, azim_end = ev_inverse(lon_rad[:-1], lat_rad[:-1], lon_rad[1:], lat_rad[1:], max_iter=num_iter)
	azim_start = np.hstack([azim_start[0], azim_start])
	azim_end = np.hstack([azim_end[0], azim_end])

	swath = np.ones(lon_rad.shape) * w/2

	# east
	alpha = azim_start + np.pi/2
	lon_right, lat_right, _ = ev_direct(lon_rad, lat_rad, alpha, swath, max_iter=num_iter)

	# west
	alpha = azim_start - np.pi/2
	lon_left, lat_left, _ = ev_direct(lon_rad, lat_rad, alpha, swath, max_iter=num_iter)

	return np.degrees(lon_right), np.degrees(lat_right), \
				np.degrees(lon_left), np.degrees(lat_left)


def get_dlon_swath(w, lon=0, lat=0):
	# w_iter in km
	# assumes 90 deg inclination polar orbit
	#	dlon_w needs to vary by inclination
	L1, phi1 = np.radians(lon), np.radians(lat)
	alpha1 = np.pi/2
	L2, phi2, alpha2 = ev_direct(L1, phi1, alpha1, s=w*1e3)
	dlon_w = np.abs(L2-L1)*180/np.pi
	return dlon_w


def spherical_swath(alt):
	# r_ecf_sat = (r_ecf.T / np.linalg.norm(r_ecf,axis=1)).T * semi_major_axis # assume e == 0
	# alt = np.linalg.norm(r_ecf_sat-r_ecf,axis=1)
	alt_eq = alt[0] # alt at eq, assuming first pt is at equator
	angle = w/R_earth
	FOV = 2*np.arctan( (R_earth*np.tan(angle/2)) / alt_eq )

	# variable swath based on curvature of Earth (spherical)
	arg = np.sin(FOV/2) * (1 + alt/R_earth)
	eps = np.arccos(arg)

	tan2eps = np.tan(eps)**2
	c = R_earth/(R_earth+alt)
	a0 = 1 + tan2eps
	b0 = -2*c
	c0 = c**2 - tan2eps

	x1 = (-b0 + np.sqrt( b0**2 - 4*a0*c0 )) / (2*a0)
	x2 = (-b0 - np.sqrt( b0**2 - 4*a0*c0 )) / (2*a0)

	theta = np.arccos(x1)
	w_iter_var = 2*R_earth*theta

	return w_iter_var



def ray_cast_vec(r0_vec, u_hat_vec, a=6378.137, f=1/298.257223563, near_soln=True):
	# analytic ray-casting from s/c to Earth
	#	vectorized process cannot distinguish
	#	tangency from invalid off-nadir pointing
	#	i.e. disc[disc < eps]
	rtn_single = 0
	if not isinstance(r0_vec, np.ndarray):
		r0_vec = np.array(r0_vec)
	if not isinstance(u_hat_vec, np.ndarray):
		u_hat_vec = np.array(u_hat_vec)

	if len(r0_vec.shape) == 1 or len(u_hat_vec.shape) == 1:
		r0_vec = np.array([r0_vec])
		u_hat_vec = np.array([u_hat_vec])
		rtn_single = 1

	# a = 6378.137
	# f = 1/298.257223563
	b = a * (1-f) # matches with pyproj #6356.7523142
	x0, y0, z0 = r0_vec.T[0], r0_vec.T[1], r0_vec.T[2]
	a0 = (u_hat_vec.T[0]/a)**2 + (u_hat_vec.T[1]/a)**2 + (u_hat_vec.T[2]/b)**2
	b0 = 2*x0*u_hat_vec.T[0]/a**2 + 2*y0*u_hat_vec.T[1]/a**2 + 2*z0*u_hat_vec.T[2]/b**2
	c0 = (x0/a)**2 + (y0/a)**2 + (z0/b)**2 - 1

	disc = b0**2 - 4*a0*c0 # discriminant
	eps = 1e-16
	disc[disc < eps] = 0.0
	r_int = np.full(r0_vec.shape, np.nan)

	tau1 = (-b0 + np.sqrt(disc)) / (2*a0)
	tau2 = (-b0 - np.sqrt(disc)) / (2*a0)
	if near_soln:
		tau = np.min([tau1, tau2], axis=0) # near-side of earth soln
	else:
		tau = np.max([tau1, tau2], axis=0) # far-side soln

	# print(tau)
	tau[tau < 0] = np.nan

	r_intx = r0_vec.T[0] + u_hat_vec.T[0]*tau
	r_inty = r0_vec.T[1] + u_hat_vec.T[1]*tau
	r_intz = r0_vec.T[2] + u_hat_vec.T[2]*tau
	r_int = np.transpose([r_intx, r_inty, r_intz])

	# r_int = r0 + u_hat*tau
	r_int[disc < eps] = np.array([np.nan, np.nan, np.nan])

	if rtn_single:
		return r_int[0]
	else:
		return r_int


def calc_omega(t, R):
	# R has shape N x 3 x 3
	# https://physics.stackexchange.com/questions/293037/how-to-compute-the-angular-velocity-from-the-angles-of-a-rotation-matrix
	# https://en.wikipedia.org/wiki/Angular_velocity_tensor
	
	R_dot = (np.gradient(R,axis=0).T / np.gradient(t)).T
	R_tr = np.transpose(R, axes=(0,2,1))
	W = R_dot @ R_tr

	w1a = -W[:,1,2]
	w1b = W[:,2,1]
	w2a = W[:,0,2]
	w2b = -W[:,2,0]
	w3a = -W[:,0,1]
	w3b = W[:,1,0]
	# za = W[:,0,0]
	# zb = W[:,1,1]
	# zc = W[:,2,2]

	w1 = (w1a+w1b)/2
	w2 = (w2a+w2b)/2
	w3 = (w3a+w3b)/2
	omega_sc = np.transpose([w1,w2,w3])
	# alpha_sc = (np.gradient(omega_sc,axis=0).T / np.gradient(t)).T
	# return omega_sc, alpha_sc
	return omega_sc


def SC_frame(r_ecf_sc, v_ecf_sc, off_nadir_C_vec=None, off_nadir_A_vec=None, LVLH=False):

	N = len(r_ecf_sc)
	r_hat_vec = (r_ecf_sc.T / mag(r_ecf_sc)).T
	v_hat_vec = (v_ecf_sc.T / mag(v_ecf_sc)).T

	A, C = 0, 0

	if off_nadir_A_vec is not None:
		A = 1
	if off_nadir_C_vec is not None:
		C = 1

	# 	off_nadir_A_vec = np.zeros(N)
	# if off_nadir_C_vec is None:
	# 	off_nadir_C_vec = np.zeros(N)
	if not LVLH:
		k_hat_vec = r_hat_vec
		i_hat0_vec = v_hat_vec
		j_hat_vec = np.cross(k_hat_vec, i_hat0_vec)
		j_hat_vec = (j_hat_vec.T / mag(j_hat_vec)).T
		i_hat_vec = np.cross(j_hat_vec, k_hat_vec)

	else:
		k_hat_vec = -r_hat_vec
		j_hat_vec = -np.cross(r_hat_vec,v_hat_vec)
		j_hat_vec = (j_hat_vec.T / mag(j_hat_vec)).T
		i_hat_vec = np.cross(j_hat_vec,k_hat_vec)


	# R is in shape (N,3,3)
	# r is in shape (N,3)
	RT = np.transpose([i_hat_vec, j_hat_vec, k_hat_vec], axes=(1,2,0))

	if A:
		# vectorized ut.R2, active
		# R2 = lambda th: np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
		c = -1 # positive clockwise, "up toward space"
		if LVLH:
			c = 1 # positive cc, "up toward space"

		R_Ac = np.cos(c*off_nadir_A_vec)
		R_As = np.sin(c*off_nadir_A_vec)
		I, Z = np.ones(N), np.zeros(N)
		R_A = np.array([[R_Ac, Z, R_As], [Z, I, Z], [-R_As, Z, R_Ac]])
		R_A = np.transpose(R_A, axes=(2,0,1)) # Nx3x3

	if C:
		# vectorized ut.R1, active
		# R1 = lambda th: np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
		R_Cc = np.cos(off_nadir_C_vec) # left
		R_Cs = np.sin(off_nadir_C_vec)
		I, Z = np.ones(N), np.zeros(N)
		R_C = np.array([[I, Z, Z], [Z, R_Cc, -R_Cs], [Z, R_Cs, R_Cc]])
		R_C = np.transpose(R_C, axes=(2,0,1)) # Nx3x3

	# by convention, rotate first by alongtrack, then by crosstrack
	#	R_nadir = RT @ R_A @ R_C @ u_hat
	#	R_nadir_z is upward, x is "fwd" and y is "left"
	#	quotes b/c frame is rotated by A and C

	R_nadir = RT
	if A and C:
		R_nadir = RT @ R_A @ R_C
	elif A:
		R_nadir = RT @ R_A
	elif C:
		R_nadir = RT @ R_C

	return R_nadir



def project_pos(r_ecf_sc, v_ecf_sc, tr_lla_ecef, off_nadir_C_vec=None, off_nadir_A_vec=None):

	# ba = off_nadir_A_vec is None
	# bc = off_nadir_C_vec is None
	# if ba & bc:
	# 	lon_int1, lat_int1, alt_int1 = tr_lla_ecef.transform(r_ecf_sc.T[0], r_ecf_sc.T[1], r_ecf_sc.T[2], direction='inverse')
	# 	r_ecf = tr_lla_ecef.transform(lon_int1, lat_int1, np.zeros(lon_int1.shape))
	# 	r_ecf = np.transpose([r_ecf[0], r_ecf[1], r_ecf[2]])
	# 	return lon_int1, lat_int1, alt_int1, r_ecf

	FOV_rad = 0.0
	r0_vec, v0_vec = r_ecf_sc, v_ecf_sc
	N = len(r0_vec)
	if off_nadir_A_vec is None:
		off_nadir_A_vec = np.zeros(N)
	if off_nadir_C_vec is None:
		off_nadir_C_vec = np.zeros(N)

	r_hat_vec = (r0_vec.T / mag(r0_vec)).T
	v_hat_vec = (v0_vec.T / mag(v0_vec)).T

	k_hat_vec = r_hat_vec
	i_hat0_vec = v_hat_vec
	j_hat_vec = np.cross(k_hat_vec, i_hat0_vec)
	j_hat_vec = (j_hat_vec.T / mag(j_hat_vec)).T
	i_hat_vec = np.cross(j_hat_vec, k_hat_vec)

	# R is in shape (N,3,3)
	# r is in shape (N,3)
	RT = np.transpose([i_hat_vec, j_hat_vec, k_hat_vec], axes=(1,2,0))

	nadir_hat = np.array([0,0,-1])

	# vectorized ut.R2, active
	# R2 = lambda th: np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
	R_Ac = np.cos(-off_nadir_A_vec) # positive clockwise
	R_As = np.sin(-off_nadir_A_vec)
	I, Z = np.ones(N), np.zeros(N)
	R_A = np.array([[R_Ac, Z, R_As], [Z, I, Z], [-R_As, Z, R_Ac]])
	R_A = np.transpose(R_A, axes=(2,0,1)) # Nx3x3

	# vectorized ut.R1, active
	# R1 = lambda th: np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
	R_Cc = np.cos(FOV_rad/2 + off_nadir_C_vec) # left
	R_Cs = np.sin(FOV_rad/2 + off_nadir_C_vec)
	# I, Z = np.ones(N), np.zeros(N)
	R_C1 = np.array([[I, Z, Z], [Z, R_Cc, -R_Cs], [Z, R_Cs, R_Cc]])
	R_C1 = np.transpose(R_C1, axes=(2,0,1)) # Nx3x3

	# R_Cc = np.cos(-FOV_rad/2 + off_nadir_C_vec) # right
	# R_Cs = np.sin(-FOV_rad/2 + off_nadir_C_vec)
	# # I, Z = np.ones(N), np.zeros(N)
	# R_C2 = np.array([[I, Z, Z], [Z, R_Cc, -R_Cs], [Z, R_Cs, R_Cc]])
	# R_C2 = np.transpose(R_C2, axes=(2,0,1)) # Nx3x3

	nadir_hat1_vec = R_C1 @ nadir_hat
	nadir_hat1_vec = matmul(R_A, nadir_hat1_vec)
	u_hat1_vec = matmul(RT, nadir_hat1_vec)

	# nadir_hat2_vec = R_C2 @ nadir_hat
	# nadir_hat2_vec = matmul(R_A, nadir_hat2_vec)
	# u_hat2_vec = matmul(RT, nadir_hat2_vec)

	r_int1_vec = ray_cast_vec(r0_vec, u_hat1_vec)
	# r_int2_vec = ray_cast_vec(r0_vec, u_hat2_vec)

	# crs_grid = CRS.from_proj4(proj4_geo)
	# crs_ecf = CRS.from_proj4(proj4_ecf)
	# tr_lla_ecef = proj.Transformer.from_crs(crs_grid, crs_ecf)
	lon_int1, lat_int1, alt_int1 = tr_lla_ecef.transform(r_int1_vec.T[0], r_int1_vec.T[1], r_int1_vec.T[2], direction='inverse')
	# lon_int2, lat_int2, alt_int2 = tr_lla_ecef.transform(r_int2_vec.T[0], r_int2_vec.T[1], r_int2_vec.T[2], direction='inverse')

	# print('alt_int1', np.ptp(alt_int1))
	if np.abs(np.ptp(alt_int1)) > 1e-3:
		print('warning project_pos: alt_int1', np.abs(np.ptp(alt_int1)))
	# lon_int1_rad, lat_int1_rad = np.radians(lon_int1), np.radians(lat_int1)
	# lon_int2_rad, lat_int2_rad = np.radians(lon_int2), np.radians(lat_int2)
	# w_ev = ev_inverse(lon_int1_rad, lat_int1_rad, lon_int2_rad, lat_int2_rad, dist_only=True)
	# w_ev = w_ev/1e3

	# return w_ev
	return lon_int1, lat_int1, mag(r_ecf_sc-r_int1_vec), r_int1_vec

