

import numpy as np
from leocat.utils.const import *
from leocat.utils.math import R1, R2, R3
# from leocat.utils.math import newton_raphson
from leocat.utils.math import unit, mag, dot, matmul

from leocat.utils.geodesy import ecf_to_lla, lla_to_ecf
from leocat.utils.geodesy import ev_direct, ev_inverse
# from leocat.utils.plot import make_fig, draw_vector, set_axes_equal, set_aspect_equal


from numba import njit



def vector_to_t_access(t_total, index):
	from pandas import DataFrame
	df = DataFrame({'index': index})
	index_indices = df.groupby('index',sort=False).indices
	t_access = {}
	for key in index_indices:
		idx = index_indices[key]
		t_access[key] = t_total[idx]
	return t_access

def t_access_to_vector(t_access):
	t_total = []
	index = []
	for key in t_access:
		tau = t_access[key]
		t_total.append(tau)
		index.append(np.full(tau.shape, key))
	t_total = np.concatenate(t_total)
	index = np.concatenate(index)
	return t_total, index



def combine_coverage(lons, lats, t_access_list, DGG):
	"""
	Assuming all lons/lats are on the same DGG,
	combine into one t_access
		Re-indexes the final t_access to reflect
		lon/lat_total

	"""
	from leocat.utils.index import hash_cr_DGG, hash_xy_DGG

	t_access_cr = {}
	for i in range(len(lons)):
		lon, lat = lons[i], lats[i]
		t_access = t_access_list[i]
		cols, rows = hash_cr_DGG(lon, lat, DGG)
		for key in t_access:
			c, r = cols[key], rows[key]
			if not ((c,r) in t_access_cr):
				t_access_cr[(c,r)] = []
			t_access_cr[(c,r)].append(t_access[key])

	cr = np.array(list(t_access_cr.keys()))
	lon_total, lat_total = hash_xy_DGG(cr.T[0], cr.T[1], DGG)

	t_access_total = {}
	for j,(c,r) in enumerate(t_access_cr):
		t_access_total[j] = np.sort(np.concatenate(t_access_cr[(c,r)]))

	# for (c,r) in t_access_cr:
	# 	t_access_cr[(c,r)] = np.sort(np.concatenate(t_access_cr[(c,r)]))
	# cr = np.array(list(t_access_cr.keys()))
	# lon_total, lat_total = hash_xy_DGG(cr.T[0], cr.T[1], DGG)

	return lon_total, lat_total, t_access_total




def get_coverage(orb, swath, JD1, JD2, verbose=2, res=None, alpha=0.25): #, lon=None, lat=None):

	from leocat.utils.bt import AnalyticCoverage

	simulation_period = (JD2-JD1)*86400
	Tn = orb.get_period('nodal')
	num_revs = simulation_period / Tn
	if res is None:
		res = alpha*swath # dx = alpha*w as in EGPA

	C = 2*np.pi*R_earth
	area_cov = num_revs * swath * C
	A_earth = 4*np.pi*R_earth**2

	# lonlat_exists = False
	# if not (lon is None) and not (lat is None):
	# 	lonlat_exists = True

	if area_cov < A_earth:
		# If less duplicate area covered than A_earth,
		# find lon/lats directly
		from leocat.utils.swath import Instrument, SwathEnvelope
		if verbose > 1:
			print('preprocessing lon/lats..')
		FOV_CT = swath_to_FOV(swath, alt=orb.get_alt(), radians=False)
		Inst = Instrument(FOV_CT)
		SE = SwathEnvelope(orb, Inst, res, JD1, JD2)
		lon, lat = SE.get_lonlat()

	else:
		from leocat.utils.geodesy import DiscreteGlobalGrid
		# if more, use global grid
		if verbose > 1:
			print('using global lon/lats..')
		DGG = DiscreteGlobalGrid(A=res**2)
		lon, lat = DGG.get_lonlat()

	# spherical lat for AC
	phi = np.radians(lat)
	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
	lat_c = np.degrees(phi_c)

	AC = AnalyticCoverage(orb, swath, lon, lat_c, JD1, JD2)
	t_access = AC.get_access(verbose=verbose, accuracy=2)


	return lon, lat, t_access



@njit
def get_dt_avg_MC_numba(t, q, N_MC):
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





def simple_coverage(orb, swath, lon, lat, JD1, JD2, dt=None, f=0.25, \
					verbose=1, solar_band=None, warn=True, spherical=False):

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

	from leocat.utils.geodesy import lla_to_ecf, RADEC_to_cart
	from leocat.utils.orbit import convert_ECI_ECF
	from leocat.utils.cov import swath_to_FOV
	from leocat.utils.math import unit, dot, mag
	from tqdm import tqdm


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

	
	dJD_des = dt_des/86400
	N = int((JD2-JD1)/dJD_des)+1
	dJD = (JD2-JD1)/N
	JD = np.linspace(JD1,JD2,N)
	t = (JD-JD1)*86400
	dt = t[1]-t[0]

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

	access = {}
	for j in iterator:

		p_hat0 = p_hat[j]
		p0 = p[j]

		alpha_max = np.arccos(R_earth/a)
		proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
		alpha = np.arccos(proj_alpha)

		dr = p0 - r_ecf_sc
		dr_hat = unit(dr)
		proj = dot(dr_hat,-r_hat)
		angle = np.arccos(proj)

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

	return t, access


# def get_max_revisit(t, access, num_pts):
# 	dt = t[1]-t[0]
# 	revisit_max = np.full(num_pts,np.nan)
# 	for j in access:
# 		t_access = t[access[j]]
# 		dt_access = np.diff(t_access)
# 		revisits = dt_access[dt_access > 2*dt] / 3600 # hrs
# 		if len(revisits) > 0:
# 			revisit_max[j] = np.max(revisits)
# 	return revisit_max

def get_revisit(t_access_avg, num_pts, revisit_type='avg'):
	revisit = np.full(num_pts,np.nan)
	for key in t_access_avg:
		t_access = t_access_avg[key]
		if len(t_access) < 2:
			continue

		# revisit exists
		if revisit_type == 'avg':
			revisit[key] = np.mean(np.diff(t_access))
		elif revisit_type == 'max':
			revisit[key] = np.max(np.diff(t_access))
		elif revisit_type == 'min':
			revisit[key] = np.min(np.diff(t_access))
		elif revisit_type == 'count':
			revisit[key] = len(np.diff(t_access))

	return revisit


# def get_num_obs(t, access, num_pts):
def get_num_obs(t_access_avg, num_pts):
	num_obs = np.zeros(num_pts)
	for key in t_access_avg:
		num_obs[key] = num_obs[key] + len(t_access_avg[key])

	# num_obs = np.zeros(num_pts)
	# for key in access:
	# 	# key = access[k]
	# 	idx = access[key]
	# 	num_obs[key] = num_obs[key] + 1
	# 	idx_change = np.where(np.diff(idx) > 1)[0].astype(int)
	# 	if len(idx_change) > 0:
	# 		num_obs[key] = num_obs[key] + len(idx_change)

	return num_obs



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


# def get_t_access_avg(t, access):
# 	t_access_avg = {}
# 	for key in access:
# 		idx = access[key]
# 		idx_change = np.where(np.diff(idx) > 1)[0].astype(int)

# 		if len(idx_change) == 0:
# 			# only one period of access
# 			t_access = t[access[key]]
# 			t_access_avg[key] = np.array([np.mean(t_access)])

# 		else:
# 			# multiple periods of access
# 			# idx_change corresponds to last element in each access period
# 			t_access = t[access[key]]
# 			t_access_avg[key] = []
# 			k1 = 0
# 			for j in range(len(idx_change)):
# 				k2 = idx_change[j]
# 				mu = np.mean(t_access[k1:k2+1])
# 				t_access_avg[key].append(mu)
# 				k1 = k2
# 			mu_last = np.mean(t_access[k2:])
# 			t_access_avg[key].append(mu_last)
# 			t_access_avg[key] = np.array(t_access_avg[key])

# 	return t_access_avg


# def FOV_from_w_new(w, alt, radians=True):
def swath_to_FOV(w, alt, radians=True):

	# stupid approach
	# here's a better one
	# Law of Sines, spherical Earth
	# swath_to_FOV
	# theta = (w/2)/R_earth
	# alpha = np.arctan( R_earth*np.sin(theta) / (R_earth + h - R_earth*np.cos(theta)) )
	# FOV2 = np.degrees(alpha*2)

	# Governing eqn
	#	sin(alpha)/R = sin(alpha+theta)/(R+h)
	#	alpha = FOV/2
	#	w = R*theta*2

	r0 = np.array([R_earth + alt, 0, 0])

	u_hat0 = -unit(r0)
	r_ground0 = ray_cast_vec(np.array([r0]), np.array([u_hat0]))
	r_ground0 = r_ground0[0]

	lon0, lat0, _ = ecf_to_lla(r_ground0[0], r_ground0[1], r_ground0[2])
	lon1, lat1, _ = ev_direct(lon0, lat0, 90.0, w/2, radians=False, unit='km')
	lon2, lat2, _ = ev_direct(lon0, lat0, 270.0, w/2, radians=False, unit='km')

	r_ground1 = lla_to_ecf(lon1, lat1, 0)
	r_ground2 = lla_to_ecf(lon2, lat2, 0)

	u_hat1 = unit(r0 - r_ground1)
	u_hat2 = unit(r0 - r_ground2)

	proj = np.dot(u_hat1,u_hat2)
	FOV = np.arccos(proj)*180/np.pi

	if radians:
		FOV = np.radians(FOV)

	return FOV
	


# def w_from_FOV_new(FOV, alt, radians=True, debug=0):
def FOV_to_swath(FOV, alt, radians=True, debug=0):

	# [208]
	# h = alt
	# psi = np.radians(FOR/2)
	# gamma = np.pi - np.arcsin((R_earth + h)*np.sin(psi) / R_earth)
	# S = R_earth * np.cos(gamma) + (R_earth + h)*np.cos(psi)
	# s = 2*S*np.sin(psi)

	# stupid approaches
	# here's a better one
	# Law of Sines, spherical Earth
	# FOV_to_swath
	# alpha = np.radians(FOV)/2
	# theta = np.arcsin((R_earth+h)/R_earth * np.sin(alpha)) - alpha
	# w = R_earth * theta * 2

	if not radians:
		FOV = np.radians(FOV)

	# alt = 700
	r0 = np.array([R_earth + alt, 0, 0])

	# FOV = np.radians(45.0)
	u_hat1 = R3(-FOV/2) @ -unit(r0)
	u_hat2 = R3(FOV/2) @ -unit(r0)

	r_ground = ray_cast_vec(np.array([r0,r0]), np.array([u_hat1,u_hat2]))
	lon1, lat1, _ = ecf_to_lla(r_ground[0][0], r_ground[0][1], r_ground[0][2])
	lon2, lat2, _ = ecf_to_lla(r_ground[1][0], r_ground[1][1], r_ground[1][2])

	eps = 1e-4
	dist = ev_inverse(lon1, lat1+eps, lon2, lat2, radians=False, dist_only=True, unit='km')

	if debug:
		from leocat.utils.plot import make_fig, draw_vector, set_axes_equal, set_aspect_equal
		# make 2D
		r0 = r0[:-1]
		r_ground = r_ground[:,:-1]

		L_plot = alt*4
		origin = np.mean(r_ground,axis=0)

		theta = np.linspace(0,2*np.pi,1000)
		x = R_earth*np.cos(theta)
		y = R_earth_pole*np.sin(theta)

		fig, ax = make_fig()
		ax.plot(x, y)
		ax.plot(r0[0], r0[1], '.')
		draw_vector(ax, r0, r_ground[0], 'k')
		draw_vector(ax, r0, r_ground[1], 'k')
		set_axes_equal(ax)
		set_aspect_equal(ax)
		ax.set_xlim(origin[0]-L_plot/2, origin[0]+L_plot/2)
		ax.set_ylim(origin[1]-L_plot/2, origin[1]+L_plot/2)
		fig.show()

	return dist



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

