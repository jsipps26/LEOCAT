
import numpy as np
from leocat.utils.const import *
from leocat.utils.general import pause
from leocat.utils.math import unit, mag, multiply, arcsin, wrap, rad
from leocat.utils.index import hash_cr_DGG, hash_xy_DGG, unique_2d
from leocat.utils.orbit import convert_ECI_ECF, get_R_ECI_ECF_GMST

from pyproj import CRS, Transformer
from leocat.utils.geodesy import DiscreteGlobalGrid

from copy import deepcopy

def get_u_hat(dtheta, u_hat0, y_hat, z_hat, alpha_max):
	proj = np.dot(u_hat0,y_hat)
	u_hat = np.sign(proj)*y_hat*np.cos(alpha_max + dtheta) - z_hat*np.sin(alpha_max + dtheta)
	return u_hat

def func(r0, u_hat, a, b):
	# u_hat = get_u_hat(dtheta, u_hat0, y_hat, z_hat, alpha_max)
	x0, y0, z0 = r0
	a0 = (u_hat[0]/a)**2 + (u_hat[1]/a)**2 + (u_hat[2]/b)**2
	b0 = 2*x0*u_hat[0]/a**2 + 2*y0*u_hat[1]/a**2 + 2*z0*u_hat[2]/b**2
	c0 = (x0/a)**2 + (y0/a)**2 + (z0/b)**2 - 1
	disc = b0**2 - 4*a0*c0
	return disc

def solve_horizon(r0, u_hat0, eps=1e-16, dtheta0=np.radians(1.0)):
	"""
	u_hat0 is assumed to be near the horizon.
	An approximate horizon angle for u_hat is based on alpha_max,
	using R_earth and satellite radius from the center of Earth.
	This serves as the initial guess for u_hat. Let F(dtheta) be
	the discriminant of the quadratic solution to the intersection
	of a line with the ellipsoid (see ray_cast). F(dtheta) = 0 when
	the intersection is tangent to the ellipsoid. Since the solution
	is numerical, we must ensure that F(dtheta) > 0 since the 
	discriminant must be positive for a real solution. So we change
	iterative change dtheta s.t. F(dtheta) is just barely above zero.

	"""
	z_hat = unit(r0)
	x_hat = unit(np.cross(z_hat,u_hat0))
	y_hat = np.cross(z_hat,x_hat)

	alpha_max = np.arccos(R_earth/mag(r0))
	f = 1/298.257223563
	a = R_earth
	b = a # a * (1-f) # matches with pyproj #6356.7523142
	# F = lambda dtheta: func(dtheta, r0, u_hat0, y_hat, z_hat, alpha_max, a, b)
	def F(dtheta):
		u_hat = get_u_hat(dtheta, u_hat0, y_hat, z_hat, alpha_max)
		return func(r0, u_hat, a, b)

	x1, x2 = -dtheta0, dtheta0
	F1 = F(x1)
	F2 = F(x2)

	m = (F2-F1)/(x2-x1)
	y0 = -m*x1 + F1

	dtheta = -y0/m
	F0 = F(dtheta)
	max_iter = 10
	j = 0
	err = 0
	while F0 < eps:
		dF = 0-F0
		ddtheta = dF/m
		dtheta = dtheta + ddtheta
		F0 = F(dtheta)
		j += 1
		if j == max_iter:
			import warnings
			warnings.warn('ray cast horizon did not converge')
			err = 1
			break

	# u_hat = y_hat*np.cos(alpha_max + dtheta) - z_hat*np.sin(alpha_max + dtheta)
	u_hat = get_u_hat(dtheta, u_hat0, y_hat, z_hat, alpha_max)

	return u_hat


def get_tau(r0, u_hat):
	a = 1.0
	b = 2*np.dot(r0,u_hat)
	c = mag(r0)**2 - R_earth**2

	tau = np.nan
	disc = b**2 - 4*a*c
	if disc >= 0.0:
		tau1 = (-b + np.sqrt(disc))/(2*a)
		tau2 = (-b - np.sqrt(disc))/(2*a)
		tau = np.min([tau1,tau2])
		if tau < 0.0:
			tau = np.nan

	return tau


def ray_cast(r0, u_hat):
	tau = get_tau(r0, u_hat)
	r = r0 + tau*u_hat
	return r

def get_theta_mesh(REG_B, r0, dist, F, H):
	theta_mesh = []
	for i,reg in enumerate(REG_B):

		theta0_init = np.mean(reg)

		theta_mesh_fwd = []
		theta_mesh_rev = []

		# fwd
		theta0 = theta0_init
		j = 0
		while True:
			u_hat1 = F(theta0)
			u_hat_dot1 = H(theta0)
			tau1 = get_tau(r0, u_hat1)
			if np.isnan(tau1):
				break

			tau_dot1 = -np.dot(u_hat_dot1, r0) * tau1 / (tau1 + np.dot(u_hat1,r0))
			theta_dot1 = dist / mag(u_hat1*tau_dot1 + u_hat_dot1*tau1)
			# if theta0 > theta_bounds[1]:
			if not (reg[0] <= theta0 <= reg[1]):
				break

			theta_mesh_fwd.append(theta0)
			theta0 = theta0 + theta_dot1
			j += 1

		#

		theta_mesh_rev = []
		theta0 = theta0_init
		j = 0
		while True:
			u_hat1 = F(theta0)
			u_hat_dot1 = H(theta0)
			tau1 = get_tau(r0, u_hat1)
			if np.isnan(tau1):
				break

			tau_dot1 = -np.dot(u_hat_dot1, r0) * tau1 / (tau1 + np.dot(u_hat1,r0))
			theta_dot1 = dist / mag(u_hat1*tau_dot1 + u_hat_dot1*tau1)
			# if theta0 < theta_bounds[0]:
			if not (reg[0] <= theta0 <= reg[1]):
				break
			if j > 0:
				theta_mesh_rev.append(theta0)

			theta0 = theta0 - theta_dot1
			j += 1


		theta_mesh_fwd = np.flip(theta_mesh_fwd)
		# theta_mesh_rev = np.flip(theta_mesh_rev)

		theta_mesh.append(theta_mesh_fwd)
		theta_mesh.append(theta_mesh_rev)

	# theta_mesh.append(REG_B.T[0])

	theta_mesh = np.concatenate(theta_mesh)
	# if (theta_mesh < 0).any():
	# 	print('warning: theta_mesh < 0')
	# if (theta_mesh > 2*np.pi).any():
	# 	print('warning: theta_mesh > 2pi')


	return theta_mesh


def theta_mesh_to_r_int(theta_mesh, r0, F):
	u_hat = []
	r_int_intp = []
	for j in range(len(theta_mesh)):
		u_hat1 = F(theta_mesh[j])
		r_int1 = ray_cast(r0, u_hat1)
		r_int_intp.append(r_int1)
		u_hat.append(u_hat1)
	r_int_intp = np.array(r_int_intp)
	u_hat = np.array(u_hat)

	return r_int_intp, u_hat


def theta_h_to_r_edge(r_int_h1, r_int_h2, r0, dist, direction):

	rc0 = unit(r0) * np.dot(r_int_h1,unit(r0))
	radius = mag(r_int_h1-rc0)
	rc1 = r_int_h1 - rc0
	rc2 = r_int_h2 - rc0

	Omega = np.arccos(np.dot(unit(rc1),unit(rc2)))
	if direction == 'long':
		Omega = 2*np.pi-Omega
	arc_length = radius*Omega
	N_intp_t = int(np.ceil(arc_length / dist))
	if N_intp_t < 3:
		N_intp_t = 3

	tau = np.linspace(0,1,N_intp_t)
	r_edge = multiply( rc1, np.sin((1-tau)*Omega)/np.sin(Omega) ) + \
				multiply( rc2, np.sin(tau*Omega)/np.sin(Omega) ) + rc0
	#
	return r_edge


def get_theta_reg(theta_space, theta_lim):
	theta_reg = []
	# for theta0 in theta_space:
	for i in range(len(theta_space)):
		theta0 = theta_space[i]
		if np.isnan(theta0):
			continue
		theta_reg.append(theta0)

	if len(theta_reg) > 0:
		theta_reg.insert(0,theta_lim[0])
		theta_reg.append(theta_lim[1])

	theta_reg = np.array(theta_reg)
	return theta_reg


def get_REG_B(r0, theta_h, F, REG_B0):
	REG_B = []
	if len(theta_h) >= 2:
		for j in range(1,len(theta_h)):
			reg = theta_h[j-1], theta_h[j]
			theta0 = np.mean(reg)
			tau = get_tau(r0, F(theta0))
			if tau >= 0.0:
				REG_B.append(reg)

		REG_B = np.array(REG_B)

	else:
		# horizon never crossed
		# either looking into space or directly at Earth
		#	REG_B = [] or REG_B = [0,2pi]
		# test single theta
		theta0 = 0.0
		tau = get_tau(r0, F(theta0))
		if tau >= 0.0:
			REG_B = [REG_B0]
		REG_B = np.array(REG_B)

	return REG_B


################################################

def mesh_to_gc(r_fpt, space_params):
	tr_lla_ecf = space_params['tr_lla_ecf']
	DGG = space_params['DGG']
	lon_int, lat_int, _ = tr_lla_ecf.transform(r_fpt.T[0], r_fpt.T[1], r_fpt.T[2], direction='inverse')
	c_int, r_int = hash_cr_DGG(lon_int, lat_int, DGG)
	cr_int = unique_2d(c_int, r_int)
	c_int, r_int = cr_int.T
	return c_int, r_int

def gc_to_rg(c_int, r_int, space_params):
	tr_lla_ecf = space_params['tr_lla_ecf']
	DGG = space_params['DGG']
	x_int, y_int = hash_xy_DGG(c_int, r_int, DGG)
	rg_int = tr_lla_ecf.transform(x_int, y_int, np.zeros(x_int.shape))
	rg_int = np.transpose([rg_int[0],rg_int[1],rg_int[2]])
	return rg_int

def gc_to_col_bounds(c_int, r_int):
	from pandas import DataFrame
	from leocat.utils.cov import get_access_interval

	df = DataFrame({'r': r_int})
	index = df.groupby('r').indices
	keys = list(index.keys())
	c_index = {}
	for key in index:
		c_index[key] = c_int[index[key]]
	c_bounds = get_access_interval(c_index)
	return keys, c_bounds

def classify_bridges(keys, c_bounds, Sensor, space_params):

	DGG_params = space_params['DGG_params']
	num_cols_lat = DGG_params['num_cols_lat']
	DGG = space_params['DGG']

	bridge_class = {}
	row_complete = dict(zip(keys,np.ones(len(keys)).astype(bool)))
	for key in c_bounds:
		cb_r = c_bounds[key]
		cb_r = np.vstack((cb_r, num_cols_lat[key] + cb_r[0]))
		c1 = cb_r.T[1][:-1]
		c2 = cb_r.T[0][1:]
		dc = c2-c1
		j_range = np.arange(len(dc))

		c_mid = np.round((c1+c2)/2).astype(int)
		r_mid = np.full(c_mid.shape, key)
		lon_mid, lat_mid = hash_xy_DGG(c_mid, r_mid, DGG)
		lon_mid = wrap(lon_mid)

		a, b = R_earth, R_earth_pole
		lam = rad(lon_mid)
		phi = rad(lat_mid) # geodetic
		phi_c = np.arctan((b/a)**2 * np.tan(phi)) # geocentric
		rg_mid = np.transpose([ a*np.cos(lam)*np.cos(phi_c),
									a*np.sin(lam)*np.cos(phi_c),
									b*np.sin(phi_c) ])
		#
		# tr_lla_ecf = space_params['tr_lla_ecf']
		# rg_mid2 = tr_lla_ecf.transform(lon_mid, lat_mid, np.zeros(lon_mid.shape))
		# rg_mid2 = np.transpose(rg_mid2)
		# rg_mid_track[key] = rg_mid
		# print(rg_mid-rg_mid2)

		b = Sensor.point_in_view(rg_mid)
		j_range = j_range[b]

		for j in j_range:
			c01, c02 = c1[j]+1, c2[j]
			num = c02 - c01
			# print(j, num)
			if num == 0:
				continue
			if not (key in bridge_class):
				bridge_class[key] = []
			bridge_class[key].append([c01,c02])
			row_complete[key] = False

	return bridge_class, row_complete


def classify_pole_N(bridge_class, row_complete, space_params):
	DGG_params = space_params['DGG_params']
	if len(bridge_class) > 0:
		row_max_bridge = np.max(list(bridge_class.keys())) + 1
		row_max = DGG_params['r_max']
		num = row_max - row_max_bridge + 1
		if num > 0:
			c_min_lat = DGG_params['c_min_lat']
			c_max_lat = DGG_params['c_max_lat']
			for key in range(row_max_bridge, row_max+1):
				c01, c02 = c_min_lat[key], c_max_lat[key]
				if key in row_complete:
					if row_complete[key]:
						continue
				bridge_class[key] = [[c01,c02+1]]

	else:
		"""
		Known error:
		If bridge_class is empty, likely because there just
		aren't any bridges, i.e. only have coverage on cr_boundary.
		That's fine, and warrants skipping this function.. 
		but it could also be the case that the s/c is directly
		above the pole with a conical FOV, and there's only 1 bridge
		that wraps around the pole exactly. Then the ellipsoid 
		projects the bridge just outside of the FOV, and the bridge
		is eliminated. The boundary is eliminated as well because
		it's also projected.. then it leaves no coverage for this 
		footprint.

		One fix is to try everything spherical.. but even then,
		you can find GPs that are just outside the FOV from the
		mesh. 

		Basically the GC boundary is undefined... it exists
		right outside the FOV. This is just a fundamental 
		limitation of boundary theory applied to GCs.. 
		Potentially can flag this particular case:
			1 row of GCs not at the pole itself
			bridge_class is empty

		Then just force-add it regardless of whether it's technically
		outside the footprint.

		"""
		pass


	return bridge_class


def classify_pole_S(bridge_class, row_complete, space_params):
	DGG_params = space_params['DGG_params']
	if len(bridge_class) > 0:
		row_min_bridge = np.min(list(bridge_class.keys())) - 1
		row_min = DGG_params['r_min']
		num = row_min_bridge - row_min + 1
		if num > 0:
			c_min_lat = DGG_params['c_min_lat']
			c_max_lat = DGG_params['c_max_lat']
			for key in range(row_min, row_min_bridge+1):
				c01, c02 = c_min_lat[key], c_max_lat[key]
				if key in row_complete:
					if row_complete[key]:
						continue
				bridge_class[key] = [[c01,c02+1]]

	else:
		# Known error: see classify_pole_N
		pass

	return bridge_class


def bridge_to_cols(bridge_class):
	cols = {}
	for key in bridge_class:
		for j in range(len(bridge_class[key])):
			if not (key in cols):
				cols[key] = []
			c01, c02 = bridge_class[key][j]
			cols[key].append(np.arange(c01,c02))
	return cols


def bridges_to_cr(bridge_class, cr_boundary):	
	if len(bridge_class) > 0:
		cols, rows = [], []
		for key in bridge_class:
			for j in range(len(bridge_class[key])):
				c01, c02 = bridge_class[key][j]
				cols0 = np.arange(c01,c02)
				cols.append(cols0)
				rows.append(np.full(cols0.shape,key))
		cr = np.transpose([np.concatenate(cols), np.concatenate(rows)])
		cr = np.vstack((cr, cr_boundary))
	else:
		cr = cr_boundary
	return cr



class Satellite:
	def __init__(self, orb=None, res=None, JD1=None, Inst=None):
		if orb is None:
			# near SSO but no particular MLST
			from leocat.orb import LEO
			orb = LEO(alt=705, inc=98.2)
		self.orb = orb

		if res is None:
			res = 100.0
		self.res = res

		if JD1 is None:
			JD1 = date_to_jd(2024,1,1)
		self.JD1 = JD1

		if Inst is None:
			Inst = Instrument(60.0)
			Inst.parent = self
		self.Inst = Inst

		self.space_params = {'res': res}
		self.set_space_params()

	def set_space_params(self):
		crs_lla = CRS.from_proj4(PROJ4_LLA).to_3d()
		crs_ecf = CRS.from_proj4(PROJ4_ECF).to_3d()
		tr_lla_ecf = Transformer.from_crs(crs_lla, crs_ecf)
		res = self.space_params['res']
		A = res**2
		DGG = DiscreteGlobalGrid(A=A)
		self.space_params['A'] = A
		self.space_params['DGG'] = DGG
		self.space_params['tr_lla_ecf'] = tr_lla_ecf
		self.space_params['DGG_params'] = DGG.get_params()

	def set_instrument(self, Inst):
		Inst = deepcopy(Inst)
		self.Inst = Inst

	def get_rv(self, t, eci=False):
		r, v = self.orb.propagate(t)
		if not eci:
			JD1 = self.JD1
			JD = JD1 + t/86400.0
			r, v = convert_ECI_ECF(JD, r, v)
		return r, v

	def get_att(self, t, roll=0.0, pitch=0.0, yaw=0.0, eci=False):
		# default is yaw, pitch, then roll
		# 	R3-R2-R1
		r_eci, v_eci = self.get_rv(t,eci=True)
		R_LVLH_ECI = self.get_R_LVLH(r_eci, v_eci)
		attitude_any = (roll != 0.0) or (pitch != 0.0) or (yaw != 0.0)
		if attitude_any:
			from leocat.utils.math import R1, R2, R3
			dR = R3(rad(yaw)) @ R2(rad(pitch)) @ R1(rad(roll))
			R_LVLH_ECI = R_LVLH_ECI @ dR
		if not eci:
			JD1 = self.JD1
			JD = JD1 + t/86400.0
			R_ECI_ECF = get_R_ECI_ECF_GMST(JD)
			R_LVLH = R_ECI_ECF @ R_LVLH_ECI # in ECF coords
		else:
			R_LVLH = R_LVLH_ECI
		return R_LVLH

	def get_access(self, r0, R):
		space_params = self.space_params
		return self.Inst.get_access(r0, R, space_params)

	def get_R_LVLH(self, r_eci, v_eci):
		r_hat = unit(r_eci)
		v_hat = unit(v_eci)
		h_hat = np.cross(r_hat,v_hat)
		k_hat = -r_hat
		j_hat = -h_hat
		i_hat = unit(np.cross(j_hat,k_hat))
		try:
			R_LVLH_ECI = np.transpose([i_hat,j_hat,k_hat], axes=(1,2,0)) # in ECI coords
		except ValueError:
			R_LVLH_ECI = np.array([i_hat,j_hat,k_hat]).T
		# R_ECI_ECF = get_R_ECI_ECF_GMST(JD[0])
		# R_LVLH0 = R_ECI_ECF @ R_LVLH_ECI # in ECF coords

		return R_LVLH_ECI

	# def point_in_view(self, lon, lat):
	# 	return self.Inst.point_in_view(lon, lat)


class Instrument:
	def __init__(self, FOV_CT, FOV_AT=None):
		is_rectangular = True
		if FOV_AT is None:
			is_rectangular = False
		self.is_rectangular = is_rectangular

		self.FOV_CT = FOV_CT
		self.FOV_AT = FOV_AT

		if is_rectangular:
			FieldOfView = RectangularFOV(FOV_CT, FOV_AT)
		else:
			FieldOfView = CircularFOV(FOV_CT)
		self.FieldOfView = FieldOfView

	def get_access(self, r0, R, space_params):
		return self.FieldOfView.get_access(r0, R, space_params)

	# def point_in_view(self, lon, lat):
	# 	return self.FieldOfView.point_in_view(lon, lat)





class CircularFOV:
	def __init__(self, FOV):
		self.FOV = FOV
		self.w_hat = self.get_w_hat()

	def classify_boundary(self, r0, R, space_params):

		res = space_params['res']
		tr_lla_ecf = space_params['tr_lla_ecf']

		r_fpt = self.get_footprint(r0, R, res)
		coverage_exists = len(r_fpt) > 0

		bridge_class = {}
		cr_boundary = np.array([])
		if coverage_exists:
			c_int, r_int = mesh_to_gc(r_fpt, space_params)
			rg_int = gc_to_rg(c_int, r_int, space_params)
			keys, c_bounds = gc_to_col_bounds(c_int, r_int)
			lon_int, lat_int, _ = \
				tr_lla_ecf.transform(rg_int.T[0], rg_int.T[1], rg_int.T[2], direction='inverse')
			#
			cr_int = np.transpose([c_int,r_int])
			boundary = self.point_in_view(rg_int)
			cr_boundary = cr_int[boundary]

			bridge_class, row_complete = classify_bridges(keys, c_bounds, self, space_params)
			pole_N = np.array([0,0,R_earth_pole])
			pole_S = np.array([0,0,-R_earth_pole])
			pole_in_view_N = self.point_in_view(pole_N)
			pole_in_view_S = self.point_in_view(pole_S)
			if pole_in_view_N:
				bridge_class = classify_pole_N(bridge_class, row_complete, space_params)
			if pole_in_view_S:
				bridge_class = classify_pole_S(bridge_class, row_complete, space_params)

		return bridge_class, cr_boundary

	def get_access(self, r0, R, space_params): #, return_GPs=False):
		bridge_class, cr_boundary = self.classify_boundary(r0, R, space_params)
		cr = bridges_to_cr(bridge_class, cr_boundary)
		return cr


	def get_footprint(self, r0, R, res, res_factor=0.25):

		self.r0 = r0
		self.R = R
		self.res = res
		self.res_factor = res_factor

		FOV = self.FOV
		w_hat = self.w_hat
		dist = res*res_factor

		u_hat0 = (R @ w_hat.T).T[0]
		x_hat = R.T[0]
		y_hat = R.T[1]
		z_hat = R.T[2]
		self.x_hat = x_hat
		self.y_hat = y_hat
		self.z_hat = z_hat

		CS0 = CircularSegment(r0, R, u_hat0, res, res_factor=res_factor)
		over_horizon = CS0.over_horizon
		r_mesh = CS0.r_mesh
		ce_mesh = CS0.ce_mesh
		self.segments = [CS0]
		self.ce_mesh = ce_mesh
		self.r_mesh = r_mesh

		r_edge = None
		ce_edge = 0
		if over_horizon:
			theta_h1, theta_h2 = CS0.theta_h_vec
			# u_hat_h1 = CS0.F(theta_h1)
			# u_hat_h1_fix = solve_horizon(r0, u_hat_h1)
			# u_hat_h2 = CS0.F(theta_h2)
			# u_hat_h2_fix = solve_horizon(r0, u_hat_h2)
			# r_int_h1 = ray_cast(r0, u_hat_h1_fix)
			# r_int_h2 = ray_cast(r0, u_hat_h2_fix)

			tau_h = np.sqrt(mag(r0)**2 - R_earth**2)
			u_hat_h1 = CS0.F(theta_h1)
			u_hat_h2 = CS0.F(theta_h2)
			r_int_h1 = r0 + u_hat_h1*tau_h
			r_int_h2 = r0 + u_hat_h2*tau_h

			rc0 = unit(r0) * np.dot(r_int_h1,unit(r0))
			radius = mag(r_int_h1-rc0)
			rc1 = r_int_h1 - rc0
			rc2 = r_int_h2 - rc0

			# assume short path
			#	if not in view, must be long path
			Omega = np.arccos(np.dot(unit(rc1),unit(rc2)))
			tau = 0.5
			r_edge_test = rc1*np.sin((1-tau)*Omega)/np.sin(Omega) + \
						rc2*np.sin(tau*Omega)/np.sin(Omega) + rc0
			#
			direction = 'short'
			in_view = self.point_in_view(r_edge_test, check_horizon=False)
			if not in_view:
				direction = 'long'

			r_edge = theta_h_to_r_edge(r_int_h1, r_int_h2, r0, dist, direction)
			ce_edge = 1


		coverage_exists = ce_mesh or ce_edge

		self.ce_edge = ce_edge
		self.r_edge = r_edge
		self.coverage_exists = coverage_exists

		r_fpt = np.array([])
		if coverage_exists:
			if ce_mesh and ce_edge:
				r_fpt = np.vstack((r_mesh, r_edge))
			elif ce_mesh:
				r_fpt = r_mesh
			elif ce_edge:
				r_fpt = r_edge

		self.r_fpt = r_fpt

		return r_fpt


	# def get_access(self):
	# 	return get_access()



	def get_w_hat(self):
		FOV = self.FOV
		EL0 = -0.5*(FOV-90)+45
		AZ = np.radians(np.linspace(0,360,1+1)[:-1])
		EL = np.radians(np.full(AZ.shape, EL0))
		w_hat = np.transpose([ np.cos(AZ)*np.cos(EL), np.sin(AZ)*np.cos(EL), np.sin(EL) ])
		return w_hat

	def point_in_view(self, r_pt, check_horizon=True):
		FOV = self.FOV
		r0 = self.r0
		z_hat = self.z_hat

		if check_horizon:
			alpha_max = np.arccos(R_earth/mag(r0))
			alpha = np.arccos(np.dot(unit(r_pt),unit(r0)))
			b_in_view = alpha < alpha_max

		p_hat = unit(r_pt-r0)
		proj = np.dot(p_hat, z_hat)
		proj = np.clip(proj,-1,1)
		angle = np.arccos(proj) * 180/np.pi
		b = (angle <= FOV/2)
		if check_horizon:
			b = b & b_in_view
			return b
		else:
			return b


class RectangularFOV:
	def __init__(self, FOV_CT, FOV_AT):
		self.FOV_AT = FOV_AT
		self.FOV_CT = FOV_CT
		self.w_hat = self.get_w_hat()

	def classify_boundary(self, r0, R, space_params):

		res = space_params['res']
		tr_lla_ecf = space_params['tr_lla_ecf']

		r_fpt = self.get_footprint(r0, R, res)
		coverage_exists = len(r_fpt) > 0

		bridge_class = {}
		cr_boundary = np.array([])
		if coverage_exists:
			c_int, r_int = mesh_to_gc(r_fpt, space_params)
			rg_int = gc_to_rg(c_int, r_int, space_params)
			keys, c_bounds = gc_to_col_bounds(c_int, r_int)
			lon_int, lat_int, _ = \
				tr_lla_ecf.transform(rg_int.T[0], rg_int.T[1], rg_int.T[2], direction='inverse')
			#
			cr_int = np.transpose([c_int,r_int])
			boundary = self.point_in_view(rg_int)
			cr_boundary = cr_int[boundary]

			bridge_class, row_complete = classify_bridges(keys, c_bounds, self, space_params)
			pole_N = np.array([0,0,R_earth_pole])
			pole_S = np.array([0,0,-R_earth_pole])
			pole_in_view_N = self.point_in_view(pole_N)
			pole_in_view_S = self.point_in_view(pole_S)
			if pole_in_view_N:
				bridge_class = classify_pole_N(bridge_class, row_complete, space_params)
			if pole_in_view_S:
				bridge_class = classify_pole_S(bridge_class, row_complete, space_params)
		return bridge_class, cr_boundary


	def get_access(self, r0, R, space_params): #, return_GPs=False):
		bridge_class, cr_boundary = self.classify_boundary(r0, R, space_params)
		cr = bridges_to_cr(bridge_class, cr_boundary)
		return cr


	def get_footprint(self, r0, R, res, res_factor=0.25):

		self.r0 = r0
		self.R = R
		self.res = res
		self.res_factor = res_factor

		FOV_AT = self.FOV_AT
		FOV_CT = self.FOV_CT
		w_hat = self.w_hat
		dist = res*res_factor

		x_hat = R.T[0]
		y_hat = R.T[1]
		z_hat = R.T[2]
		self.x_hat = x_hat
		self.y_hat = y_hat
		self.z_hat = z_hat

		# w_hat = self.get_w_hat()

		u_hat = (R @ w_hat.T).T
		self.u_hat = u_hat
		segments = []
		for j in range(1,len(u_hat)):
			CS0 = LinearSegment(r0, R, u_hat[j-1], u_hat[j], res, res_factor=res_factor)
			segments.append(CS0)

		r_int_h = []
		r_mesh = []
		ce_mesh = 0
		for CS0 in segments:
			if CS0.ce_mesh:
				ce_mesh = 1
				r_mesh.append(CS0.r_mesh)
			if not CS0.over_horizon:
				continue
			for theta_h0 in CS0.theta_h_vec:
				if np.isnan(theta_h0):
					continue
				u_hat_h0 = CS0.F(theta_h0)
				r_int_h0 = r0 + u_hat_h0*np.sqrt(mag(r0)**2 - R_earth**2)
				r_int_h.append(r_int_h0)

		if ce_mesh:
			r_mesh = np.vstack(r_mesh)

		r_int_h = np.array(r_int_h)
		if len(r_int_h) > 0:
			up = unit(r0)
			dr = r_int_h-r0
			dr_proj = []
			for j in range(len(dr)):
				proj_z = np.dot(dr[j],up)
				dr_proj.append(dr[j] - up*np.dot(dr[j],up))
			dr_proj = np.array(dr_proj)
			fwd = unit(dr[0] - up*np.dot(dr[0],up))
			left = unit(np.cross(up,fwd))

			proj_x = np.dot(unit(dr_proj),fwd)
			proj_x[proj_x > 1] = 1.0
			proj_x[proj_x < -1] = -1.0
			proj_y = np.dot(unit(dr_proj),left)
			proj_y[proj_y > 1] = 1.0
			proj_y[proj_y < -1] = -1.0
			angle = np.arccos(proj_x)
			angle[proj_y < 0] = 2*np.pi - angle[proj_y < 0]
			j_idx = np.argsort(angle)
			angle = angle[j_idx]
			r_int_h = r_int_h[j_idx]

			r_int_h = np.vstack((r_int_h, r_int_h[0]))
			angle = np.hstack((angle, angle[0] + 2*np.pi))


		self.r_int_h = r_int_h
		self.segments = segments
		self.ce_mesh = ce_mesh
		self.r_mesh = r_mesh

		r_edge = None
		ce_edge = 0
		if len(r_int_h) >= 2:
			r_edge = []
			for j in range(1,len(r_int_h)):
				r_int_h1, r_int_h2 = r_int_h[j-1], r_int_h[j]
				rc0 = unit(r0) * np.dot(r_int_h1,unit(r0))
				radius = mag(r_int_h1-rc0)
				rc1 = r_int_h1 - rc0
				rc2 = r_int_h2 - rc0

				# Omega = np.arccos(np.dot(unit(rc1),unit(rc2)))
				Omega = angle[j]-angle[j-1]
				tau = 0.5
				r_edge_test = rc1*np.sin((1-tau)*Omega)/np.sin(Omega) + \
							rc2*np.sin(tau*Omega)/np.sin(Omega) + rc0
				#
				in_view = self.point_in_view(r_edge_test, check_horizon=False)
				if not in_view:
					continue

				direction = 'short'
				if Omega > np.pi:
					direction = 'long'
				r_edge0 = theta_h_to_r_edge(r_int_h1, r_int_h2, r0, dist, direction)
				r_edge.append(r_edge0)

			if len(r_edge) > 0:
				ce_edge = 1
				r_edge = np.vstack(r_edge)


		coverage_exists = ce_mesh or ce_edge

		self.ce_edge = ce_edge
		self.r_edge = r_edge
		self.coverage_exists = coverage_exists

		r_fpt = np.array([])
		if coverage_exists:
			if ce_mesh and ce_edge:
				r_fpt = np.vstack((r_mesh, r_edge))
			elif ce_mesh:
				r_fpt = r_mesh
			elif ce_edge:
				r_fpt = r_edge

		self.r_fpt = r_fpt

		return r_fpt


	def get_w_hat(self):
		FOV_AT = self.FOV_AT
		FOV_CT = self.FOV_CT
		theta_AT = np.radians(FOV_AT)/2
		theta_CT = np.radians(FOV_CT)/2
		AZ0 = np.arctan(np.tan(theta_CT)/np.tan(theta_AT)) * 180/np.pi
		arg = 1 / ( np.tan(theta_AT)**2 + np.tan(theta_CT)**2 )
		EL0 = np.arctan(np.sqrt(arg)) * 180/np.pi
		AZ = np.radians(np.array([AZ0, 180-AZ0, 180+AZ0, 360-AZ0]))
		EL = np.radians(np.full(AZ.shape, EL0))
		w_hat = np.transpose([ np.cos(AZ)*np.cos(EL), np.sin(AZ)*np.cos(EL), np.sin(EL) ])
		w_hat = np.vstack((w_hat, w_hat[0]))
		return w_hat


	def point_in_view(self, r_pt, check_horizon=True):
		FOV_AT = self.FOV_AT
		FOV_CT = self.FOV_CT
		r0 = self.r0
		x_hat = self.x_hat
		y_hat = self.y_hat
		z_hat = self.z_hat

		single = 0
		if len(r_pt.shape) == 1:
			# r_pt = np.array([r_pt])
			single = 1

		if check_horizon:
			alpha_max = np.arccos(R_earth/mag(r0))
			alpha = np.arccos(np.dot(unit(r_pt),unit(r0)))
			b_in_view = alpha < alpha_max

		if single:
			u = r_pt-r0
			m = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
			# u_hat = u / m
			proj_x = u[0]*x_hat[0] + u[1]*x_hat[1] + u[2]*x_hat[2]
			proj_y = u[0]*y_hat[0] + u[1]*y_hat[1] + u[2]*y_hat[2]
			u_AT = u - proj_y*y_hat
			u_CT = u - proj_x*x_hat

			u_hat_AT = u_AT / np.sqrt(u_AT[0]**2 + u_AT[1]**2 + u_AT[2]**2)
			proj_gamma_AT = z_hat[0]*u_hat_AT[0] + z_hat[1]*u_hat_AT[1] + z_hat[2]*u_hat_AT[2]
			proj_gamma_AT = np.clip(proj_gamma_AT, -1, 1)
			gamma_AT = np.arccos(proj_gamma_AT)

			u_hat_CT = u_CT / np.sqrt(u_CT[0]**2 + u_CT[1]**2 + u_CT[2]**2)
			proj_gamma_CT = z_hat[0]*u_hat_CT[0] + z_hat[1]*u_hat_CT[1] + z_hat[2]*u_hat_CT[2]
			proj_gamma_CT = np.clip(proj_gamma_CT, -1, 1)
			gamma_CT = np.arccos(proj_gamma_CT)

			b_AT = gamma_AT <= np.radians(FOV_AT)/2
			b_CT = gamma_CT <= np.radians(FOV_CT)/2
			b_ATCT = b_AT & b_CT

			if check_horizon:
				b = b_ATCT & b_in_view
			else:
				b = b_ATCT

		else:
			u = r_pt-r0
			proj_x = np.dot(u,x_hat)
			proj_y = np.dot(u,y_hat)
			u_AT = u - multiply(y_hat,proj_y)
			u_CT = u - multiply(x_hat,proj_x)

			u_hat_AT = unit(u_AT)
			proj_gamma_AT = np.dot(u_hat_AT,z_hat)
			proj_gamma_AT = np.clip(proj_gamma_AT, -1, 1)
			gamma_AT = np.arccos(proj_gamma_AT)

			u_hat_CT = unit(u_CT)
			proj_gamma_CT = np.dot(u_hat_CT,z_hat)
			proj_gamma_CT = np.clip(proj_gamma_CT, -1, 1)
			gamma_CT = np.arccos(proj_gamma_CT)

			b_AT = gamma_AT <= np.radians(FOV_AT)/2
			b_CT = gamma_CT <= np.radians(FOV_CT)/2
			b_ATCT = b_AT & b_CT

			alpha_max = np.arccos(R_earth/mag(r0))
			alpha = np.arccos(np.dot(unit(r_pt),unit(r0)))
			b_in_view = alpha < alpha_max

			if check_horizon:
				b = b_ATCT & b_in_view
			else:
				b = b_ATCT

		return b



class LinearSegment:
	def __init__(self, r0, R, u_hat1, u_hat2, res, res_factor=0.25):
		dist = res*res_factor

		x_hat = R.T[0]
		y_hat = R.T[1]
		z_hat = R.T[2]

		self.r0 = r0
		self.x_hat = x_hat
		self.y_hat = y_hat
		self.z_hat = z_hat
		self.R = R
		self.dist = dist
		self.u_hat1 = u_hat1
		self.u_hat2 = u_hat2

		Omega = np.arccos(np.dot(u_hat1,u_hat2))
		self.Omega = Omega

		theta_h_vec, over_horizon = self.get_theta_h()
		theta_h = get_theta_reg(theta_h_vec, [0.0,1.0])
		REG_B = get_REG_B(r0, theta_h, self.F, [0.0,1.0])
		r_mesh = None
		ce_mesh = 0
		if len(REG_B) > 0:
			theta_mesh = get_theta_mesh(REG_B, r0, dist, self.F, self.H)
			if len(theta_mesh) > 0:
				r_mesh, u_hat = theta_mesh_to_r_int(theta_mesh, r0, self.F)
				ce_mesh = 1
		self.ce_mesh = ce_mesh
		self.r_mesh = r_mesh
		self.theta_h = theta_h
		self.theta_h_vec = theta_h_vec
		self.over_horizon = over_horizon


	def get_theta_h(self):

		r0 = self.r0
		Omega = self.Omega
		u_hat1, u_hat2 = self.u_hat1, self.u_hat2

		# tau_h = np.sqrt(mag(r0)**2 - R_earth**2)
		# const = (-tau_h**2 - (mag(r0)**2 - R_earth**2)) / (2*tau_h)
		const = -np.sqrt(mag(r0)**2 - R_earth**2)
		cx = np.dot(r0,u_hat1)
		cy = np.dot(r0,u_hat2)/np.sin(Omega) - np.dot(r0,u_hat1)/np.sin(Omega)*np.cos(Omega)
		c0 = const
		m = np.sqrt(cx**2 + cy**2)
		beta = np.arctan2(cx,cy)

		theta_h1, theta_h2 = np.nan, np.nan
		sin_arg = 2.0
		over_horizon = False
		if np.abs(m) > 1e-8:
			# m=0 when nadir?
			sin_arg = c0/m
			theta_h1, theta_h2 = arcsin(sin_arg, offset=beta)
			theta_h1 = theta_h1/Omega
			theta_h2 = theta_h2/Omega

			# if not np.isnan(theta_h1):
			# 	if not (0 <= theta_h1 < Omega):
			# 		theta_h1 = np.nan
			# 	if not (0 <= theta_h2 < Omega):
			# 		theta_h2 = np.nan

			if not (0 <= theta_h1 < 1.0):
				theta_h1 = np.nan
			if not (0 <= theta_h2 < 1.0):
				theta_h2 = np.nan
			over_horizon = True
			if np.isnan(theta_h1) and np.isnan(theta_h2):
				over_horizon = False

		return np.sort([theta_h1, theta_h2]), over_horizon


	def F(self, theta):
		Omega = self.Omega
		u_hat1, u_hat2 = self.u_hat1, self.u_hat2
		return u_hat1*np.sin((1-theta)*Omega)/np.sin(Omega) + \
				u_hat2*np.sin(theta*Omega)/np.sin(Omega)
		#

	def H(self, theta):
		Omega = self.Omega
		u_hat1, u_hat2 = self.u_hat1, self.u_hat2
		return -u_hat1*Omega*np.cos((1-theta)*Omega)/np.sin(Omega) + \
				u_hat2*Omega*np.cos(theta*Omega)/np.sin(Omega)
		#



class CircularSegment:
	def __init__(self, r0, R, u_hat0, res, res_factor=0.25):

		x_hat = R.T[0]
		y_hat = R.T[1]
		z_hat = R.T[2]

		dist = res*res_factor

		self.r0 = r0
		self.x_hat = x_hat
		self.y_hat = y_hat
		self.z_hat = z_hat
		self.R = R
		self.u_hat0 = u_hat0
		self.dist = dist

		theta_h_vec, over_horizon = self.get_theta_h()
		theta_h1, theta_h2 = theta_h_vec
		theta_h = get_theta_reg(theta_h_vec, [0.0,2*np.pi])
		REG_B = get_REG_B(r0, theta_h, self.F, [0.0,2*np.pi])
		r_mesh = None
		ce_mesh = 0
		if len(REG_B) > 0:
			theta_mesh = get_theta_mesh(REG_B, r0, dist, self.F, self.H)
			# for j in range(len(theta_mesh)):
			# 	tau = get_tau(r0,self.F(theta_mesh[j]))
			# 	print('tau', tau)
			# 	# if np.isnan(tau):
			# 	# 	print('tau', tau)

			if len(theta_mesh) > 0:
				r_mesh, u_hat = theta_mesh_to_r_int(theta_mesh, r0, self.F)
				ce_mesh = 1
		self.ce_mesh = ce_mesh
		self.r_mesh = r_mesh
		self.theta_h = theta_h
		self.theta_h_vec = theta_h_vec
		self.over_horizon = over_horizon


	def get_theta_h(self, tol=1e-8):

		"""
		dot(r0,u_hat) = +/-sqrt(r0**2 - R_earth**2)
		where u_hat = F(theta) (on FOV)

		Shouldn't it just be negative RHS? Only thetas
		that are valid are when dot product is negative..

		"""

		r0 = self.r0
		u_hat0 = self.u_hat0
		z_hat = self.z_hat

		def get_theta_h_params(sign):
			const = np.sign(sign)*np.sqrt(mag(r0)**2 - R_earth**2)
			cx = np.dot(r0,u_hat0) - np.dot(r0,z_hat)*np.dot(z_hat,u_hat0)
			cy = np.dot(r0,np.cross(z_hat,u_hat0))
			c0 = const-np.dot(r0,z_hat)*np.dot(z_hat,u_hat0)
			m = np.sqrt(cx**2 + cy**2)
			beta = np.arctan2(cx,cy)
			return c0, m, beta

		theta_h1, theta_h2 = np.nan, np.nan
		c0, m, beta = get_theta_h_params(-1)
		sin_arg = 2.0
		over_horizon = False
		if np.abs(m) > tol:
			# m=0 when nadir?
			sin_arg = c0/m
			theta_h1, theta_h2 = arcsin(sin_arg, offset=beta)
			if not np.isnan(theta_h1):
				over_horizon = True

		return np.sort([theta_h1, theta_h2]), over_horizon


	def F(self, theta):
		u_hat0 = self.u_hat0
		z_hat = self.z_hat
		return u_hat0*np.cos(theta) + \
			np.cross(z_hat,u_hat0)*np.sin(theta) + \
			z_hat*np.dot(z_hat,u_hat0)*(1-np.cos(theta))

	def H(self, theta):
		u_hat0 = self.u_hat0
		z_hat = self.z_hat
		return -u_hat0*np.sin(theta) + \
			np.cross(z_hat,u_hat0)*np.cos(theta) + \
			z_hat*np.dot(z_hat,u_hat0)*np.sin(theta)
