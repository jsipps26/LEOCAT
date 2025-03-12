

import numpy as np
from leocat.utils.index import hash_index


def angle_in_region(theta0, theta1, theta2):
	theta0 = theta0 % (2*np.pi)
	theta1 = theta1 % (2*np.pi)
	theta2 = theta2 % (2*np.pi)
	if theta1 < theta2:
		REG = [[theta1,theta2]]
	else:
		REG = [[theta1,2*np.pi],[0,theta2]]

	found = False
	for reg in REG:
		if reg[0] <= theta0 < reg[1]:
			found = True
			break

	return found

def arcsin_vec(arg, offset=0.0):
	gamma = np.full(arg.shape, np.nan)
	b = np.abs(arg) <= 1.0
	gamma[b] = np.arcsin(arg[b])
	gamma1 = gamma
	gamma2 = np.pi-gamma
	y1 = (gamma1-offset) % (2*np.pi)
	y2 = (gamma2-offset) % (2*np.pi)
	return y1, y2
	
def arcsin(arg, offset=0.0, solution=0):
	# Finds two solutions for theta in [0,2pi]
	# given an offset in theta. Can enforce 
	# first or second solution (modulo by 2pi)
	# by solution flag = 1 or 2
	if not solution:
		y1, y2 = np.nan, np.nan
		if np.abs(arg) <= 1.0:
			gamma = np.arcsin(arg)
			gamma1 = gamma
			gamma2 = np.pi-gamma
			y1 = (gamma1-offset) % (2*np.pi)
			y2 = (gamma2-offset) % (2*np.pi)
		return y1, y2

	elif solution == 1:
		y1 = np.nan
		if np.abs(arg) <= 1.0:
			gamma1 = np.arcsin(arg)
			y1 = (gamma1-offset) % (2*np.pi)
		return y1

	elif solution == 2:
		y2 = np.nan
		if np.abs(arg) <= 1.0:
			gamma1 = np.arcsin(arg)
			gamma2 = np.pi-gamma1
			y2 = (gamma2-offset) % (2*np.pi)
		return y2


def multiply(a_vec, b_vec):
	# vectorized multiplication
	# 	a_vec and b_vec are arrays of different lengths
	return (np.array([a_vec]).T * np.array([b_vec])).T

def nanmin(x, axis=None):
	if not np.isnan(x).all():
		return np.nanmin(x, axis=axis)
	return np.nan

def nanmax(x, axis=None):
	if not np.isnan(x).all():
		return np.nanmax(x, axis=axis)
	return np.nan

def nanmean(x, axis=None):
	# warningless nanmean
	if not np.isnan(x).all():
		return np.nanmean(x, axis=axis)
	return np.nan

def rad(val):
	return np.radians(val)

def deg(val):
	return np.degrees(val)


def get_hist(x, dx, x_mid=None, bounds=None, normalize=True):
	"""
	RETURNS BINS, HIST
	
	"""
	from pandas import DataFrame
	from leocat.utils.index import hash_index

	if x_mid is None:
		x_mid = np.nanmedian(x)
	if bounds is None:
		bounds = [-1e12, 1e12]
	bounds = np.array(bounds)

	x0 = x_mid - dx/2
	bx = (x > bounds[0]) & (x < bounds[1])
	x = x[bx]

	c = hash_index(x, x0, dx)
	c_min, c_max = np.min(c), np.max(c)
	cols = np.arange(c_min, c_max+1)

	df = DataFrame({'c': c, 'x': x})
	df_agg = df.groupby(['c']).agg({'x': [np.size]})
	df_agg.columns = ['x_agg']
	df_agg = df_agg.reset_index()
	x_agg = df_agg['x_agg'].to_numpy().astype(int)
	c_agg = df_agg['c'].to_numpy().astype(int)

	if normalize:
		hist = x_agg / np.sum(x_agg) / dx
	else:
		hist = x_agg
	bins = x0 + c_agg*dx + dx/2

	return bins, hist

	
def R1(theta):
	# vectorized ut.R1, active
	# R1 = lambda th: np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
	if not (type(theta) is np.ndarray):
		# scalar
		_R1 = lambda th: np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
		return _R1(theta)
	N = len(theta)
	I, Z = np.ones(N), np.zeros(N)
	Rc = np.cos(theta)
	Rs = np.sin(theta)
	R = np.array([[I, Z, Z], [Z, Rc, -Rs], [Z, Rs, Rc]])
	R = np.transpose(R, axes=(2,0,1)) # Nx3x3
	return R

def R2(theta):
	# vectorized ut.R2, active
	# R2 = lambda th: np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
	if not (type(theta) is np.ndarray):
		# scalar
		_R2 = lambda th: np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
		return _R2(theta)
	N = len(theta)
	I, Z = np.ones(N), np.zeros(N)
	Rc = np.cos(theta)
	Rs = np.sin(theta)
	R = np.array([[Rc, Z, Rs], [Z, I, Z], [-Rs, Z, Rc]])
	R = np.transpose(R, axes=(2,0,1)) # Nx3x3
	return R

def R3(theta):
	# vectorized ut.R3, active
	# R3 = lambda th: np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
	if not (type(theta) is np.ndarray):
		# scalar
		_R3 = lambda th: np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
		return _R3(theta)
	N = len(theta)
	I, Z = np.ones(N), np.zeros(N)
	Rc = np.cos(theta)
	Rs = np.sin(theta)
	# R = np.array([[I, Z, Z], [Z, Rc, -Rs], [Z, Rs, Rc]])
	R = np.array([[Rc, -Rs, Z], [Rs, Rc, Z], [Z, Z, I]])
	R = np.transpose(R, axes=(2,0,1)) # Nx3x3
	return R

def interp(t_new, t, vec):
	# N-dim np.interp
	vec_new = []
	for axis in vec.T:
		vec_new.append(np.interp(t_new,t,axis))
	vec_new = np.transpose(vec_new)
	return vec_new

def log10(arr):
	return log(arr, log_func=np.log10)

def divide(x, y):
	# warningless numpy division
	with np.errstate(divide='ignore', invalid='ignore'):
		return x/y

def log(arr, log_func=np.log):
	if type(arr) is list:
		arr = np.array(arr)
	if type(arr) is np.ndarray:
		b = arr > 0
		arr_out = np.full(arr.shape, np.nan)
		arr_out[b] = log_func(arr[b])
		return arr_out
	else:
		# single value
		if arr > 0:
			return log_func(arr)
		else:
			return np.nan



def nanprod(args, axis=None):
	"""
	This nanprod function preserves nans.

	Updated nanprod function from numpy.
	nanprod from numpy makes nans into 1s,
	which is a problem if both values are nan
	It works like the following cases:
	1. 	x *	y 			xy
	2. 	x *	np.nan 		x
	3. 	np.nan * y 		y
	4. np.nan * np.nan 	1

	This last case is working as numpy intended,
	but is not what I need. Domain must be 
	preserved; new information should be assigned
	properly (cases 1-3) but if no new information
	is present, that must remain nan. This code does
	this.

	"""
	x, y = args[0], args[1]

	if not type(x) is np.ndarray:
		x = np.array(x)
	if not type(y) is np.ndarray:
		y = np.array(y)

	if axis is None:
		return np.nanprod([x,y],axis=axis)

	a_nan = np.isnan(x)
	b_nan = np.isnan(y)
	m = np.nanprod([x, y], axis=axis)
	m[a_nan & b_nan] = np.nan

	return m


def nansum(args, axis=None):

	# see nanprod for details

	x, y = args[0], args[1]

	if not type(x) is np.ndarray:
		x = np.array(x)
	if not type(y) is np.ndarray:
		y = np.array(y)

	if axis is None:
		return np.nansum([x,y],axis=axis)

	a_nan = np.isnan(x)
	b_nan = np.isnan(y)
	m = np.nansum([x, y], axis=axis)
	m[a_nan & b_nan] = np.nan

	return m


def dot(v1, v2):
	# vectorized dot product
	# https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
	"""
	Possible inputs
	[0,0,0]
	np.array([0,0,0])
	[0,0]
	np.array([0,0])
	[[0,0,0],...,[0,0,0]]
	np.array([[0,0,0],...,[0,0,0]])
	[[0,0],...,[0,0]]
	np.array([[0,0],...,[0,0]])

	"""

	if not type(v1) is np.ndarray:
		v1 = np.array(v1)
	if not type(v2) is np.ndarray:
		v2 = np.array(v2)

	L1, L2 = len(v1.shape), len(v2.shape)
	if L1 == 2 and L2 == 2:
		return np.einsum('ij,ij->i', v1, v2)
	else:
		try:
			val = np.dot(v1,v2)
		except ValueError:
			val = np.dot(v2,v1)
		return val


def matmul(R, r):

	"""
	Vectorized matrix multiplication
	R and r have the same length

	R is in shape (N,3,3)
	r is in shape (N,3)

	Should be a way of doing this with np.einsum

	"""

	# by row
	# t1 = time.time()
	n = len(r)
	rf = r.flatten()

	Rx = R[:,0].flatten()
	xm = (Rx * rf).reshape((n,3))
	x = np.sum(xm, axis=1)

	Ry = R[:,1].flatten()
	ym = (Ry * rf).reshape((n,3))
	y = np.sum(ym, axis=1)

	Rz = R[:,2].flatten()
	zm = (Rz * rf).reshape((n,3))
	z = np.sum(zm, axis=1)

	r_rot = np.transpose([x,y,z])
	# t2 = time.time()
	# print(t2-t1)

	return r_rot


def wrap(angle, radians=False, two_pi=False):
	if not two_pi:
		# wraps angle to within
		#	[-pi,pi] or [-180,180] exactly
		if radians:
			x_wrap = np.arctan2(np.sin(angle), np.cos(angle))
		else:
			angle = np.radians(angle)
			x_wrap = np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))

	else:
		is_float = 0
		if not (type(angle) is np.ndarray):
			angle = np.array([angle])
			is_float = 1
		# wraps angle to within
		#	[0,2pi] or [0,360] exactly
		if radians:
			x_wrap = np.fmod(angle,2*np.pi)
			b = x_wrap < 0
			if b.any():
				x_wrap[b] = x_wrap[b] + 2*np.pi
		else:
			angle = np.radians(angle)
			x_wrap = np.fmod(angle,2*np.pi)
			b = x_wrap < 0
			if b.any():
				x_wrap[b] = x_wrap[b] + 2*np.pi
			x_wrap = np.degrees(x_wrap)

		if is_float:
			x_wrap = x_wrap[0]

	return x_wrap



def unwrap(x, radians=False):
	if radians:
		x_unwrap = np.unwrap(x)
	else:
		x_unwrap = np.degrees(np.unwrap(np.radians(x)))
	return x_unwrap

def quat_intp(JD_intp, JD, q, return_rot=False, warn=0, same=False):

	"""
	Vectorized interpolation using SLERP

	JD_intp is just a vector of times to interpolate to
	JD is the interpolant time-series
	quaternions follow the quaternion module format, so
		q[0] = [w, x, y, z]

	warn lets user know if any JD_intp pts fall outside
	of the q_data JD range

	same forces JD_intp to fall within the intp range of
	q_data; without "same", this function could return 
	R_intp as a different size than the input JD_intp.


	Caveats
	Assumes equal spacing in q_data JD times

	The quaternions in q_data should not be greater than
	or equal to 180 degrees from each other, as intp is
	ambiguous.

	Only as accurate as the spacing in q_data
		higher density q_data -> higher accuracy intp

	"""

	import quaternion

	# JD = q_data[:,0]
	# q = q_data[:,1:]

	# t1 = time.time()

	i0 = hash_index(JD_intp, JD[0], JD[1]-JD[0])
	i1 = i0 + 1

	b_valid = (i0 >= 0) & (i1 < len(JD))
	num_valid = b_valid.sum()
	if num_valid == 0:
		# print('b_valid 0')
		# return 
		raise IndexError('all intp points outside of quaternion time bounds')

	elif num_valid != len(JD_intp):
		if warn:
			print('warning: some intp points outside of quaternion time bounds')
		if same:
			raise IndexError('some intp points outside of quaternion time bounds')

	i0, i1 = i0[b_valid], i1[b_valid]
	JD0, JD1 = JD[i0], JD[i1]

	# slerp
	q0, q1 = quaternion.as_quat_array(q[i0]), quaternion.as_quat_array(q[i1])
	q_intp = quaternion.quaternion_time_series.slerp(q0, q1, JD0, JD1, JD_intp[b_valid])
	# t2 = time.time()
	# print('slerp', t2-t1)

	if return_rot:
		R_intp = quaternion.as_rotation_matrix(q_intp)
		return R_intp
	else:
		return quaternion.as_float_array(q_intp)



def newton_raphson(x0, f, fp, max_iter=2, iter_output=False):
	"""
	Vectorized NR
	Cannot quit before max iterations because all elements
	of x0 run each iteration... that is, if some stopped,
	then you'd have to track those. It's possible but it didn't
	seem necessary for the application.

	"""
	if iter_output:
		x_out = np.zeros((max_iter, len(x0)))

	x_prev = x0
	for i in range(max_iter):
		if iter_output:
			x_out[i,:] = x_prev

		# print(x_prev)
		x_new = x_prev - np.divide( f(x_prev), fp(x_prev) )
		dx_abs = np.abs(x_new - x_prev)
		# print(dx_abs)
		# save/remove elements that converge...
		# or remove elements that diverge

		x_prev = x_new

	if iter_output:
		return x_new, dx_abs, x_out
	else:
		return x_new, dx_abs


def mag(r):
	# magnitude of a vector
	#	either nx1, n-dim, or nxm
	#	vectorized
	if len(r.shape) > 1:
		return np.linalg.norm(r,axis=1)
	else:
		return np.linalg.norm(r)
		# if not array:
		# else:
		# 	return np.array([np.linalg.norm(r)])

		
def unit(v):
	if not isinstance(v,np.ndarray):
		v = np.array(v)
	if len(v.shape) == 1:
		mag0 = np.linalg.norm(v)
		# return v/mag0
		return divide(v,mag0)
	else:
		# n x 3 vector series
		# return (v.T/mag(v)).T
		return divide(v.T,mag(v)).T

def check_proj(proj):
	proj[proj > 1.0] = 1.0
	proj[proj < -1.0] = -1.0
	return proj

def convert_radians(x):
	if x is not None:
		return np.radians(x)
	return x

def scale(x, out_range=(-1, 1)):
	# https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range
	domain = np.min(x), np.max(x)
	y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
	return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def quartic_eqn(a,b,c,d,e):
	from leocat.fqs.fqs import multi_quartic
	roots = multi_quartic(a,b,c,d,e)
	roots = np.transpose([roots[0], roots[1], roots[2], roots[3]])
	return roots


def angle_about_z(r_in):
	vec = (r_in.T / mag(r_in)).T
	vec_xy = np.transpose([vec.T[0], vec.T[1]])
	vec_xy = (vec_xy.T / mag(vec_xy)).T
	proj_x = np.dot(vec_xy,[1,0])
	proj_y = np.dot(vec_xy,[0,1])

	angle = np.arccos(proj_x)
	angle[proj_y < 0] = 2*np.pi - angle[proj_y < 0]
	return angle

def interp_hat(t_new, t, x_hat):
	x0_hat = np.interp(t_new, t, x_hat.T[0])
	x1_hat = np.interp(t_new, t, x_hat.T[1])
	x2_hat = np.interp(t_new, t, x_hat.T[2])
	x_hat_interp = np.transpose([x0_hat,x1_hat,x2_hat])
	return unit(x_hat_interp)

def q_eqn(a, b, c):
	det = b**2 - 4*a*c
	valid = det >= 0
	sol1, sol2 = np.full(a.shape,np.nan), np.full(a.shape,np.nan)
	sol1[valid] = (-b[valid] + np.sqrt(det[valid]))/(2*a[valid])
	sol2[valid] = (-b[valid] - np.sqrt(det[valid]))/(2*a[valid])
	return sol1, sol2

def fit_parabola(tau, x0, x_dot0, x_dotdot0):
	return x0 + x_dot0*tau + 0.5*x_dotdot0*tau**2

def fit_parabola_der(tau, x_dot0, x_dotdot0):
	return x_dot0 + x_dotdot0*tau

def calc_vec_cross(a_vec, b_vec, known_len_a=True, known_len_b=True, angle_tol=0.01, warn=False, debug=0):

	"""
	a_vec and b_vec are numpy vectors consisting of two points each:
	a_vec: [point a0, point a1]
	b_vec: [point b0, point b1]
	Each point is a 3-dimensional array, such as np.array([1.2, 3.2, -4.5])

	angle_tol defines where a_vec and b_vec are parallel. There is no cross
	for parallel lines, but code will attempt to find one. angle_tol is in
	degrees.
	
	Returns the least-squares crossing point vector, if it exists in the
	bounds given by a_vec and b_vec. If a_vec and b_vec are close but not
	quite crossing, this will return the best estimate of the crossing
	position.

	"""
	def check_proj(proj):
		if proj > 1.0:
			proj = 1.0
		elif proj < -1.0:
			proj = -1.0
		return proj

	if type(a_vec) == list:
		a_vec = np.array(a_vec)
	if type(b_vec) == list:
		b_vec = np.array(b_vec)

	a0, b0 = a_vec[0], b_vec[0]
	da, db = a_vec[1]-a_vec[0], b_vec[1]-b_vec[0]
	proj = check_proj(np.dot(unit(da), unit(db)))
	angle = np.arccos(proj) * 180.0/np.pi

	# check that given vectors are not parallel, defined by angle_tol
	if angle_tol < angle < 180.0-angle_tol:
		p = np.array([b0[0]-a0[0], b0[1]-a0[1], b0[2]-a0[2]])
		M = np.array([[da[0], -db[0]],
					  [da[1], -db[1]],
					  [da[2], -db[2]]])
		tau_vec = np.matmul(np.linalg.pinv(M), p)

		c = 0
		if known_len_a and known_len_b:
			# since both lengths known, test both tau_vecs
			if 0 < tau_vec[0] < 1 and 0 < tau_vec[1] < 1:
				c = 1

		else:
			if known_len_a and not known_len_b:
				if 0 < tau_vec[0] < 1: # a has restriction
					c = 1

			elif (not known_len_a and known_len_b):
				if 0 < tau_vec[1] < 1: # b has restriction
					c = 1

		# if 0 < tau_vec[0] < 1 and 0 < tau_vec[1] < 1:
		if c:
			pt_cross_a = a0 + da*tau_vec[0]
			pt_cross_b = b0 + db*tau_vec[1]
			mag_err = np.sqrt(np.linalg.norm(da)**2 + np.linalg.norm(db)**2)
			# pt_cross_a == pt_cross_b, theoretically
			if np.linalg.norm(pt_cross_b - pt_cross_a) / mag_err > 0.01:
				# check within 1% of relative length
				if warn:
					print('warning: line cross solution may be poor')

			return 1, (pt_cross_a+pt_cross_b)/2 #np.array(pt_cross_a)
	else:
		# parallel
		return 2, np.array([0,0,0])

	return 0, np.array([0,0,0])