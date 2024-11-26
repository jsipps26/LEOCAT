
import numpy as np
from scipy.interpolate import CubicSpline

from leocat.utils.const import R_earth
from leocat.utils.general import pause
from leocat.utils.geodesy import spherical_dist, spherical_proj
from leocat.utils.math import mag, unit, dot, newton_raphson
from leocat.utils.index import hash_cr, hash_xy, cantor_pairing, unique_index, intersect_2d, boolean_cut_2d
from leocat.utils.cov import ray_cast_vec


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


# def project_corner_to_crosstrack(n_hat, r_corner_star, z_hat, r_cent_star):
def project_corner_to_crosstrack(v_cent_star, r_corner_star, r_cent_star):

	n_hat = unit(v_cent_star)
	z_hat = unit(r_cent_star)

	"""
	Exact (spherical)
		Level 1 swath envelope

	y_hat is crosstrack direction
	z_hat = unit(r_cent_star)

	1. Corner plane normal is perpendicular to
	crosstrack plane
	then
		corner plane normal must be on yz-plane

	2. Corner plane passes through the origin
		can derive psi, rotation thru yz-plane

	Normal to n_hat
	Normal to r_corner_star
	n_hat_corner = n_hat x r_corner_star

	# n_hat_corner = []
	# for j in range(len(r_corner_star)):
	# 	ri = r_corner_star[j]
	# 	# div by zero unlikely
	# 	# corner would need to be +/-90 deg from centroid
	# 	psi = np.arctan(-ut.dot(ri,y_hat)/ut.dot(ri,z_hat))
	# 	n_hati_x = y_hat.T[0]*np.cos(psi) + z_hat.T[0]*np.sin(psi)
	# 	n_hati_y = y_hat.T[1]*np.cos(psi) + z_hat.T[1]*np.sin(psi)
	# 	n_hati_z = y_hat.T[2]*np.cos(psi) + z_hat.T[2]*np.sin(psi)
	# 	n_hati = np.transpose([n_hati_x,n_hati_y,n_hati_z])
	# 	n_hat_corner.append(n_hati)
	# n_hat_corner = np.array(n_hat_corner)
	# n_hat = x_hat

	"""

	r_corner_avg_star = []
	# n_hat = x_hat
	for j in range(len(r_corner_star)):
		n_hat0 = unit(np.cross(n_hat,unit(r_corner_star[j])))
		u_hat = np.cross(n_hat,n_hat0)
		proj = dot(u_hat,z_hat)
		u_hat[proj < 0] = -u_hat[proj < 0]
		r_corner_avg_star.append(R_earth*u_hat)
	r_corner_avg_star = np.array(r_corner_avg_star)

	s_proj = []
	for j in range(len(r_corner_star)):
		# s_proj0 = spherical_proj(r_cent_star, r_corner_avg_star[j], v_cent_star)
		# s_proj.append(s_proj0)
		dist0 = spherical_dist(r_cent_star, r_corner_avg_star[j])
		fwd_vec = np.cross(r_corner_avg_star[j],r_cent_star) # nominally along v_cent_star
		sign = np.ones(len(r_cent_star))
		b_sign = dot(v_cent_star,fwd_vec) < 0
		sign[b_sign] = -sign[b_sign]
		s_proj.append(sign*dist0)
	s_proj = np.array(s_proj)
	# j_left = np.argmax(s_proj,axis=0).astype(int)
	# j_right = np.argmin(s_proj,axis=0).astype(int)

	# print(s_proj.shape)
	# left = np.max(s_proj,axis=0)
	# right = np.min(s_proj,axis=0)
	# print(np.unique(left).shape)
	# print(np.unique(right).shape)
	# print(r_corner_avg_star.shape)

	max_index = np.argmax(s_proj,axis=0) # left
	min_index = np.argmin(s_proj,axis=0) # right
	A = np.transpose(r_corner_avg_star,axes=(1,0,2))
	r_corner_star_left = A[np.arange(A.shape[0]),max_index,:]
	r_corner_star_right = A[np.arange(A.shape[0]),min_index,:]

	if 0:
		# Linear approximation
		# 	project r_corner_star right onto y_hat,
		# 	then normalize a point at that distance
		t1 = time.perf_counter()
		proj = []
		for j in range(len(r_corner_star)):
			proj.append( dot(r_corner_star[j]-r_cent_star,y_hat) )
		proj = np.array(proj)
		proj_left = np.max(proj,axis=0)
		proj_right = np.min(proj,axis=0)

		r_corner_star_left_lin = r_cent_star + (y_hat.T * proj_left).T
		r_corner_star_left_lin = R_earth*unit(r_corner_star_left_lin)
		r_corner_star_right_lin =  r_cent_star + (y_hat.T * proj_right).T
		r_corner_star_right_lin = R_earth*unit(r_corner_star_right_lin)

		t2 = time.perf_counter()
		print('linear approx',t2-t1)


	return r_corner_star_left, r_corner_star_right


def interp_corner_to_cent(t, r_corner_star0, v_cent_star, direction):

	# r_corner_star, t, v_cent_star, c_v_bar, c_a_var

	if direction == 'fwd':
		# "fwd"
		# 	find time along corner path s.t.
		#	spherical projection to track is orthogonal

		cw = CubicSpline(t,r_corner_star0)
		dcw = cw.derivative()
		def J_fwd(t, v):
			return dot(cw(t),v)
		def H_fwd(t, v):
			return dot(dcw(t),v)
		f = lambda t: J_fwd(t,v_cent_star)
		fp = lambda t: H_fwd(t,v_cent_star)

		tau_est = t
		tau_est, dtau = newton_raphson(tau_est, f, fp, max_iter=1, iter_output=False)

		tol_dtau = 1e-6
		b_tau = (dtau > tol_dtau)

		i_NR = 0
		max_iter = 10
		dtau_max = np.max(dtau)
		while dtau_max > tol_dtau:
			f = lambda t: J_fwd(t,v_cent_star[b_tau])
			fp = lambda t: H_fwd(t,v_cent_star[b_tau])
			tau_est[b_tau], dtau[b_tau] = newton_raphson(tau_est[b_tau], f, fp, max_iter=1, iter_output=False)
			dtau_max = np.max(dtau)
			i_NR += 1
			if i_NR == max_iter:
				break

		tau_est[dtau > tol_dtau] = np.nan
		# b = (tau_est < t[0]) | (tau_est > t[-1])
		# tau_est[b] = np.nan


	elif direction == 'rev':
		# "rev"
		# 	find time along track s.t.
		#	spherical projection to corner path is orthogonal

		def J_rev(t, r):
			return dot(r,c_v_bar(t))
		def H_rev(t, r):
			return dot(r,c_a_bar(t))
		f = lambda t: J_rev(t,r_corner_star0)
		fp = lambda t: H_rev(t,r_corner_star0)

		tau_est = t
		tau_est, dtau = newton_raphson(tau_est, f, fp, max_iter=1, iter_output=False)

		tol_dtau = 1e-6
		b_tau = (dtau > tol_dtau)

		i_NR = 0
		max_iter = 10
		dtau_max = np.max(dtau)
		while dtau_max > tol_dtau:
			f = lambda t: J_rev(t,r_corner_star0[b_tau])
			fp = lambda t: H_rev(t,r_corner_star0[b_tau])
			tau_est[b_tau], dtau[b_tau] = newton_raphson(tau_est[b_tau], f, fp, max_iter=1, iter_output=False)
			dtau_max = np.max(dtau)
			i_NR += 1
			if i_NR == max_iter:
				break

		tau_est[dtau > tol_dtau] = np.nan
		# b = (tau_est < t[0]) | (tau_est > t[-1])
		# tau_est[b] = np.nan


	return tau_est



def get_swath_envelope(t, r_ecf_sc, R_FOV, fp_geom, level=1):

	"""
	Notes
	"star" refers to a sphere with radius = R_earth
	The idea is that you can build the swath envelope on
	the star sphere, then project it to the ellipsoid
	- this way you save on a lot of ellipsoidal calculations
	- may be extensible to any geoid (instead of just ellipsoid),
		so long as the mapping from geoid to star is one-to-one

	"""

	# project footprint onto ellipsoid
	r_corner = project_footprint(r_ecf_sc, R_FOV, fp_geom)
	N_corner = len(r_corner)

	# footprint on star sphere
	r_corner_star = np.zeros(r_corner.shape)
	for j in range(N_corner):
		r_corner_star[j] = unit(r_corner[j])*R_earth

	# centroid of footprint on star sphere
	r_cent_star = np.mean(r_corner_star,axis=0)
	r_cent_star = unit(r_cent_star)*R_earth

	# spline fit of centroid (i.e. track pos/vel on star sphere)
	c_r_bar = CubicSpline(t,r_cent_star)
	c_v_bar = c_r_bar.derivative()
	# c_a_bar = c_v_bar.derivative() # for NR intp rev
	v_cent_star = c_v_bar(t)

	# instantaneous bounding circle def by max radius/corner on sphere
	dist_star = []
	for j in range(N_corner):
		# for each corner
		dist_star0 = spherical_dist(r_cent_star, r_corner_star[j])
		dist_star.append(dist_star0)
	dist_star = np.array(dist_star)
	max_dist_star = np.max(dist_star,axis=0)

	wx_hat = unit(v_cent_star)
	wz_hat = unit(r_cent_star)
	wy_hat = np.cross(wz_hat,wx_hat) # crosstrack direction

	phi = max_dist_star/R_earth

	# Project distance -along- sphere via spherical intp
	#	level 1 swath envelope
	#	fast footprint shape change -can- breach this bound, but for many
	#	cases it is enough
	w1_left_star = (wz_hat.T*np.cos(phi) + wy_hat.T*np.sin(phi)).T * R_earth
	w1_right_star = (wz_hat.T*np.cos(-phi) + wy_hat.T*np.sin(-phi)).T * R_earth
	w1_left = ray_cast_vec(w1_left_star, -unit(w1_left_star))
	w1_right = ray_cast_vec(w1_right_star, -unit(w1_right_star))

	if level <= 1:
		w_left = w1_left
		w_right = w1_right
		return w_left, w_right

	else:

		# Like projecting the corner path onto the track, except 
		# it's on a sphere now
		#	spherical projection
		w2_p_left_star, w2_p_right_star = \
			project_corner_to_crosstrack(v_cent_star, r_corner_star, r_cent_star)
		#

		# Interpolate along corner to find when its projection intersects track on sphere
		# t1 = time.time()
		tau_est_corner_fwd = []
		# tau_est_corner_rev = []
		for j in range(N_corner):
			tau_est_fwd = interp_corner_to_cent(t, r_corner_star[j], v_cent_star, direction='fwd')
			tau_est_corner_fwd.append(tau_est_fwd)
			# tau_est_rev = interp_to_swath_envelope(t, r_corner_star[j], v_cent_star, direction='rev')
			# tau_est_corner_rev.append(tau_est_rev)

		# t2 = time.time()
		# print('NR',t2-t1)

		# Find distances to corner paths from track on sphere
		# 	take maximum distance of all corners
		#	if NR fails, use max_dist_star from lvl 1 swath
		swath = []
		for j in range(N_corner):
			cw = CubicSpline(t, r_corner_star[j])
			w0 = cw(tau_est_corner_fwd[j]) # w2_p_star, both left and right
			proj_w0 = spherical_proj(r_cent_star, w0, v_cent_star)

			if 1:
				# limit NR by level 1 swath, w1_left/right
				#	will contain nans from unconverged sols
				b_right = proj_w0 < -max_dist_star
				proj_w0[b_right] = -max_dist_star[b_right]
				b_left = proj_w0 > max_dist_star
				proj_w0[b_left] = max_dist_star[b_left]

			swath.append(proj_w0) # has nans

		# print(r_cent_star.shape, w2_p_left_star.shape)
		# print(r_cent_star.shape, w2_p_right_star.shape)
		# sys.exit()
		proj_left = spherical_dist(r_cent_star, w2_p_left_star)
		proj_right = -spherical_dist(r_cent_star, w2_p_right_star)
		swath.append(proj_left)
		swath.append(proj_right)
		swath = np.array(swath)

		dist_left_star = np.nanmax(swath,axis=0)
		dist_right_star = np.nanmin(swath,axis=0)

		# spherical projection/intp to get swath vectors
		phi_left = dist_left_star/R_earth
		w_left_star = R_earth*(wz_hat.T*np.cos(phi_left) + wy_hat.T*np.sin(phi_left)).T

		phi_right = dist_right_star/R_earth
		w_right_star = R_earth*(wz_hat.T*np.cos(phi_right) + wy_hat.T*np.sin(phi_right)).T

		# After determining swath vectors on sphere, project final result onto ellipsoid
		w_left = ray_cast_vec(w_left_star, -unit(w_left_star))
		w_right = ray_cast_vec(w_right_star, -unit(w_right_star))

		return w_left, w_right


