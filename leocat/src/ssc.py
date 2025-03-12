
import numpy as np
import os, sys

from leocat.utils.const import *
from leocat.utils.plot import make_fig, set_axes_equal, set_aspect_equal, remove_axes, split_lon
from leocat.utils.geodesy import DiscreteGlobalGrid, poly_grid_cell, cart_to_RADEC

from leocat.utils.index import hash_xy_DGG, hash_cr_DGG, sort_col
from leocat.utils.cov import get_access_interval, project_pos
from leocat.utils.orbit import convert_ECI_ECF
from leocat.utils.math import mag, unit, nanprod

from pyproj import CRS, Transformer
from scipy.interpolate import CubicSpline
from pandas import DataFrame
from itertools import combinations
from tqdm import tqdm

from numba import njit


def check_SSC(col, row, access_intervals, SSC):
	DGG = SSC.space_params['DGG']
	tr_lla_ecf = SSC.space_params['tr_lla_ecf']
	cr = SSC.cr
	r_ecf = cr(t_ssc)
	lon_gt, lat_gt, _ = tr_lla_ecf.transform(r_ecf.T[0], r_ecf.T[1], r_ecf.T[2], direction='inverse')
	cols, rows = hash_cr_DGG(lon_gt, lat_gt, DGG)
	lon_idx = split_lon(lon_gt)

	keys = list(access_intervals.keys())
	# for k,(col,row) in enumerate(keys):
	intvl = access_intervals[(col,row)]
	# idx = np.where(np.diff(intvl,axis=1) == 0)[0]
	idx = indices[(col,row)]
	# if (np.diff(intvl,axis=1) == 0).any():
	# print(col, row)
	# print(intvl)
	# print(idx)
	# print(np.diff(idx))
	# print('')

	# fig, ax = make_fig()
	# ax.plot(idx, '.')
	# xlim = ax.get_xlim()
	# for j in range(len(intvl)):
	# 	ax.plot(xlim, [intvl[j][0], intvl[j][0]], 'g--')
	# 	ax.plot(xlim, [intvl[j][1], intvl[j][1]], 'r--')
	# fig.show()

	# pause()

	b = (cols == col) & (rows == row)
	lon_gt0 = lon_gt[b]
	lat_gt0 = lat_gt[b]

	r_min_offset = DGG.r_min_offset
	dlon = DGG.dlon[row-r_min_offset]
	dlat = DGG.dlat
	lon, lat = hash_xy_DGG(col, row, DGG)
	poly = poly_grid_cell(lon, lat, dlon, dlat, N_side=1)

	fig, ax = make_fig()
	for idx in lon_idx:
		ax.plot(lon_gt[idx], lat_gt[idx], c='C0')
	ax.plot(lon_gt0, lat_gt0, '.')
	xlim = [np.min(poly.T[0]), np.max(poly.T[0])]
	ylim = [np.min(poly.T[1]), np.max(poly.T[1])]
	dxlim, dylim = np.ptp(xlim), np.ptp(ylim)
	dxlim = dxlim/10
	dylim = dylim/10
	xlim = [xlim[0]-dxlim,xlim[1]+dxlim]
	ylim = [ylim[0]-dylim,ylim[1]+dylim]
	ax.plot(poly.T[0], poly.T[1], 'r')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	fig.show()

	# pause()
	# plt.close(fig)


def get_cr_idx(col, row, access_intervals):
	cr_idx = np.array(list(access_intervals.keys()))
	b = (cr_idx.T[0] == col) & (cr_idx.T[1] == row)
	i = np.where(b)[0][0]
	return i


@njit
def get_t_info(intvl, t_ssc):
	T_vec = []
	t_vec = []
	dt_ssc = t_ssc[1]-t_ssc[0]
	for (k1,k2) in intvl:
		num = k2-k1+1
		T = num*dt_ssc
		t0 = np.mean(t_ssc[k1:k2+1])
		t_vec.append(t0)
		T_vec.append(T)
	# t_vec = np.array(t_vec)
	# T_vec = np.array(T_vec)
	return t_vec, T_vec

@njit
def build_P(q):
	P0 = 0.0
	for k in range(len(q)):
		P0 = 1-(1-P0)*(1-q[k])
	return P0


@njit
def get_int_idx(r, u_hat):
	angle_tol = 0.01
	r_int = []
	idx_int = []
	for k in range(len(r)):
		for j in range(k):
			r1, r2 = r[k], r[j]
			u1, u2 = u_hat[k], u_hat[j]
			# q1, q2 = q[k], q[j]
			# q_c = q1*q2
			# rnd = np.random.uniform(0,1)
			# if rnd > q_c:
			# 	continue
			proj = np.dot(u1,u2)
			angle = np.degrees(np.arccos(proj))
			# if angle < angle_tol:
			if not (angle_tol < angle < 180.0-angle_tol):
				continue

			p = np.array([r2[0]-r1[0], r2[1]-r1[1], r2[2]-r1[2]])
			M = np.array([[u1[0],-u2[0]],
						  [u1[1],-u2[1]],
						  [u1[2],-u2[2]]])
			#
			# tau1, tau2 = np.matmul(np.linalg.pinv(M), p)
			tau_vec = np.linalg.pinv(M) @ p
			tau1, tau2 = tau_vec[0], tau_vec[1]

			r_int1 = r1 + tau1*u1
			r_int2 = r2 + tau2*u2
			r_int0 = (r_int1+r_int2)/2
			r_int.append(r_int0)
			idx_int.append([k,j])

	# r_int = np.array(r_int)
	# idx_int = np.array(idx_int)
	return r_int, idx_int


@njit
def get_int_idx_MC(r, u_hat, length, N_MC, q):
	tau = np.linspace(0,1,N_MC+1)[:-1]
	dtau = tau[1]-tau[0]
	tau = tau + dtau/2
	# L = tau*length
	# dL = dtau*length
	dL = length/N_MC

	r_MC = []
	for k in range(len(r)):
		r1 = r[k]-u_hat[k]*length/2 # tau=0
		r2 = r[k]+u_hat[k]*length/2 # tau=1

		rnd = np.random.uniform(0,1,N_MC)
		b = rnd < q[k]
		tau_q = tau[b]
		if len(tau_q) > 0:
			if q[k] > 0:
				tau_q = tau_q + np.random.uniform(-dtau/2,dtau/2)
			# r_MC0 = interp(tau_q, np.array([0,1]), np.array([r1,r2]))
			r_MC0 = np.zeros((len(tau_q),3))
			for kk in range(len(tau_q)):
				r_MC0[kk,:] = (r2-r1)/(1-0)*(tau_q[kk] - 0) + r1

		else:
			r_MC0 = np.zeros((0,3))

		r_MC.append(r_MC0)


	angle_tol = 0.01
	r_int_MC = []
	idx_MC = []

	for k in range(len(r)):
		r_MC_k = r_MC[k] # N_MC x 3
		N_MC_k = len(r_MC_k)
		if N_MC_k == 0:
			continue

		for j in range(k):
			r_MC_j = r_MC[j] # N_MC x 3
			N_MC_j = len(r_MC_j)
			if N_MC_j == 0:
				continue

			proj = np.dot(u_hat[k],u_hat[j])
			angle = np.degrees(np.arccos(proj))
			if not (angle_tol < angle < 180.0-angle_tol):
				continue

			for kk in range(N_MC_k):
				for jj in range(N_MC_j):
					r1 = r_MC_k[kk] - u_hat[k]*dL/2 # start of segment
					r2 = r_MC_j[jj] - u_hat[j]*dL/2
					u1, u2 = u_hat[k]*dL, u_hat[j]*dL

					p = np.array([r2[0]-r1[0], r2[1]-r1[1], r2[2]-r1[2]])
					M = np.array([[u1[0],-u2[0]],
								  [u1[1],-u2[1]],
								  [u1[2],-u2[2]]])
					#
					tau_vec = np.linalg.pinv(M) @ p
					tau1, tau2 = tau_vec[0], tau_vec[1]

					if (0 < tau1 < 1) and (0 < tau2 < 1):
						# line segments intersect
						r_int1 = r1 + tau1*u1
						r_int2 = r2 + tau2*u2
						r_int0 = (r_int1+r_int2)/2
						r_int_MC.append(r_int0)
						idx_MC.append([k,j,kk,jj])

	#
	return r_int_MC, idx_MC, r_MC



def MC_validation(col, row, t, r, u_hat, q_obs, M, SSC, N_MC=25, return_r_MC=False):

	tr_lla_ecf = SSC.space_params['tr_lla_ecf']
	res = SSC.space_params['res']
	DGG = SSC.space_params['DGG']

	length = 2*res

	dt_count_vec = []
	dt_avg_vec = []
	dt_min_vec = []
	dt_max_vec = []

	for I in range(M):
		# if M > 10 and (I+1) % 10 == 0:
		# 	print('%d/%d' % (I+1,M))

		# N_MC = 25
		dL = length/N_MC
		# t1 = time.time()
		r_int_MC, idx_MC, r_MC = get_int_idx_MC(r, u_hat, length, N_MC, q_obs)
		r_int_MC = np.array(r_int_MC)
		idx_MC = np.array(idx_MC)
		# t2 = time.time()
		# print(t2-t1)
		# pause()
			
		dt_avg0_MC = np.nan
		dt_min0_MC = np.nan
		dt_max0_MC = np.nan
		dt_count0_MC = 0
		if len(idx_MC) > 0:
			lon_int_MC, lat_int_MC, _ = \
					tr_lla_ecf.transform(r_int_MC.T[0], r_int_MC.T[1], r_int_MC.T[2], direction='inverse')
			#
			col_int_MC, row_int_MC = hash_cr_DGG(lon_int_MC, lat_int_MC, DGG)
			b = (col_int_MC == col) & (row_int_MC == row)
			r_int_MC = r_int_MC[b]
			idx_MC = idx_MC[b]
			if len(r_int_MC) > 0:

				k_int_MC = idx_MC.T[0]
				j_int_MC = idx_MC.T[1]
				t1_MC = t[j_int_MC]
				t2_MC = t[k_int_MC]
				dt_MC = t2_MC-t1_MC
				if (dt_MC < 0).any():
					print('warning: dt_MC < 0')

				dt_avg0_MC = np.mean(dt_MC)
				dt_min0_MC = np.min(dt_MC)
				dt_max0_MC = np.max(dt_MC)
				dt_count0_MC = len(idx_MC)

		dt_count_vec.append(dt_count0_MC)
		dt_avg_vec.append(dt_avg0_MC)
		dt_max_vec.append(dt_max0_MC)
		dt_min_vec.append(dt_min0_MC)

	dt_count_vec = np.array(dt_count_vec)
	dt_avg_vec = np.array(dt_avg_vec)
	dt_min_vec = np.array(dt_min_vec)
	dt_max_vec = np.array(dt_max_vec)

	# metrics = {'dt_count_vec': dt_count_vec,
	# 			'dt_avg_vec': dt_avg_vec,
	# 			'dt_min_vec': dt_min_vec,
	# 			'dt_max_vec': dt_max_vec}
	# #
	# return r_MC, metrics
	if return_r_MC:
		return dt_count_vec, dt_avg_vec, dt_min_vec, dt_max_vec, r_MC, r_int_MC
	else:
		return dt_count_vec, dt_avg_vec, dt_min_vec, dt_max_vec



###########################################

def get_num_combs(num):
	from math import comb
	num_comb = 0
	for j in range(0,num):
		num_comb = num_comb + comb(num,j+1)
	return num_comb


def get_combs(dt, func):
	idx0 = np.arange(len(dt))
	idx = []
	for j in range(1,len(idx0)+1):
		for subset in combinations(idx0,j):
			idx.append(list(subset))

	dt_vec = []
	for j,idx0 in enumerate(idx):
		t0 = func(dt[idx0])
		dt_vec.append(t0)

	return idx, dt_vec

def get_p_dt(p, idx, dt_vec, P_R, return_unique=False):
	num_pass = len(p)
	p_dt = []
	for j in range(len(dt_vec)):
		p0 = []
		for k in range(num_pass):
			if k in idx[j]:
				p0.append(p[k])
			else:
				p0.append(1-p[k])

		p0 = np.prod(p0)/P_R
		p_dt.append([dt_vec[j], p0])
	p_dt = np.array(p_dt)
	p_dt = sort_col(p_dt)

	return p_dt


def get_dt_true(p, idx, dt_vec, return_unique=False):
	P_R = 1 - np.prod(1-p) # SSC-specific
	p_dt = get_p_dt(p, idx, dt_vec, P_R, return_unique=return_unique)
	dt_true = np.dot(p_dt.T[0],p_dt.T[1])
	return dt_true, p_dt


def numba_get_dt_MC(t, q, N_MC, func):
	# All of these are SSC SPECIFIC
	#	do not use these functions in general
	if func is np.mean:
		dt_MC = numba_get_dt_MC_mean(t, q, N_MC)
	elif func is np.min:
		dt_MC = numba_get_dt_MC_min(t, q, N_MC)
	elif func is np.max:
		dt_MC = numba_get_dt_MC_max(t, q, N_MC)
	return dt_MC

@njit
def numba_get_dt_MC_mean(t, q, N_MC):
	# All of these are SSC SPECIFIC
	#	do not use these functions in general
	t_bar_MC = 0.0
	count = 0
	for i in range(N_MC):
		rnd = np.random.uniform(0,1,len(q))
		b = rnd < q
		num = b.sum()
		if num == 0:
			continue
		t_bar_MC = t_bar_MC + np.mean(t[b])
		count += 1
	if count > 0:
		t_bar_MC = t_bar_MC / count
	else:
		t_bar_MC = np.nan

	return t_bar_MC


@njit
def numba_get_dt_MC_min(t, q, N_MC):
	# All of these are SSC SPECIFIC
	#	do not use these functions in general
	t_bar_MC = 0.0
	count = 0
	for i in range(N_MC):
		rnd = np.random.uniform(0,1,len(q))
		b = rnd < q
		num = b.sum()
		if num == 0:
			continue
		t_bar_MC = t_bar_MC + np.min(t[b])
		count += 1
	if count > 0:
		t_bar_MC = t_bar_MC / count
	else:
		t_bar_MC = np.nan

	return t_bar_MC


@njit
def numba_get_dt_MC_max(t, q, N_MC):
	# All of these are SSC SPECIFIC
	#	do not use these functions in general
	t_bar_MC = 0.0
	count = 0
	for i in range(N_MC):
		rnd = np.random.uniform(0,1,len(q))
		b = rnd < q
		num = b.sum()
		if num == 0:
			continue
		t_bar_MC = t_bar_MC + np.max(t[b])
		count += 1
	if count > 0:
		t_bar_MC = t_bar_MC / count
	else:
		t_bar_MC = np.nan

	return t_bar_MC


def get_crossover_revisits(dt_ij, q_i=None, q_j=None, N_MC=None):

	dt_avg0 = np.mean(dt_ij)
	dt_min0 = np.min(dt_ij)
	dt_max0 = np.max(dt_ij)
	dt_count0 = float(len(dt_ij))

	if q_i is None and q_j is None:
		# deterministic
		metrics = {'mean': dt_avg0, 'min': dt_min0, 'max': dt_max0, 'count': dt_count0}
		return metrics

	else:
		b_i = ((q_i == 0.0) | (q_i == 1.0)).all()
		b_j = ((q_j == 0.0) | (q_j == 1.0)).all()
		if b_i and b_j:
			# q's are 0s or 1s -> filtered deterministic
			p_ij = q_i*q_j
			b_ij = p_ij > 0.0
			if b_ij.sum() > 0:
				dt_avg0 = np.mean(dt_ij[b_ij])
				dt_min0 = np.min(dt_ij[b_ij])
				dt_max0 = np.max(dt_ij[b_ij])
				dt_count0 = float(len(dt_ij[b_ij]))
			else:
				dt_avg0 = np.nan
				dt_min0 = np.nan
				dt_max0 = np.nan
				dt_count0 = 0.0

			metrics = {'mean': dt_avg0, 'min': dt_min0, 'max': dt_max0, 'count': dt_count0}
			return metrics

		else:
			# q's are stochastic
			dt_avg0_avg = np.nan
			dt_min0_avg = np.nan
			dt_max0_avg = np.nan
			dt_count0_avg = 0.0

			p_ij = q_i*q_j
			if (p_ij > 0.0).any():
				# num_comb = get_num_combs(len(dt_ij))
				# print(len(dt_ij), num_comb)
				if N_MC is None:
					N_MC = int(np.round(10000/len(dt_ij)))
					if N_MC < 100:
						N_MC = 100

				for func in [np.mean, np.min, np.max]:
					if len(dt_ij) < 15:
						# true soln
						idx_c, dt_c = get_combs(dt_ij, func)
						dt_c_avg, p_dt = get_dt_true(p_ij, idx_c, dt_c, return_unique=False)
					else:
						# estimated soln
						dt_c_avg = numba_get_dt_MC(dt_ij, p_ij, N_MC, func)

					if func is np.mean:
						dt_avg0_avg = dt_c_avg
					elif func is np.min:
						dt_min0_avg = dt_c_avg
					elif func is np.max:
						dt_max0_avg = dt_c_avg

				dt_count0_avg = np.sum(p_ij) # num revisits == num crossovers

			metrics = {'mean': dt_avg0_avg, 'min': dt_min0_avg, \
						'max': dt_max0_avg, 'count': dt_count0_avg}
			#
			return metrics


###########################################


class SmallSwathCoverage:
	"""
	Things to do
	Edge-case at sim start/end
		Can have few pts that aren't a full track
		across GC. Leads to incorrect coverage at 
		start/end GCs.

		Correct by limiting crossover calc not just
		by GC bounds but also by simulation period.

	"""

	def __init__(self, orb, swath, res, JD1, JD2):
		self.orb = orb
		self.swath = swath
		self.space_params = {'res': res}
		self.time_params = {'JD1': JD1, 'JD2': JD2}

		# self.set_space_params()
		# self.propagate_ground_track()

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


	def propagate_ground_track(self, dt_orb=10.0, off_nadir_C_vec=None, off_nadir_A_vec=None):
		JD1 = self.time_params['JD1']
		JD2 = self.time_params['JD2']
		tr_lla_ecf = self.space_params['tr_lla_ecf']
		orb = self.orb

		T_sim = (JD2-JD1)*86400
		N_orb = int(T_sim/dt_orb)+1
		t_orb = np.linspace(0,T_sim,N_orb)
		dt_orb = t_orb[1]-t_orb[0]
		JD_orb = t_orb/86400 + JD1

		r_eci_sc, v_eci_sc = orb.propagate(t_orb)
		r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD_orb, r_eci_sc, v_eci_sc)

		# R_FOV = SC_frame(r_ecf_sc, v_ecf_sc, off_nadir_C_vec, off_nadir_A_vec, LVLH=LVLH) # Nx3x3
		# fp_geom = Inst.get_FOV_geometry()
		# swath_level = 1
		# if FOV_shape == 'rectangular':
		# 	swath_level = 2
		# w_l_ecf, w_r_ecf = cov_mesh.get_swath_envelope(t, r_ecf_sc, R_FOV, fp_geom, level=swath_level)
		# w_ev = ellipsoidal_dist(w_l_ecf, w_r_ecf, tr_lla_ecf)

		if off_nadir_C_vec is None and off_nadir_A_vec is None:
			lon, lat, _ = tr_lla_ecf.transform(r_ecf_sc.T[0], r_ecf_sc.T[1], r_ecf_sc.T[2], direction='inverse')
			r_ecf = tr_lla_ecf.transform(lon, lat, np.zeros(lon.shape))
			r_ecf = np.transpose([r_ecf[0],r_ecf[1],r_ecf[2]])
		else:
			lon, lat, slant_rgt, r_ecf = \
					project_pos(r_ecf_sc, v_ecf_sc, tr_lla_ecf, off_nadir_C_vec=off_nadir_C_vec, off_nadir_A_vec=off_nadir_A_vec)
			#

		cr = CubicSpline(t_orb,r_ecf)
		cv = cr.derivative()
		v_ecf = cv(t_orb)

		self.cr = cr
		self.cv = cv
		self.v_ecf_mag_max = np.max(mag(v_ecf))

		self.time_params['T_sim'] = T_sim
		self.time_params['dt_orb'] = dt_orb


	def get_time(self, res_factor_AT=0.05):

		self.set_space_params()
		self.propagate_ground_track()
		
		if res_factor_AT > 0.25:
			import warnings
			warnings.warn('res_factor_AT > 0.25, SSC may miss grid cells')

		v_ecf_mag_max = self.v_ecf_mag_max
		res = self.space_params['res']
		T_sim = self.time_params['T_sim']

		dt_intp = res_factor_AT*res/v_ecf_mag_max
		Nt = int(T_sim/dt_intp) + 1
		t_ssc = np.linspace(0,T_sim,Nt)
		return t_ssc


	def get_access(self, t_ssc, return_indices=False, verbose=1):

		if verbose:
			print('computing access..')

		# self.space_params = {'res': self.res}
		# self.time_params = {'JD1': self.JD1, 'JD2': self.JD2}

		# self.set_space_params()
		# self.propagate_ground_track()
		# if t_ssc is None:
		# 	t_ssc = self.get_time()
		# self.t_ssc = t_ssc

		space_params = self.space_params
		cr = self.cr
		res = space_params['res']
		tr_lla_ecf = space_params['tr_lla_ecf']
		DGG = space_params['DGG']
		r_ecf_intp = cr(t_ssc)
		lon_intp, lat_intp, _ = \
			tr_lla_ecf.transform(r_ecf_intp.T[0], r_ecf_intp.T[1], r_ecf_intp.T[2], direction='inverse')
		#

		cols, rows = hash_cr_DGG(lon_intp, lat_intp, DGG)
		df = DataFrame({'c': cols, 'r': rows}) # , 't': t_ssc})
		group_by = df.groupby(['c','r'], sort=False)
		indices = group_by.indices
		# keys = list(indices.keys())
		# cols, rows = np.array(keys).T # unique c,r

		access_intervals = get_access_interval(indices)

		if return_indices:
			return access_intervals, indices

		return access_intervals


	def get_lonlat(self, access_intervals):
		DGG = self.space_params['DGG']
		cols, rows = np.transpose(list(access_intervals.keys()))
		lon_GP, lat_GP = hash_xy_DGG(cols, rows, DGG)
		return lon_GP, lat_GP


	def get_metrics(self, t_ssc, access_intervals, revisit=False, GL=None, verbose=2, debug_MC=0):
		DGG = self.space_params['DGG']
		cr = self.cr
		cv = self.cv
		A = self.space_params['A']
		res = self.space_params['res']
		tr_lla_ecf = self.space_params['tr_lla_ecf']

		time_params = self.time_params
		JD1, JD2 = time_params['JD1'], time_params['JD2']

		swath = self.swath

		if verbose > 0:
			print('computing FOMs..')
		keys = list(access_intervals.keys())
		iterator = range(len(keys))
		if verbose > 1:
			iterator = tqdm(iterator)

		N, P = [], []
		num_pass = []

		if revisit:
			dt_avg = np.full(len(keys),np.nan)
			dt_min = np.full(len(keys),np.nan)
			dt_max = np.full(len(keys),np.nan)
			dt_count = np.zeros(len(keys))

			if debug_MC:
				dt_avg_avg_MC = np.full(len(keys),np.nan)
				dt_min_avg_MC = np.full(len(keys),np.nan)
				dt_max_avg_MC = np.full(len(keys),np.nan)
				dt_count_avg_MC = np.zeros(len(keys))

		for i in iterator:
			col, row = keys[i]
			intvl = access_intervals[(col,row)]
			t, T = get_t_info(intvl, t_ssc)
			t = np.array(t)
			T = np.array(T)
			v = cv(t)
			L = T*mag(v)
			q_ssc = swath*L/A

			q_obs = np.ones(q_ssc.shape)
			if not (GL is None):
				lon0, lat0 = hash_xy_DGG(col, row, DGG)
				lon = np.full(q_ssc.shape,lon0)
				lat = np.full(q_ssc.shape,lat0)
				JD = JD1 + t/86400
				q_obs = GL.get_quality(lon, lat, JD)
			q = nanprod([q_ssc,q_obs],axis=0)
			# q = q_obs

			N0 = np.sum(q)
			P0 = build_P(q)

			N.append(N0)
			P.append(P0)
			num_pass.append(len(t))

			if revisit:
				if len(t) > 1:
					debug_int = 0
					# if len(t) > 10:
					# 	debug_int = 1

					r = cr(t) # mid-points of each track in ECEF
					u_hat = unit(v) # direction of tracks in ECEF
					r_int, idx_int = get_int_idx(r, u_hat)
					r_int = np.array(r_int)
					idx_int = np.array(idx_int)

					if len(idx_int) > 0:
						lon_int, lat_int, _ = \
								tr_lla_ecf.transform(r_int.T[0], r_int.T[1], r_int.T[2], direction='inverse')
						#
						col_int, row_int = hash_cr_DGG(lon_int, lat_int, DGG)
						b = (col_int == col) & (row_int == row)
						r_int = r_int[b]
						idx_int = idx_int[b]

					if len(idx_int) > 0:
						j_int = idx_int.T[0]
						i_int = idx_int.T[1]
						t_i = t[i_int]
						t_j = t[j_int]
						dt_ij = t_j-t_i

						# dt_avg0 = np.mean(dt_ij)
						# dt_min0 = np.min(dt_ij)
						# dt_max0 = np.max(dt_ij)
						# dt_count0 = len(idx_int)

						q_i = q_obs[i_int]
						q_j = q_obs[j_int]
						revisit_metrics = get_crossover_revisits(dt_ij, q_i, q_j)

						dt_avg[i] = revisit_metrics['mean']
						dt_min[i] = revisit_metrics['min']
						dt_max[i] = revisit_metrics['max']
						dt_count[i] = revisit_metrics['count']

						if debug_MC:
							M = debug_MC
							N_MC = 25
							length = 2*res
							dL = length/N_MC
							dt_count_vec, dt_avg_vec, dt_min_vec, dt_max_vec, r_MC, r_int_MC = \
									MC_validation(col, row, t, r, u_hat, q_obs, M, self, return_r_MC=True)
							#
							dt_count0_avg_MC = np.mean(dt_count_vec)
							dt_avg0_avg_MC = np.nan
							dt_min0_avg_MC = np.nan
							dt_max0_avg_MC = np.nan
							if not (np.isnan(dt_avg_vec)).all():
								dt_avg0_avg_MC = np.nanmean(dt_avg_vec)
							if not (np.isnan(dt_min_vec)).all():
								dt_min0_avg_MC = np.nanmean(dt_min_vec)
							if not (np.isnan(dt_max_vec)).all():
								dt_max0_avg_MC = np.nanmean(dt_max_vec)

							# print(dt_count0_avg, dt_count0_avg_MC)
							# print(dt_avg0_avg/86400, dt_avg0_avg_MC/86400)
							# print(dt_min0_avg/86400, dt_min0_avg_MC/86400)
							# print(dt_max0_avg/86400, dt_max0_avg_MC/86400)
							# pause()

							dt_avg_avg_MC[i] = dt_avg0_avg_MC
							dt_min_avg_MC[i] = dt_min0_avg_MC
							dt_max_avg_MC[i] = dt_max0_avg_MC
							dt_count_avg_MC[i] = dt_count0_avg_MC


						if debug_int:
							r_min_offset = DGG.r_min_offset
							dlon = DGG.dlon[row-r_min_offset]
							dlat = DGG.dlat
							lon, lat = hash_xy_DGG(col, row, DGG)
							poly = poly_grid_cell(lon, lat, dlon, dlat, N_side=10)
							poly_ecf = tr_lla_ecf.transform(poly.T[0], poly.T[1], np.zeros(len(poly)))
							poly_ecf = np.transpose([poly_ecf[0],poly_ecf[1],poly_ecf[2]])

							e_hat1 = unit(poly_ecf[1]-poly_ecf[0])
							e_hat2 = unit(poly_ecf[-2]-poly_ecf[0])
							n_hat = unit(np.cross(e_hat1,e_hat2))
							RA, DEC = cart_to_RADEC(n_hat)

							fig, ax = make_fig('3d')
							ax.plot(poly_ecf.T[0], poly_ecf.T[1], poly_ecf.T[2], 'k', lw=3.0)

							length = 2*res
							for k in range(len(t)):
								r1 = r[k]-u_hat[k]*length/2
								r2 = r[k]+u_hat[k]*length/2
								# draw_vector(ax, r1, r2, 'k')
								pos_x = [r1[0], r2[0]]
								pos_y = [r1[1], r2[1]]
								pos_z = [r1[2], r2[2]]
								ax.plot(pos_x, pos_y, pos_z, 'k', alpha=0.25) #, lw=3.0, alpha=0.5)

							if debug_MC:
								for k in range(len(t)):
									r_MC_k = r_MC[k]
									N_MC_k = len(r_MC_k)
									for kk in range(N_MC_k):
										r1 = r_MC_k[kk] - u_hat[k]*dL/2
										r2 = r_MC_k[kk] + u_hat[k]*dL/2
										# draw_vector(ax, r1, r2, linestyle='--')
										pos_x = [r1[0], r2[0]]
										pos_y = [r1[1], r2[1]]
										pos_z = [r1[2], r2[2]]
										ax.plot(pos_x, pos_y, pos_z, lw=3.0, c='C0')


								if len(r_int_MC) > 0:
									for scatter_is_annoying in range(5):
										ax.scatter(r_int_MC.T[0], r_int_MC.T[1], r_int_MC.T[2], \
													s=100, facecolors='none', edgecolors='r')
										#

							ax.plot(r_int.T[0], r_int.T[1], r_int.T[2], 'rx')

							ax.view_init(azim=RA, elev=DEC)
							set_axes_equal(ax)
							set_aspect_equal(ax)
							remove_axes(fig,ax)
							ax.set_title('(%d,%d)' % (col,row))
							fig.show()

							pause()
							plt.close(fig)

				#

		N = np.array(N)
		P = np.array(P)
		num_pass = np.array(num_pass)

		metrics = {'num_obs': N, 'p_cov': P, 'num_pass': num_pass}
		if revisit:
			metrics['dt_avg'] = dt_avg
			metrics['dt_min'] = dt_min
			metrics['dt_max'] = dt_max
			metrics['dt_count'] = dt_count

			if debug_MC:
				metrics['dt_avg_MC'] = dt_avg_avg_MC
				metrics['dt_min_MC'] = dt_min_avg_MC
				metrics['dt_max_MC'] = dt_max_avg_MC
				metrics['dt_count_MC'] = dt_count_avg_MC
			

		return metrics


