

import numpy as np
import os,sys
from leocat.utils.const import *

from leocat.utils.general import pause
from leocat.utils.math import unit, unwrap
from leocat.utils.orbit import get_GMST
from leocat.utils.cov import swath_to_FOV

from tqdm import tqdm

from pandas import DataFrame

# import warnings
# from copy import deepcopy

from numba import njit #, types, typed


class FieldMapCoverage:
	"""
	Reference
	C. Han, Y. Zhang, and S. Bai, “Geometric Analysis of Ground-Target 
	Coverage from a Satellite by Field-Mapping Method,” Journal of 
	Guidance, Control, and Dynamics, vol. 44, no. 8, pp. 1469–1480, 
	Aug. 2021, doi: 10.2514/1.G005719.

	Remaining details
		Robustly ensuring entire simulation period is sampled
		Replacing shapely with jit-intersection
		Generalizing t0 to different omega,nu init
		Not handling polar obs. on every rev
		Equatorial inclination
		Robust testing
		flipud not robust for len(AZ_range) = 2

	"""
	def __init__(self, orb, swath, lon, lat, JD1, JD2):
		self.orb = orb
		self.swath = swath
		self.lon = lon
		self.lat = lat
		self.JD1 = JD1
		self.JD2 = JD2

		if orb.omega != 0.0 or orb.nu != 0.0:
			raise Exception('orbit omega and nu must be zero (for now)')


	def get_access(self, verbose=1, debug=0, num_AZ=2):
		orb = self.orb
		swath = self.swath
		lon, lat = self.lon, self.lat
		JD1, JD2 = self.JD1, self.JD2

		if debug:
			import matplotlib.pyplot as plt
			from leocat.utils.plot import pro_plot, make_fig
			pro_plot()

		t0 = 0.0
		num_days = int(np.ceil(JD2-JD1)) + 1

		alt = orb.get_alt()
		FOV = swath_to_FOV(swath,alt)
		Tn = orb.get_period('nodal')
		Dn = orb.get_nodal_day()
		theta_g0 = get_GMST(JD1)

		AZ_range = np.array([np.pi/2, 3*np.pi/2])
		if num_AZ > 2:
			AZ_range = np.linspace(0,2*np.pi,num_AZ)

		a = R_earth + alt
		# lam, phi = np.radians(lon), np.radians(lat)

		inc_rad = orb.inc
		LAN0 = orb.LAN - np.radians(theta_g0)


		df = DataFrame({'lat': lat})
		lat_data = df.groupby('lat', sort=False).indices
		keys = list(lat_data.keys())


		EL = np.radians(90-FOV/2)

		u_dot = orb.get_omega_dot() + orb.get_M_dot()
		LAN_dot = orb.get_LAN_dot()
		K = u_dot/(LAN_dot - W_EARTH)
		n_hat = unit(np.array([1,-1/K]))

		alpha = swath / (2*R_earth)
		theta_e = alpha

		iterator = range(len(keys))
		if verbose:
			iterator = tqdm(iterator)

		t_access_init_total = []
		index_total = []
		for k in iterator:
			lat0 = keys[k]
			lon_idx = lat_data[lat0]
			lon_vec = lon[lon_idx]

			lam_vec = np.radians(lon_vec)
			phi = np.radians(lat0)

			complete = 1
			if (inc_rad-theta_e <= np.abs(phi) <= inc_rad+theta_e):
				complete = 0

			det = R_earth**2 - (a*np.cos(EL))**2
			if det < 0:
				print('error: det < 0')
			rho = a*np.sin(EL) - np.sqrt(det)
			lower = R_earth*np.sin(phi - inc_rad)
			upper = R_earth*np.sin(phi + inc_rad)

			u1_vec, u2_vec, LAN_G1_vec, LAN_G2_vec = \
				get_u_LAN_vec(lam_vec, phi, AZ_range, rho, EL, lower, upper, a, inc_rad)
			#

			for kk in range(len(lam_vec)):
				lon0 = np.degrees(lam_vec[kk])
				poly1, poly2, count = get_poly_maps(kk, u1_vec, u2_vec, LAN_G1_vec, LAN_G2_vec)

				if count == 0:
					t_bar = np.array([])

				else:
					poly1 = np.vstack(poly1)
					poly2 = np.vstack(poly2)
					# poly = format_poly(poly1, poly2, complete, len(AZ_range))


					num = len(AZ_range)
					if num > 2:
						poly1.T[0] = unwrap(poly1.T[0])
						poly1.T[1] = unwrap(poly1.T[1])
						poly2.T[0] = unwrap(poly2.T[0])
						poly2.T[1] = unwrap(poly2.T[1])
						poly1 = np.vstack((poly1,poly1[0]))
						poly2 = np.vstack((poly2,poly2[0]))

						poly = []
						if not complete:
							x_hat = np.array([1,0])
							y_hat = np.array([0,1])

							poly3 = np.vstack((poly1[:-1], poly2[:-1]))
							poly3.T[0] = unwrap(poly3.T[0])
							poly3.T[1] = unwrap(poly3.T[1])
							pt_mid = np.mean(poly3,axis=0)
							dr3 = poly3 - pt_mid
							theta = np.arctan2(np.dot(dr3,y_hat),np.dot(dr3,x_hat))
							idx_rot = np.argsort(theta)
							poly3 = poly3[idx_rot]
							poly3 = np.vstack((poly3, poly3[0]))
							poly.append(poly3)

							# fig, ax = make_fig()
							# ax.plot(dr1.T[0], dr1.T[1])
							# ax.plot(dr2.T[0], dr2.T[1])
							# fig.show()

							# fig, ax = make_fig()
							# ax.plot(poly1.T[0], poly1.T[1])
							# ax.plot(poly2.T[0], poly2.T[1])
							# ax.plot(pt_mid[0], pt_mid[1], 'r.')
							# ax.plot(poly3.T[0], poly3.T[1], 'k--')
							# fig.show()

							# sys.exit()

							# poly3 = np.vstack((np.flipud(poly1[:-1]), poly2[:-1]))
							# poly3 = np.vstack((poly3, poly3[0]))
							# poly3.T[0] = unwrap(poly3.T[0])
							# poly3.T[1] = unwrap(poly3.T[1])
							# poly.append(poly3)

						else:
							poly.append(poly1)
							poly.append(poly2)

					else:
						# single line
						# poly1 = np.vstack(poly1)
						# poly2 = np.vstack(poly2)
						poly = []
						if not complete:
							poly3 = np.vstack((np.flipud(poly1), poly2))
							poly3.T[0] = unwrap(poly3.T[0])
							poly3.T[1] = unwrap(poly3.T[1])
							poly.append(poly3)
						else:
							# poly1.T[0] = unwrap(poly1.T[0])
							# poly1.T[1] = unwrap(poly1.T[1])
							# poly2.T[0] = unwrap(poly2.T[0])
							# poly2.T[1] = unwrap(poly2.T[1])
							poly.append(poly1)
							poly.append(poly2)

					#

					reg = get_regions(poly, num_days, n_hat)
					k_range, proj = get_k_proj(LAN0, LAN_dot, Tn, num_days, K, n_hat)
					idx, idx_poly = get_orbit_poly_indices(proj, reg, k_range)

					t_bar, poly_access, points_int = \
						get_t_bar(poly, idx, idx_poly, u_dot, t0, complete, Tn, LAN0, K, num_days, \
								len(AZ_range), return_points=True)
					#
					if len(t_bar) == 0:
						continue

					t_access_init_total.append(t_bar)
					index_total.append(np.full(len(t_bar),lon_idx[kk]))

					if debug:
						poly_shift = []
						for j in range(num_days+1):
							for poly0 in poly:
								poly_shift.append(np.transpose([poly0.T[0]-360*j,poly0.T[1]]))
						poly_shift = np.array(poly_shift)

						fig, ax = make_fig()
						for poly0 in poly_shift:
							ax.plot(poly0.T[0], poly0.T[1], 'k', alpha=0.25)
						for poly0 in poly_access:
							ax.plot(poly0.T[0], poly0.T[1])

						# ax.plot(poly[1].T[0], poly[1].T[1], 'r')
						# N = 10
						# N_range = np.arange(1,N+1)
						

						lines = get_lines(k_range, LAN0, K)
						for [p1,p2] in lines:
							pos_x = np.array([p1[0], p2[0]])
							pos_y = np.array([p1[1], p2[1]])
							ax.plot(pos_x, pos_y, 'k', alpha=0.25)

						lines = get_lines(idx, LAN0, K)
						for [p1,p2] in lines:
							pos_x = np.array([p1[0], p2[0]])
							pos_y = np.array([p1[1], p2[1]])
							ax.plot(pos_x, pos_y)

						if 1:
							for points in points_int:
								pt1 = points[0]
								try:
									pt2 = points[1]
									ax.plot(pt1[0],pt1[1],'.',c=[0,1,0])
									ax.plot(pt2[0],pt2[1],'r.')
								except IndexError:
									ax.plot(pt1[0],pt1[1],'r.')


						# ax.set_xlim(0,360)
						# ax.set_ylim(0,360)
						# ax.grid()

						ax.set_xlabel(r'$\Omega + \lambda$' + ' (deg)')
						ax.set_ylabel(r'$u$' + ' (deg)')

						title = f'lon = {lon0:.2f}, lat = {lat0:.2f}'
						ax.set_title(title)

						fig.show()

						pause()
						plt.close(fig)
		#

		t_access_init_total = np.concatenate(t_access_init_total)
		index_total = np.concatenate(index_total)

		b = (t_access_init_total >= 0.0) & (t_access_init_total < (0.0 + (JD2-JD1)*86400))
		t_access_init_total = t_access_init_total[b]
		index_total = index_total[b]

		t_FM = {}
		if b.sum() > 0:
			df = DataFrame({'index': index_total})
			index_indices = df.groupby('index').indices
			for key in index_indices:
				idx = index_indices[key]
				t_key = t_access_init_total[idx]
				j_sort = np.argsort(t_key)
				idx = idx[j_sort]
				t_FM[key] = t_access_init_total[idx]
				# t_BT[key] = np.sort(t_access_init_total[idx])
				# if GL_flag:
				# 	q_BT[key] = q_total[idx]

		return t_FM



def get_t_bar(poly, idx, idx_poly, u_dot, t0, complete, Tn, LAN0, K, num_days, \
				num_AZ, return_points=True):

	from shapely.geometry import LineString, Polygon
	from shapely import get_coordinates

	t_node = t0 + Tn*idx
	lines_access = get_lines(idx, LAN0, K)

	poly_access = []
	points_int = []
	t_bar = []
	for i in range(len(idx)):
		k, j = idx[i], idx_poly[i]
		if complete:
			if j < num_days+1:
				poly0 = poly[0]
				days = j
			else:
				poly0 = poly[1]
				days = j-(num_days+1)
		else:
			poly0 = poly[0]
			days = j

		poly_access0 = np.transpose([poly0.T[0]-360*days, poly0.T[1]])
		lines_access0 = lines_access[i]
		l0 = LineString([lines_access0[0], lines_access0[1]])
		if num_AZ > 2:
			p0 = Polygon(poly_access0)
			ints = p0.boundary.intersection(l0)
		else:
			p0 = LineString([poly_access0[0], poly_access0[1]])
			ints = p0.intersection(l0)
		points = get_coordinates(ints)

		u_int0 = points.T[1]
		idx_sort = np.argsort(u_int0)
		points = points[idx_sort]
		u_int0 = u_int0[idx_sort]

		t_int0 = t_node[i] + u_int0 / np.degrees(u_dot)
		t_bar.append(np.mean(t_int0))

		points_int.append(points)
		poly_access.append(poly_access0)

	points_int = np.array(points_int)
	t_bar = np.array(t_bar)

	if return_points:
		return t_bar, poly_access, points_int
	else:
		return t_bar



def get_k_proj(LAN0, LAN_dot, Tn, num_days, K, n_hat):
	dT = -2*np.pi / (LAN_dot - W_EARTH)
	num_orb = int(np.ceil(num_days*dT / Tn))
	k_range = np.arange(0,num_orb+2)
	px0 = np.degrees(LAN0 + 2*np.pi*k_range/K)
	proj = px0*n_hat[0]
	return k_range, proj


def get_regions(poly, num_days, n_hat):
	dx = 360 * n_hat[0]
	xg1, xg2 = [], []
	reg = []
	idx_rev = []
	for poly0 in poly:
		x = np.dot(poly0,n_hat)
		x1, x2 = np.min(x), np.max(x)
		xg1 = x1 - np.arange(num_days+1)*dx
		xg2 = x2 - np.arange(num_days+1)*dx
		reg.append(np.transpose([xg1,xg2]))
	#
	reg = np.vstack(reg)
	return reg
	

def format_poly(poly1, poly2, complete, num):

	from leocat.utils.math import unwrap

	if num > 2:
		poly1.T[0] = unwrap(poly1.T[0])
		poly1.T[1] = unwrap(poly1.T[1])
		poly2.T[0] = unwrap(poly2.T[0])
		poly2.T[1] = unwrap(poly2.T[1])
		poly1 = np.vstack((poly1,poly1[0]))
		poly2 = np.vstack((poly2,poly2[0]))

		poly = []
		if not complete:
			x_hat = np.array([1,0])
			y_hat = np.array([0,1])
			pt_mid = (np.mean(poly1[:-1],axis=0) + np.mean(poly2[:-1],axis=0)) / 2.0
			dr1 = poly1[:-1] - pt_mid
			dr2 = poly2[:-1] - pt_mid

			theta1 = np.arctan2(np.dot(dr1,y_hat),np.dot(dr1,x_hat))
			theta2 = np.arctan2(np.dot(dr2,y_hat),np.dot(dr2,x_hat))
			theta = np.concatenate((theta1,theta2)) % (2*np.pi)
			idx_rot = np.argsort(theta)

			poly3 = np.vstack((poly1[:-1], poly2[:-1]))
			poly3 = poly3[idx_rot]
			poly3 = np.vstack((poly3, poly3[0]))
			poly.append(poly3)

			# fig, ax = make_fig()
			# ax.plot(dr1.T[0], dr1.T[1])
			# ax.plot(dr2.T[0], dr2.T[1])
			# fig.show()

			# fig, ax = make_fig()
			# ax.plot(poly1.T[0], poly1.T[1])
			# ax.plot(poly2.T[0], poly2.T[1])
			# ax.plot(pt_mid[0], pt_mid[1], 'r.')
			# ax.plot(poly3.T[0], poly3.T[1], 'k--')
			# fig.show()

			# sys.exit()

			# poly3 = np.vstack((np.flipud(poly1[:-1]), poly2[:-1]))
			# poly3 = np.vstack((poly3, poly3[0]))
			# poly3.T[0] = unwrap(poly3.T[0])
			# poly3.T[1] = unwrap(poly3.T[1])
			# poly.append(poly3)

		else:
			poly.append(poly1)
			poly.append(poly2)

	else:
		# single line
		# poly1 = np.vstack(poly1)
		# poly2 = np.vstack(poly2)
		poly = []
		if not complete:
			poly3 = np.vstack((np.flipud(poly1), poly2))
			poly3.T[0] = unwrap(poly3.T[0])
			poly3.T[1] = unwrap(poly3.T[1])
			poly.append(poly3)
		else:
			# poly1.T[0] = unwrap(poly1.T[0])
			# poly1.T[1] = unwrap(poly1.T[1])
			# poly2.T[0] = unwrap(poly2.T[0])
			# poly2.T[1] = unwrap(poly2.T[1])
			poly.append(poly1)
			poly.append(poly2)

	#
	return poly
	


@njit
def get_orbit_poly_indices(proj, reg, k_range):
	idx = []
	idx_poly = []
	for kk,proj0 in enumerate(proj):
		for jj,reg0 in enumerate(reg):
			if reg0[0] < proj0 < reg0[1]:
				idx.append(k_range[kk])
				idx_poly.append(jj)
				break
	#
	idx = np.array(idx)
	idx_poly = np.array(idx_poly)
	return idx, idx_poly

@njit
def get_lines(idx, LAN0, K):
	px1 = (LAN0 + 2*np.pi*(idx-1)/K) * 180.0/np.pi
	px2 = (LAN0 + 2*np.pi*(idx+2)/K) * 180.0/np.pi
	lines = np.full((len(idx),2,2),np.nan)
	for j in range(len(idx)):
		p1 = np.array([px1[j], -360.0])
		p2 = np.array([px2[j], 2*360.0])
		lines[j,0] = p1
		lines[j,1] = p2
	return lines


@njit
def get_poly_maps(kk, u1_vec, u2_vec, LAN_G1_vec, LAN_G2_vec):
	poly1 = []
	poly2 = []
	count = 0
	for j in range(len(u1_vec)):
		u1 = u1_vec[j]
		u2 = u2_vec[j]
		LAN_G1 = LAN_G1_vec[j][kk]
		LAN_G2 = LAN_G2_vec[j][kk]

		if np.isnan(u1) or np.isnan(u2) or np.isnan(LAN_G1) or np.isnan(LAN_G2):
			continue

		u1_deg = u1 * 180.0/np.pi
		u2_deg = u2 * 180.0/np.pi
		LAN_G1_deg = LAN_G1 * 180.0/np.pi
		LAN_G2_deg = LAN_G2 * 180.0/np.pi

		poly1.append([LAN_G1_deg, u1_deg])
		poly2.append([LAN_G2_deg, u2_deg])

		count += 1

	return poly1, poly2, count


@njit
def get_u_LAN_vec(lam_vec, phi, AZ_range, rho, EL, lower, upper, a, inc_rad):
	u1_vec = np.full(AZ_range.shape, np.nan)
	u2_vec = np.full(AZ_range.shape, np.nan)
	LAN_G1_vec = np.full((len(AZ_range),len(lam_vec)), np.nan)
	LAN_G2_vec = np.full((len(AZ_range),len(lam_vec)), np.nan)
	for j in range(len(AZ_range)):

		AZ = AZ_range[j]

		val = rho*np.sin(AZ)*np.cos(EL)
		mid = -val
		if val == 0:
			mid = val

		if not (lower < mid < upper):
			# not observed
			continue

		phi0 = np.arcsin((-rho * np.sin(AZ) * np.cos(EL)) / R_earth)
		lam0 = np.arctan((rho*np.cos(AZ)*np.cos(EL))/(a - rho*np.sin(EL)))

		psi = (np.sin(phi) - np.cos(inc_rad)*np.sin(phi0)) / (np.sin(inc_rad)*np.cos(phi0))
		theta = (np.sin(phi0) - np.cos(inc_rad)*np.sin(phi))/(np.sin(inc_rad)*np.cos(phi))

		gamma1 = np.arcsin(psi)
		gamma2 = np.pi - np.arcsin(psi)

		LAN1 = np.arcsin(theta)
		LAN2 = np.pi - np.arcsin(theta)

		check1 = np.abs( np.cos(gamma1)*np.cos(phi0) - np.cos(LAN1)*np.cos(phi) )
		check2 = np.abs( np.cos(gamma2)*np.cos(phi0) - np.cos(LAN2)*np.cos(phi) )
		if check1 > 1e-8 or check2 > 1e-8:
			LAN1, LAN2 = LAN2, LAN1

		u1 = gamma1 - lam0
		u2 = gamma2 - lam0

		LAN_G1 = LAN1 + lam_vec
		LAN_G2 = LAN2 + lam_vec

		u1 = u1 % (2*np.pi)
		u2 = u2 % (2*np.pi)
		LAN_G1 = LAN_G1 % (2*np.pi)
		LAN_G2 = LAN_G2 % (2*np.pi)

		u1_vec[j] = u1
		u2_vec[j] = u2
		LAN_G1_vec[j][:] = LAN_G1
		LAN_G2_vec[j][:] = LAN_G2

	return u1_vec, u2_vec, LAN_G1_vec, LAN_G2_vec


