

import numpy as np
from leocat.utils.const import *
from leocat.utils.cov import get_access_interval, get_t_access_avg, swath_to_FOV
from leocat.utils.cov import get_num_obs, get_revisit
from pyproj import CRS, Transformer

from leocat.utils.geodesy import lla_to_ecf, RADEC_to_cart, DiscreteGlobalGrid
from leocat.utils.orbit import convert_ECI_ECF
from leocat.utils.math import unit, dot, mag
from tqdm import tqdm

from scipy.interpolate import CubicSpline
from numba import njit

@njit
def get_roots_mid(roots):
	roots_mid = []
	for k in range(1,len(roots)):
		k1 = k-1
		k2 = k
		roots_mid.append((roots[k1]+roots[k2])/2.0)
	roots_mid = np.array(roots_mid)
	return roots_mid

class SplineCoverage:
	def __init__(self, orb, swath, lon, lat, JD1, JD2):
		self.orb = orb
		self.swath = swath
		self.lon, self.lat = lon, lat
		self.JD1, self.JD2 = JD1, JD2
		self.num_pts = len(lon)
		self.dt = None

	def get_access(self, num_intp=5, verbose=1, \
						warn=True, spherical=False, debug=0):
		#
		# if solar_band is not None:
		t_access_avg = self.compute_coverage(num_intp, verbose, warn, spherical, debug)
		# else:
		# t, access = self.compute_coverage_fast(dt, f, verbose, warn, spherical)
		# access_interval = get_access_interval(access)
		# t_access_avg = get_t_access_avg(t, access_interval)
		return t_access_avg

	def get_num_obs(self, t_access_avg, GL=None):
		num_obs = get_num_obs(t_access_avg, self.num_pts)
		return num_obs

	def get_revisit(self, t_access_avg, revisit_type='avg', GL=None):
		revisit = get_revisit(t_access_avg, self.num_pts, revisit_type=revisit_type)
		return revisit

	def compute_coverage(self, num_intp=5, verbose=1, \
						warn=True, spherical=False, debug=0):

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
		
		Tn = orb.get_period('nodal')
		dT = Tn/num_intp
		sim_period = (JD2-JD1)*86400

		N = int(sim_period/dT) + 2
		t = np.linspace(0,(JD2-JD1)*86400,N)
		JD = t/86400 + JD1

		r_eci_sc, v_eci_sc = orb.propagate(t)
		r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD, r_eci_sc, v_eci_sc)

		if debug:

			from leocat.utils.plot import make_fig
			from leocat.utils.general import pause

			dT_hi = 1.0 # sec
			N_hi = int(sim_period/dT_hi) + 2
			t_hi = np.linspace(0,sim_period,N_hi)
			JD_hi = t_hi/86400 + JD1
			r_eci_sc_hi, v_eci_sc_hi = orb.propagate(t_hi)
			r_ecf_sc_hi, v_ecf_sc_hi = convert_ECI_ECF(JD_hi, r_eci_sc_hi, v_eci_sc_hi)
			r_hat_hi = unit(r_ecf_sc_hi)


		p_lla = np.transpose([lon,lat,np.zeros(lon.shape)])
		# p = RADEC_to_cart(p_lla.T[0], p_lla.T[1]) * R_earth
		if spherical:
			p = RADEC_to_cart(p_lla.T[0], p_lla.T[1]) * R_earth
		else:
			p = lla_to_ecf(p_lla.T[0], p_lla.T[1], p_lla.T[2])
		p_hat = unit(p)

		# alt = a - R_earth
		# FOV = swath_to_FOV(swath, alt)
		r_hat = unit(r_ecf_sc)

		alpha = swath / (2*R_earth)
		thres = np.cos(alpha)

		iterator = range(len(lon))
		if verbose:
			iterator = tqdm(iterator)

		t_access_avg = {}
		for j in iterator:
			p_hat0 = p_hat[j]
			p0 = p[j]
			proj_alpha = p_hat0[0]*r_hat.T[0] + p_hat0[1]*r_hat.T[1] + p_hat0[2]*r_hat.T[2]
			proj_alpha_adj = proj_alpha - thres

			if debug:
				proj_alpha_hi = p_hat0[0]*r_hat_hi.T[0] + p_hat0[1]*r_hat_hi.T[1] + p_hat0[2]*r_hat_hi.T[2]
				proj_alpha_hi_adj = proj_alpha_hi - thres

			cs = CubicSpline(t, proj_alpha_adj)
			roots = cs.roots()
			b = (0.0 <= roots) & (roots <= sim_period)
			roots = roots[b]
			if len(roots) <= 1:
				continue

			roots_mid = get_roots_mid(roots)
			idx = np.where(cs(roots_mid) > 0)[0]
			reg_idx = np.transpose([idx, idx+1])
			t_bar = np.mean(roots[reg_idx],axis=1)
			t_access_avg[j] = t_bar

			if debug:
				fig, ax = make_fig()
				ax.plot(t_hi, proj_alpha_hi_adj)
				ax.plot(t, proj_alpha_adj, '.')
				ax.plot(t_hi, cs(t_hi), '--')
				xlim = ax.get_xlim()
				ax.plot(xlim, [0,0], 'k--')
				ax.plot(roots, np.zeros(roots.shape), 'r.')
				# ax.plot(roots_mid, np.zeros(roots_mid.shape), 'g.')
				fig.show()

				pause()

		return t_access_avg

