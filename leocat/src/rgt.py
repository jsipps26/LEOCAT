
import numpy as np
from leocat.utils.const import *
from leocat.utils.orbit import Orbit, MLST_to_LAN, get_LAN_dot, convert_ECI_ECF
from leocat.utils.geodesy import ecf_to_lla

class RepeatGroundTrack:
	def __init__(self, D, R, propagator='SPE+frozen', warn=True):
		self.D = D
		self.R = R
		if warn:
			self.warn_degenerate()
		self.propagator_options = ['kepler','SPE','SPE+frozen']
		self.set_propagator(propagator)
		self.set_IK()

	def set_IK(self):
		# R/D = I + K/D
		D, R = self.D, self.R
		K = R % D
		I = (R-K)//D
		self.I = I
		self.K = K
		self.Q = I + K/D

	def set_propagator(self, propagator):
		propagator = propagator.lower()
		if 'spe' in propagator:
			propagator = propagator.replace('spe','SPE')
		if not (propagator in self.propagator_options):
			raise Exception(f'propagator \'{propagator}\' invald, must choose any of the following: ' \
							+ str(self.propagator_options))
			#
		self.propagator = propagator

	def warn_degenerate(self, exit=False):
		D, R = self.D, self.R
		from math import gcd
		import warnings
		g = gcd(R,D)
		R_new = R//g
		D_new = D//g
		if g > 1:
			warnings.warn('%d/%d is degenerate (can be represented as %d/%d)' % (R,D,R_new,D_new))
			if exit:
				import sys
				sys.exit()

	def get_a(self, inc, e=0.0, max_iter=10, tol=1e-12):

		"""
		Vallado v4 ch. 11 algorithm 71
			accurate -assuming- R_ECI_ECF matches W_EARTH

		"""
		propagator = self.propagator
		propagator_type = {'kepler': 1, 'SPE': 2, 'SPE+frozen': 3}
		prop_num = propagator_type[propagator]

		k_rev2rep = self.R
		k_day2rep = self.D

		k_revpday = 1.0 * k_rev2rep / k_day2rep
		n = k_revpday * W_EARTH
		a_est = (MU * (1/n)**2)**(1/3)
		# print(a_est)

		j = 0
		while j < max_iter:
			p = a_est*(1-e**2)
			if prop_num == 1:
				# kepler
				LAN_dot0 = 0.0
				omega_dot0 = 0.0
				M0_dot0 = 0.0
			elif prop_num == 2:
				# SPE
				LAN_dot0 = -3*n*J2/2 * (R_earth/p)**2 * np.cos(inc)
				omega_dot0 = 3*n*J2/4 * (R_earth/p)**2 * (4 - 5*np.sin(inc)**2)
				M0_dot0 = 3*n*J2/4 * (R_earth/p)**2 * np.sqrt(1-e**2) * (2 - 3*np.sin(inc)**2)
			elif prop_num == 3:
				# SPE+frozen
				LAN_dot0 = -3*n*J2/2 * (R_earth/p)**2 * np.cos(inc)
				omega_dot0 = 0.0
				M0_dot0 = 3*n*J2/4 * (R_earth/p)**2 * np.sqrt(1-e**2) * (2 - 3*np.sin(inc)**2)

			n = k_revpday * (W_EARTH - LAN_dot0) - (M0_dot0 + omega_dot0)
			a_est_new = (MU*(1/n)**2)**(1/3)
			# print(a_est)
			if np.abs(a_est_new-a_est) < tol:
				break
			a_est = a_est_new

			j += 1

		return a_est



	def get_sso(self, e=0.0, max_iter=10, tol=1e-12):

		inc_est = np.radians(98.2) # initial guess
		LAN_dot = LAN_dot_SSO

		i = 0
		while i < max_iter:
			a_est = self.get_a(inc_est, e)
			# ut.pause()

			arg1 = -2*a_est**(7/2) * LAN_dot * (1-e**2)**2
			arg2 = 3*R_earth**2 * J2 * np.sqrt(MU)
			inc_est_new = np.arccos(arg1 / arg2)

			if np.abs(inc_est_new - inc_est) < tol:
				break

			i += 1
			inc_est = inc_est_new

		return a_est, inc_est


	def get_sso_LAN(self, MLST, JD, e=0.0, direction='ascending'):
		if direction == 'descending':
			MLST = (MLST + 12) % 24
		LAN0 = MLST_to_LAN(MLST, JD)
		return LAN0


	def get_nodal_day(self, a, inc, e=0.0):
		propagator = self.propagator
		LAN_dot = 0.0
		if 'SPE' in propagator:
			LAN_dot = get_LAN_dot(a, e, inc)
		Dn = 2*np.pi / (W_EARTH - LAN_dot) # sec
		return Dn


	def get_track_dist(self, return_deg=False):
		D, R = self.D, self.R
		dlam_main = 2*np.pi*D/R
		dlam_sub = 2*np.pi/R
		if return_deg:
			return np.degrees(dlam_main), np.degrees(dlam_sub)
		d_main = dlam_main * R_earth
		d_sub = dlam_sub * R_earth
		return d_main, d_sub


	def get_subcycles(self, return_k_max=False, return_inverse=True):
		"""
		Use RGT sub-cycles to determine max revisit time given wide
		swath width
			CAVEATS: RT only applies for SSO at ascending or descending node
			Luo, X., Wang, M., Dai, G., and Chen, X., "A Novel Technique 
			to Compute the Revisit Time of Satellites and Its Application 
			in Remote Sensing Satellite Optimization Design," International 
			Journal of Aerospace Engineering, Vol. 2017, 2017, pp. 1–9. 
			https://doi.org/10.1155/2017/6469439

		"""
		D, R = self.D, self.R
		I, K = self.I, self.K
		Q = I + K/D

		dk = K
		if K > D/2:
			dk = K-D

		sub_cycles = {0: 0}
		num = 0
		k_rel_abs_max = 0
		for d in range(1,D):
			if dk < 0:
				# westward
				k = d*dk + num*D
				if k < -D:
					k = k + D
					num += 1
				k_rel = k
				if k_rel < -D/2:
					k_rel = k_rel + D

			elif dk > 0:
				# eastward
				k = d*dk - num*D
				if k > D:
					k = k - D
					num += 1
				k_rel = k
				if k_rel > D/2:
					k_rel = k_rel - D

			if return_inverse:
				sub_cycles[d] = k_rel # f(days) = offset
			else:
				sub_cycles[k_rel] = d # f(offset) = days

			k_rel_abs = np.abs(k_rel)
			if k_rel_abs > k_rel_abs_max:
				k_rel_abs_max = k_rel_abs

		if return_k_max:
			return sub_cycles, k_rel_abs_max
		else:
			return sub_cycles


	def get_RT(self, inc, w=None, return_min=False):

		"""
		CAVEATS: RT only applies for SSO at ascending or descending node
			i.e. daytime revisits for an SSO

		Given a swath smaller than the distance between adjacent
		tracks, the maximum revisit time is trivially equal to
		the number of days to repeat the cycle. However, if the
		swath width > distance between adjacent tracks, the swath
		can overlap other parts of the RGT, consistently, in less
		time than the number of days to repeat.
			Luo, X., Wang, M., Dai, G., and Chen, X., "A Novel Technique 
			to Compute the Revisit Time of Satellites and Its Application 
			in Remote Sensing Satellite Optimization Design," International 
			Journal of Aerospace Engineering, Vol. 2017, 2017, pp. 1–9. 
			https://doi.org/10.1155/2017/6469439

		Assumptions
			inclination > 0 (non-equatorial)
			partial overlap of other tracks is not considered a repeat
				i.e., swath must be at least 2x the distance between
				adjacent tracks to obtain an advantage in repeats

		"""

		D, R = self.D, self.R
		I, K = self.I, self.K
		inc_deg = np.degrees(inc)

		sub_cycles, k_max = self.get_subcycles(return_k_max=True, return_inverse=False)
		Q = I + K/D

		# dlon_rev, dlon_pass = get_repeat_dlon(D, R, deg=True)
		# dlon_rev, dlon_pass = self.get_track_dist(return_deg=False)
		# Sq = dlon_rev
		# Sq_km = dlon_rev*np.pi/180 * R_earth
		# Sd = dlon_pass
		# Sd_km = dlon_pass*np.pi/180 * R_earth
		Sq_km, Sd_km = self.get_track_dist()

		inc = np.radians(inc_deg)
		inc_app = np.arctan2(np.sin(inc),np.cos(inc)-1/Q)
		inc_app_deg = np.degrees(inc_app)
		# print(inc_app_deg)

		if w is None:
			w = Sd_km # distance between adjacent tracks, km
		w_app = w / np.sin(inc_app)
		# print(w_app)
		if w_app >= 2*Sd_km:
			n = int(w_app/(2*Sd_km))
			n_range = np.arange(-n,n+1)
			b = np.abs(n_range) <= k_max
			n_range = n_range[b]
			S = []
			for k in n_range:
				if not (k in sub_cycles):
					continue
				d = sub_cycles[k]
				S.append(d)
			S = np.array(S)
			Sp = np.sort(S)
			# print('S', S)
			# print('Sp', Sp)

			RT_max = np.max(np.diff(Sp))
			RT_min = np.min(np.diff(Sp))

		else:
			RT_max = D
			RT_min = D

		if not return_min:
			return RT_max
		else:
			return RT_max, RT_min


	def plot_subcycles(self, a, inc, e=0.0, iterate=True):

		# import matplotlib.pyplot as plt
		from leocat.utils.general import pause
		from leocat.utils.time import date_to_jd
		from leocat.utils.plot import plt, split_lon

		JD1 = date_to_jd(2024,1,1)

		D, R = self.D, self.R

		propagator = self.propagator
		sub_cycles = self.get_subcycles()

		Sq, Sd = self.get_track_dist(return_deg=True)
		Sq_km, Sd_km = self.get_track_dist()

		orb = Orbit(a,0,inc,0,np.radians(270),np.radians(90),propagator=propagator)
		Dn = orb.get_nodal_day()
		Tn = orb.get_period('nodal')

		# fig, ax = make_fig()

		d = []
		for k in sub_cycles:
			d.append(Sd_km*sub_cycles[k])
		d.append(d[0])
		d = np.abs(np.array(d))
		D_range = np.arange(1,len(d)+1)

		fig = plt.figure(figsize=(10,4))

		ax2 = fig.add_subplot(1,2,2)
		ax2.plot(D_range, d)
		ax2.set_xlabel('Nodal Day')
		ax2.set_ylabel('Shift (km)')
		ax3 = ax2.twinx()
		ax3.plot(D_range, d/R_earth*180/np.pi, c='w', lw=0.1)
		ax3.set_ylabel('Shift (deg)')
		ax2.set_title('Distance from Initial Asc/Desc Node')
		ax2.grid()

		ax1 = fig.add_subplot(1,2,1)
		ax1.grid()
		ax1.set_xlabel('Longitude (deg)')
		ax1.set_ylabel('Latitude (deg)')
		ax1.set_title('Ground Track Geometry')


		lon0 = None
		xlim = None
		ylim = [-10,10]
		lon_cycle = []
		lat_cycle = []
		for k in sub_cycles:
			tk = k*Dn
			t = orb.t0 + np.linspace(0,Tn,100) + tk
			r, v = orb.propagate(t)
			JD = JD1 + t/86400
			r_ecf = convert_ECI_ECF(JD,r)
			lon, lat, alt = ecf_to_lla(r_ecf.T[0], r_ecf.T[1], r_ecf.T[2])
			if lon0 is None:
				lon0 = lon[0]
			lon = lon-lon0

			if xlim is None:
				xlim = np.sort([lon[0], lon[-1]])
				xlim[0] = xlim[0] - Sd
				xlim[1] = xlim[1] + Sd

			lon_cycle.append(lon)
			lat_cycle.append(lat)
			idx_split = split_lon(lon)
			for idx in idx_split:
				ax1.plot(lon[idx], lat[idx], c='k', alpha=0.1)

		ax1.set_xlim(xlim)
		ax1.set_ylim(ylim)

		for i in range(len(lon_cycle)):
			lon = lon_cycle[i]
			lat = lat_cycle[i]
			idx_split = split_lon(lon)
			im = None
			for idx in idx_split:
				if im is None:
					im = ax1.plot(lon[idx], lat[idx], lw=2.0)
				else:
					ax1.plot(lon[idx], lat[idx], c=im[0].get_c(), lw=2.0)

			title = f'{D}/{R}, interval {i+1}/{D}'

			ax2.plot(D_range[i], d[i], '.', c=im[0].get_c())

			fig.suptitle(title)

			fig.canvas.draw()
			fig.canvas.flush_events()
			fig.show()

			if iterate:
				pause()


