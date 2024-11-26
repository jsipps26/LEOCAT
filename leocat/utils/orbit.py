

import numpy as np
from leocat.utils.const import *
from leocat.utils.geodesy import lla_to_ecf, ecf_to_lla
from leocat.utils.math import matmul, newton_raphson, unit, wrap, mag
from leocat.utils.math import R1, R2, R3

"""
References

Common orbital mechanics functions
UT Celestial Mechanics I (Ryan Russell)

Celestial phenomena
Vallado, D., "Fundamentals of Astrodynamics and Applications", version 4

Other sources used to validate RGT design
Luo et. al. "A Novel Technique to Compute the Revisit Time of Satellites and Its Application in Remote Sensing Satellite Optimization Design"
Pie, N. "Mission Design Concepts for Repeat Groundtrack Orbits and Application to the ICESat Mission"

Some parts of RepeatGroundTrack class
	Luo, X., Wang, M., Dai, G., and Chen, X., "A Novel Technique 
	to Compute the Revisit Time of Satellites and Its Application 
	in Remote Sensing Satellite Optimization Design," International 
	Journal of Aerospace Engineering, Vol. 2017, 2017, pp. 1–9. 
	https://doi.org/10.1155/2017/6469439


"""

def LEO(alt, inc, e=0.0, LAN=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	a = R_earth + alt
	inc = np.radians(inc)
	LAN = np.radians(LAN)
	omega = np.radians(omega)
	nu = np.radians(nu)
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator,warn=warn)
	return orb

def LEO_SSO(alt, MLST, JD, direction='ascending', e=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	"""
	Technically, the arg of latitude must be zero if direction is 'ascending',
	and u should be 180 deg. if direction is 'descending'. However, the change
	in LAN across a revolution is <0.05 deg., which does not significantly 
	change the true equatorial crossing time.

	If you wanted to compensate, you must find TOF to the ascending/descending
	node given u = omega + nu, then subtract/add TOF to JD1, which specifies
	the LAN at which the satellite crosses the ascending/descending node.

	"""
	if not (propagator == 'SPE' or propagator == 'SPE+frozen'):
		raise Exception('SSO requires J2 perturbation, propagator must be one of: [SPE, SPE+frozen]')
	mu = MU
	J2 = 0.00108248
	a = R_earth + alt
	arg1 = -2*a**(7/2) * LAN_dot_SSO * (1-e**2)**2
	arg2 = 3*R_earth**2 * J2 * np.sqrt(mu)
	inc = np.arccos(arg1/arg2)

	if direction == 'ascending':
		LTAN = MLST % 24
		LTDN = (MLST + 12) % 24
	elif direction == 'descending':
		LTAN = (MLST + 12) % 24
		LTDN = MLST % 24

	if direction == 'descending':
		MLST = (MLST + 12) % 24
	LAN = MLST_to_LAN(MLST, JD)

	omega = np.radians(omega)
	nu = np.radians(nu)
	metadata = {'LTAN': LTAN, 'LTDN': LTDN, 'direction': direction, 'JD': JD}
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator, metadata=metadata, warn=warn)
	return orb


def LEO_RGT(D, R, inc, e=0.0, LAN=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	RGT = RepeatGroundTrack(D,R,propagator=propagator)
	inc = np.radians(inc)
	a = RGT.get_a(inc, e=e)
	LAN = np.radians(LAN)
	omega = np.radians(omega)
	nu = np.radians(nu)
	metadata = {'D': D, 'R': R}
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator, metadata=metadata, warn=warn)
	return orb

def LEO_RGT_SSO(D, R, MLST, JD, direction='ascending', e=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	if not (propagator == 'SPE' or propagator == 'SPE+frozen'):
		raise Exception('SSO requires J2 perturbation, propagator must be one of: [SPE, SPE+frozen]')
	RGT = RepeatGroundTrack(D,R,propagator=propagator)
	a, inc = RGT.get_sso(e=e)
	LAN = RGT.get_sso_LAN(MLST,JD,e=e,direction=direction)

	if direction == 'ascending':
		LTAN = MLST
		LTDN = (MLST + 12) % 24
	elif direction == 'descending':
		LTAN = (MLST + 12) % 24
		LTDN = MLST

	omega = np.radians(omega)
	nu = np.radians(nu)
	metadata = {'D': D, 'R': R, 'LTAN': LTAN, 'LTDN': LTDN, 'direction': direction, 'JD': JD}
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator, metadata=metadata, warn=warn)
	return orb

def LEO_SSO_RGT(D, R, MLST, JD, direction='ascending', e=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	return LEO_RGT_SSO(D, R, MLST, JD, direction=direction, e=e, omega=omega, nu=nu, propagator=propagator, warn=warn)


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
			from leocat.utils.orbit import get_LAN_dot
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




def GMAT_UTC_to_JD(GMAT_UTC):
	# 2000-01-01T11:59:28.000
	from leocat.utils.time import date_to_jd
	parse = GMAT_UTC.split('-')
	year = int(parse[0])
	month = int(parse[1])

	day_hms = parse[2].split('T')
	day = int(day_hms[0])

	hms = day_hms[1].split(':')
	hour = int(hms[0])
	minute = int(hms[1])
	sec = float(hms[2])

	JD = date_to_jd(year, month, day + hour/24 + minute/(24*60) + sec/(24*60*60))
	return JD


def read_GMAT_oem(file, num_lines=None):

	UTC = []
	JD = []
	r_eci_sc = []
	v_eci_sc = []
	record = 0
	num = 0
	with open(file,'r') as fp:
		for line in fp:
			line = line.split()
			if len(line) == 0:
				continue
			if line[0] == 'META_STOP':
				record = 1
				continue
			if record:
				if not (len(line) == 7):
					print('line',line)
					raise Exception('len(line) != 7')
					# input('enter to continue')
					# continue

				# UTC.append(line[0])
				GMAT_UTC = line[0]
				UTC.append(GMAT_UTC)
				JD.append(GMAT_UTC_to_JD(GMAT_UTC))
				rx, ry, rz = [float(val) for val in line[1:4]]
				vx, vy, vz = [float(val) for val in line[4:7]]
				r_eci_sc.append([rx,ry,rz])
				v_eci_sc.append([vx,vy,vz])

				num += 1
				if not (num_lines is None):
					if num == num_lines:
						break

	JD = np.array(JD)
	r_eci_sc = np.array(r_eci_sc)
	v_eci_sc = np.array(v_eci_sc)

	return JD, r_eci_sc, v_eci_sc


def read_STK_eph(file, JD_epoch, unit='km'):

	t = []
	t_hold = -1
	rx, ry, rz = [], [], []
	vx, vy, vz = [], [], []
	switch = False
	with open(file,'r') as fp:
		for i,line in enumerate(fp):
			line = line.split()
			if len(line) == 0:
				continue
			if 'EphemerisTimePosVel' in line:
				switch = True
				continue
			if switch:
				try:
					t0, rx0, ry0, rz0, vx0, vy0, vz0 = \
						[float(val) for val in line]
					#
				except ValueError:
					continue
				
				if t0 - t_hold > 0:
					t.append(t0)
					rx.append(rx0)
					ry.append(ry0)
					rz.append(rz0)
					vx.append(vx0)
					vy.append(vy0)
					vz.append(vz0)
					t_hold = t0
				else:
					continue


	t = np.array(t)
	rx = np.array(rx)
	ry = np.array(ry)
	rz = np.array(rz)
	vx = np.array(vx)
	vy = np.array(vy)
	vz = np.array(vz)

	JD = t/86400 + JD_epoch

	r_eci_sc = np.transpose([rx,ry,rz])
	v_eci_sc = np.transpose([vx,vy,vz])

	if unit == 'km':
		r_eci_sc = r_eci_sc/1e3
		v_eci_sc = v_eci_sc/1e3

	return JD, r_eci_sc, v_eci_sc



class Orbit:
	def __init__(self, *args, \
					t0=0.0, propagator='kepler', mu=398600.4418, J2=0.00108248,
					prop_tol=2.220446049250313e-14, warn=True, metadata={}):
		#
		self.file = None
		self.file_exists = False
		self.t_file = None
		self.r_file = None
		self.v_file = None
		if len(args) == 1:
			file = os.path.abspath(args[0])
			if file.endswith('.oem'):
				# assuming epoch is at start of file, JD[0]
				JD, r_eci_sc, v_eci_sc = read_GMAT_oem(file)
				r, v = r_eci_sc[0], v_eci_sc[0]
				t_file = (JD-JD[0])*86400 + t0
				if (np.diff(t_file) <= 0).any():
					raise ValueError(f'{file} time must be strictly increasing')
				self.t_file = t_file
				self.r_file = r_eci_sc
				self.v_file = v_eci_sc

			elif file.endswith('.e'):
				# default to setting epoch to zero
				JD, r_eci_sc, v_eci_sc = read_STK_eph(file, 0.0)
				r, v = r_eci_sc[0], v_eci_sc[0]
				t_file = (JD-JD[0])*86400 + t0
				if (np.diff(t_file) <= 0).any():
					raise ValueError(f'{file} time must be strictly increasing')
				self.t_file = t_file
				self.r_file = r_eci_sc
				self.v_file = v_eci_sc
			else:
				raise NotImplementedError('Orbit can only read .oem files from GMAT')
			OE = RV2OE(r, v, mu=mu)
			a, e, inc, LAN, omega, nu = OE
			self.file = file
			self.file_exists = True

		# elif len(args) == 2:
		elif len(args) == 2:
			r, v = args
			OE = RV2OE(r,v,mu=mu)
			a, e, inc, LAN, omega, nu = OE

		elif len(args) == 6:
			a, e, inc, LAN, omega, nu = args

		else:
			raise Exception('Invalid number of input args. Must create orbit either through ' + \
							'(r,v), (a,e,i,LAN,omega,nu), or external file.')
			#

		self.warn = warn
		self.t0 = t0
		self.mu = mu
		self.J2 = J2
		self.set_OEs(a, e, inc, LAN, omega, nu)
		self.propagator_options = ['kepler','SPE','SPE+frozen','cowell+kepler','cowell+J2']
		self.set_propagator(propagator, prop_tol)

		self.metadata = metadata


	def set_propagator(self, propagator, prop_tol=2.220446049250313e-14):
		propagator = propagator.lower()
		if 'spe' in propagator:
			propagator = propagator.replace('spe','SPE')
		if 'j2' in propagator:
			propagator = propagator.replace('j2','J2')

		if propagator == 'SPE+frozen':
			# check omega
			tol = 1e-10
			# omega = self.omega
			omega = np.arctan2(np.sin(self.omega), np.cos(self.omega))
			if not (np.abs(omega-np.pi/2) < tol or np.abs(omega+np.pi/2) < tol) and self.warn:
				# technically relies on Earth-only orbits
				import warnings
				warnings.warn('orbit is frozen but arg of periapsis is not 90 or 270 deg.')

		if not (propagator in self.propagator_options):
			raise Exception(f'propagator \'{propagator}\' invald, must choose any of the following: ' \
							+ str(self.propagator_options))
			#

		self.propagator = propagator
		self.prop_tol = prop_tol


	def set_M0(self):
		t0 = self.t0
		nu = self.nu
		e = self.e
		E = 2*np.arctan(np.tan(nu/2)/(np.sqrt((1+e)/(1-e))))
		self.M0 = E - e*np.sin(E) # since nu is at t0

	# def _wrap180(angle):
	# 	return np.arctan2(np.sin(angle), np.cos(angle))
	def get_nodal_day(self):
		LAN_dot = self.get_LAN_dot()
		Dn = 2*np.pi / (W_EARTH - LAN_dot) # sec
		return Dn

	def get_LAN_dot(self):
		propagator = self.propagator
		a, e, inc = self.a, self.e, self.inc
		LAN_dot = get_LAN_dot(a, e, inc, mu=self.mu, J2=self.J2)
		if 'kepler' in propagator:
			LAN_dot = 0.0
		return LAN_dot

	def get_omega_dot(self):
		propagator = self.propagator
		a, e, inc = self.a, self.e, self.inc
		omega_dot = get_omega_dot(a, e, inc, mu=self.mu, J2=self.J2)
		if 'kepler' in propagator or 'frozen' in propagator:
			omega_dot = 0.0
		return omega_dot

	def get_M_dot(self):
		propagator = self.propagator
		a, e, inc = self.a, self.e, self.inc
		M_dot = get_M_dot(a, e, inc, mu=self.mu, J2=self.J2)
		if 'kepler' in propagator:
			M_dot = self.n
		return M_dot

	def get_alt(self, radius=R_earth):
		return self.a - radius

	def set_OEs(self, a, e, inc, LAN, omega, nu):
		self.a = a
		self.e = e
		self.inc = inc
		self.LAN = LAN
		self.omega = omega
		self.nu = nu
		self.OE = np.array([a,e,inc,LAN,omega,nu])
		self.n = np.sqrt(self.mu/a**3)
		self.T = 2*np.pi/self.n
		self.set_M0()

		r, v = OE2RV(self.OE, mu=self.mu)
		self.r = r
		self.v = v


	def get_period(self, option=None):

		if option is None:
			if 'kepler' in self.propagator:
				option = 'kepler'
			else:
				option = 'nodal'

		option = option.lower()
		if option == 'kepler':
			return self.T

		elif option == 'nodal' or option == 'draconitic' or option == 'node' or option == 'draconic':

			if self.propagator == 'kepler' or self.propagator == 'cowell+kepler':
				return self.T

			elif self.propagator == 'SPE' or self.propagator == 'cowell+J2':
				# only approximate nodal period
				#	avg over 1 period
				a, e, inc = self.a, self.e, self.inc
				M_dot = get_M_dot(a, e, inc, mu=self.mu, J2=self.J2)
				omega_dot = get_omega_dot(a, e, inc, mu=self.mu, J2=self.J2)
				Tn = 2*np.pi / (M_dot + omega_dot)
				return Tn

			elif self.propagator == 'SPE+frozen':
				a, e, inc = self.a, self.e, self.inc
				M_dot = get_M_dot(a, e, inc, mu=self.mu, J2=self.J2)
				omega_dot = 0.0
				Tn = 2*np.pi / (M_dot + omega_dot)
				return Tn

		elif option == 'anomalistic' or option == 'anomaly':
			if self.propagator == 'kepler' or self.propagator == 'cowell+kepler':
				return self.T
			elif self.propagator == 'SPE' or self.propagator == 'cowell+J2':
				a, e, inc = self.a, self.e, self.inc
				M_dot = get_M_dot(a, e, inc, mu=self.mu, J2=self.J2)
				Ta = 2*np.pi/M_dot
				return Ta
			elif self.propagator == 'SPE+frozen':
				a, e, inc = self.a, self.e, self.inc
				M_dot = get_M_dot(a, e, inc, mu=self.mu, J2=self.J2)
				Ta = 2*np.pi/M_dot
				return Ta


	def get_tn_kepler(self, k=0):
		# Time of kth nodal crossing for keplerian orbit
		# 	k = 0 -> closest crossing (+ or -)
		#	k > 0 -> next crossings
		#	k < 0 -> previous crossings

		n = self.n
		u0 = (self.omega + self.nu) % (2*np.pi)
		M0 = self.M0
		M_node = nu2M(-self.omega,self.e)
		t0 = self.t0
		dM = (M_node - M0) % (2*np.pi)
		if u0 <= np.pi:
			dt = (dM-2*np.pi)/n
		else:
			dt = dM/n
		tn0 = t0 + dt

		T = self.T
		tn = tn0
		if dt < 0.0:
			if k > 0:
				tn = tn0 + k*T
			elif k < 0:
				tn = tn0 + (k+1)*T

		elif dt > 0.0:
			if k > 0:
				tn = tn0 + (k-1)*T
			elif k < 0:
				tn = tn0 + k*T

		return tn


	def get_tn(self, k=0):
		"""
		Perturbed node
		Approx time of closest nodal crossing for any orbit 
			best if e less than ~0.9
			less accurate with numerical and high eccentricity

		Can fail if k is very large as tn0 isnt in the right
		nodal orbit
			k like 10s-100s
			depends on how elliptical orbit is
			can loop over get_tn and ensure that every successive
			k has tn greater than the last, then it will be correct

		"""

		t0 = self.t0
		tn0 = self.get_tn_kepler(k)
		if 'kepler' in self.propagator:
			return tn0

		a, e, inc = self.a, self.e, self.inc
		LAN0 = self.LAN
		LAN_dot = get_LAN_dot(a,e,inc,mu=self.mu,J2=self.J2)
		# if 'kepler' in self.propagator:
		# 	LAN_dot = 0.0
		i_hat = np.array([1,0,0])
		j_hat = np.array([0,1,0])

		"""
		Theory
		dot(r,n_hat) = r*cos(angle)
		when angle is zero, r_hat == n_hat
		-> dot(r,n_hat) = r

		Newton's method to solve
		f(t) = dot(r(t),n_hat(t0)) - r_mag(t) is max
		when t = tn
		-> maximize f(t)
		Take derivative of f and set to zero, 
		solve for t
			approx n_hat as SPE, n_hat_dot is simple
			otherwise, need n_hat from r,v, which will
			fail with singularities

		should get within a few 100s km for cowell+J2 and high e
		"""
		from scipy.optimize import newton
		def func_prime(tn):
			r_vec, v_vec = self.propagate(tn)
			LAN = LAN0 + LAN_dot*(tn-t0)
			n_hat = i_hat*np.cos(LAN) + j_hat*np.sin(LAN)
			n_hat_dot = LAN_dot*(-i_hat*np.sin(LAN) + j_hat*np.cos(LAN))
			r_mag_dot = np.dot(r_vec,v_vec)/mag(r_vec)
			return np.dot(v_vec,n_hat) + np.dot(r_vec,n_hat_dot) - r_mag_dot

		try:
			tn = newton(func_prime, tn0)
		except RuntimeError:
			if self.warn:
				import warnings
				warnings.warn('Newton method failed to converge, returning keplerian node')
			return tn0
		return tn


	def propagate(self, t, return_OEs=False):
		"""
		This takes current OEs at epoch and propagates
		them forward in time, given the propagator
			t is scalar or vector
			outputs r,v
		"""
		propagate_method = {'kepler': lambda t: self.propagate_kepler(t, return_OEs=return_OEs),
							'SPE': lambda t: self.propagate_SPE(t, frozen=False, return_OEs=return_OEs),
							'SPE+frozen': lambda t: self.propagate_SPE(t, frozen=True, return_OEs=return_OEs),
							'cowell+kepler': lambda t: self.propagate_cowell(t, pert='kepler', return_OEs=return_OEs),
							'cowell+J2': lambda t: self.propagate_cowell(t, pert='J2', return_OEs=return_OEs)}
		#
		propagator = self.propagator
		propagate_func = propagate_method[propagator]
		# return propagate_func(t)

		if not self.file_exists:
			return propagate_func(t)

		else:
			from scipy.interpolate import CubicSpline
			t_file = self.t_file
			r_file, v_file = self.r_file, self.v_file
			cr = CubicSpline(t_file, r_file)
			cv = CubicSpline(t_file, v_file)
			t_min, t_max = np.min(t_file), np.max(t_file)

			t, scalar = self._format_input_array(t)
			b = (t_min <= t) & (t <= t_max)
			if b.all():
				# interpolate only
				r = cr(t)
				v = cv(t)

			else:
				# intp over b
				# extrap over ~b
				b_left = t < t_min
				b_right = t > t_max
				b_intp = b

				t_left, t_right = t[b_left], t[b_right]
				r, v = [], []

				if b_left.any():
					# print('left')
					r_left, v_left = propagate_func(t_left)
					r.append(r_left)
					v.append(v_left)


				if b_intp.any():
					# print('intp')
					r_intp = cr(t[b_intp])
					v_intp = cv(t[b_intp])
					r.append(r_intp)
					v.append(v_intp)


				if b_right.any():
					# print('right')

					t0_init = float(np.copy(self.t0)) # just in case
					OE_init = np.copy(self.OE)

					OE = RV2OE(r_file[-1],v_file[-1])
					self.t0 = t_max
					self.set_OEs(*OE)
					r_right, v_right = propagate_func(t_right)
					r.append(r_right)
					v.append(v_right)

					# OE = RV2OE(r_file[0],v_file[0])
					# self.t0 = t_min
					# self.set_OEs(*OE)
					self.t0 = t0_init
					self.set_OEs(*OE_init)

				r = np.vstack(r)
				v = np.vstack(v)


			if not return_OEs:
				if scalar:
					return r[0], v[0]
				else:
					return r, v
			else:
				OE_rtn = RV2OE(r,v,mu=self.mu)
				if scalar:
					return OE_rtn[0]
				else:
					return OE_rtn



	def propagate_epoch(self, t0_new, reset_epoch=False):
		"""
		This takes current OEs at epoch and propagates
		them forward in time, -setting- OEs to the new
		epoch
			t0_new is scalar
			no output, just sets OEs to new epoch
		"""
		propagate_method = {'kepler': lambda t: self.propagate_kepler(t, return_OEs=True),
							'SPE': lambda t: self.propagate_SPE(t, frozen=False, return_OEs=True),
							'SPE+frozen': lambda t: self.propagate_SPE(t, frozen=True, return_OEs=True),
							'cowell+kepler': lambda t: self.propagate_cowell(t, pert='kepler', return_OEs=True),
							'cowell+J2': lambda t: self.propagate_cowell(t, pert='J2', return_OEs=True)}
		#
		propagator = self.propagator
		propagate_func = propagate_method[propagator]
		# OE_new = propagate_func(t0_new)
		# self.t0 = t0_new
		# self.set_OEs(*OE_new)

		if not self.file_exists:
			OE_new = propagate_func(t0_new)

		else:
			"""
			propagate_epoch does -not- update t_file
			The reason is because if you update t_file and not r/v_file, you 
			effectively time-shift the file orbit, not propagate it. Alternatively,
			if you change r/v_file, either the start or end will get replaced by
			extrapolation, and it'll lose file data. So what ends up happening is
			the t/r/v_file info is static with respect to the initial t0.

			If you generate the orbit via a file, the propagate_epoch will only
			line up with comparison orbits if you extrapolate the epoch. If you
			interpolate the epoch, the intp OEs will not match a comparison, either
			because the force model from GMAT/etc. is different, or because we're 
			using a spline to intp between points, which produces slight perturbations
			making effectively different orbits at interpolated points. And if you
			set your epoch orbit to an interpolated location, it will be slightly
			off.

			"""
			from scipy.interpolate import CubicSpline
			t_file = self.t_file
			r_file, v_file = self.r_file, self.v_file
			cr = CubicSpline(t_file, r_file)
			cv = CubicSpline(t_file, v_file)
			t_min, t_max = np.min(t_file), np.max(t_file)

			if t_min <= t0_new <= t_max:
				# self.t0 = t0_new
				r = cr(t0_new)
				v = cv(t0_new)
				OE_new = RV2OE(r,v,mu=self.mu)

			elif t0_new < t_min:
				# extrapolate left
				# OE = RV2OE(r_file[0],v_file[0],mu=self.mu)
				# self.t0 = t_min
				# self.set_OEs(*OE)
				OE_new = propagate_func(t0_new)
				# self.t0 = t0_new

			elif t0_new > t_max:
				OE = RV2OE(r_file[-1],v_file[-1],mu=self.mu)
				self.t0 = t_max
				self.set_OEs(*OE)
				OE_new = propagate_func(t0_new)
				# self.t0 = t0_new

			# self.t0 = t0_new
			# self.set_OEs(*OE_new)
			# self.t_file = (t_file-t_file[0]) + t0_new
			# r_file_new, v_file_new = self.propagate(self.t_file)
			# self.r_file = r_file_new
			# self.v_file = v_file_new


		self.t0 = t0_new
		self.set_OEs(*OE_new)
		if reset_epoch:
			self.t0 = 0.0

		# Maybe should be
		# if not reset_epoch:
		#	self.t0 = t0_new
		# don't update t0 otherwise


	def _format_input_array(self, t):
		if not (type(t) is np.ndarray):
			t = np.array(t)
		scalar = 0
		if len(t.shape) == 0:
			t = np.array([t])
			scalar = 1
		return t, scalar

	def propagate_kepler(self, t, return_OEs=False):
		# from leocat.utils.math import R1, R3
		R1 = lambda th: np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
		R3 = lambda th: np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
		t, scalar = self._format_input_array(t)

		mu = self.mu

		a = self.a
		e = self.e
		inc = self.inc		
		LAN = self.LAN
		omega = self.omega

		t0 = self.t0
		n = self.n
		M0 = self.M0
		M = M0 + n*(t-t0)
		nu_prop = M2nu(M,e)
		# nu_prop = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2.0)) # rad
		nu = nu_prop

		if return_OEs:
			num = len(t)
			OE_rtn = np.transpose([np.full(num,a),np.full(num,e),np.full(num,inc),
									np.full(num,LAN),np.full(num,omega),nu])
			#
			if scalar:
				return OE_rtn[0]
			else:
				return OE_rtn

		p = a*(1-e**2)

		Z = np.zeros(nu.shape)
		r = p/(1+e*np.cos(nu))
		r_pqw = np.transpose([r*np.cos(nu),r*np.sin(nu),Z])

		v0 = np.sqrt(mu/p)
		v_pqw = np.transpose([-v0*np.sin(nu),v0*(e + np.cos(nu)),Z])

		R_LAN = R3(LAN)
		R_inc = R1(inc)
		R_omega = R3(omega)
		R_313 = R_LAN @ (R_inc @ R_omega)

		# This is only valid b/c R_313 is constant for kepler
		#	otherwise need matmul (as used in SPE thru OE2RV)
		r_eci = (R_313 @ r_pqw.T).T
		v_eci = (R_313 @ v_pqw.T).T

		if scalar:
			return r_eci[0], v_eci[0]
		else:
			return r_eci, v_eci


	def propagate_SPE(self, t, frozen=False, return_OEs=False):

		t, scalar = self._format_input_array(t)
		a, e, inc = self.a, self.e, self.inc
		M_dot = get_M_dot(a,e,inc,mu=self.mu,J2=self.J2)
		omega_dot = get_omega_dot(a,e,inc,mu=self.mu,J2=self.J2)
		if frozen:
			omega_dot = 0.0
		LAN_dot = get_LAN_dot(a,e,inc,mu=self.mu,J2=self.J2)
		M0 = self.M0
		t0 = self.t0
		M = M0 + M_dot*(t-t0)
		nu_prop = M2nu(M,e)
		nu = nu_prop

		omega0 = self.omega
		LAN0 = self.LAN
		omega = omega0 + omega_dot*(t-t0)
		LAN = LAN0 + LAN_dot*(t-t0)

		num = len(t)
		OE_SPE = np.transpose([np.full(num,a), np.full(num,e), np.full(num,inc),
								LAN, omega, nu])
		#
		if return_OEs:
			if scalar:
				return OE_SPE[0]
			else:
				return OE_SPE

		r_eci, v_eci = OE2RV(OE_SPE, mu=self.mu)
		if scalar:
			return r_eci[0], v_eci[0]
		else:
			return r_eci, v_eci


	def propagate_cowell(self, t, pert='kepler', return_OEs=False):
		from scipy import integrate

		rtol, atol = self.prop_tol, self.prop_tol
		t, scalar = self._format_input_array(t)
		if pert == 'kepler':
			ode_cowell = lambda t, y: ode_kepler(t, y, R_earth, self.mu)
		elif pert == 'J2':
			ode_cowell = lambda t, y: ode_kepler_J2(t, y, R_earth, self.mu, self.J2)

		t0 = self.t0
		y0 = np.hstack([self.r,self.v])
		if scalar:
			if t0 != t[0]:
				sol = integrate.solve_ivp(ode_cowell, [t0,t[0]], t_eval=t, y0=y0, rtol=rtol, atol=atol)
				y = sol.y.T
			else:
				y = np.array([y0])
		else:
			bf = t >= t0 # forward
			br = t < t0 # reverse
			y = np.full((len(t),6), np.nan)
			# print(y.shape)
			if br.any():
				sol_r = integrate.solve_ivp(ode_cowell, [t0,np.min(t[br])], t_eval=np.flip(t[br]), y0=y0, rtol=rtol, atol=atol)
				yr = sol_r.y.T
				# print(yr.shape)
				y[br] = np.flipud(yr)
			if bf.any():
				sol_f = integrate.solve_ivp(ode_cowell, [t0,np.max(t[bf])], t_eval=t[bf], y0=y0, rtol=rtol, atol=atol)
				yf = sol_f.y.T
				# print(yf.shape)
				y[bf] = yf

		# sol = integrate.solve_ivp(ode_cowell, [self.t0,t[-1]], t_eval=tau, y0=y0, rtol=rtol, atol=atol)
		# y = sol.y
		r_eci = y[:,0:3]
		v_eci = y[:,3:6]
		if return_OEs:
			OE_rtn = RV2OE(r_eci,v_eci,mu=self.mu)
			if scalar:
				return OE_rtn[0]
			else:
				return OE_rtn

		if scalar:
			return r_eci[0], v_eci[0]
		else:
			return r_eci, v_eci


	def plot_orbit(self, t1=None, t2=None, N=1000, elev=40, azim=40, return_fig_ax=False):

		from leocat.utils.plot import make_fig, set_axes_equal, \
									remove_axes, plot_axes, pro_plot, \
									draw_vector, set_aspect_equal
		#
		pro_plot()

		t0 = self.t0
		T = self.T
		if t1 is None:
			t1 = t0
		if t2 is None:
			t2 = t0 + T

		t = np.linspace(t1,t2,N)
		r, v = self.propagate(t)
		r1, v1 = r[0], v[0]
		r2, v2 = r[-1], v[-1]

		a, e, inc = self.a, self.e, self.inc
		LAN, omega, nu = self.LAN, self.omega, self.nu

		e_hat = R3(LAN) @ R1(inc) @ R3(omega) @ np.array([1,0,0])
		h_hat = unit(np.cross(r1,v1))
		q_hat = np.cross(h_hat,e_hat)

		n_hat = R3(LAN) @ np.array([1,0,0])

		fig, ax = make_fig('3d')
		ax.plot(r.T[0], r.T[1], r.T[2])
		ax.plot(r1[0], r1[1], r1[2], 'g.', ms=5)
		ax.plot(r2[0], r2[1], r2[2], 'r.', ms=2)
		set_axes_equal(ax)
		set_aspect_equal(ax)
		remove_axes(fig,ax)
		plot_axes(ax,R_earth)
		# ax.view_init(30,60)
		ax.view_init(elev,azim)

		draw_vector(ax, [0,0,0], R_earth*e_hat, c='r', linestyle='--')
		draw_vector(ax, [0,0,0], R_earth*q_hat, c=[0,0.9,0], linestyle='--')
		draw_vector(ax, [0,0,0], R_earth*h_hat, c='b', linestyle='--')
		draw_vector(ax, [0,0,0], R_earth*n_hat, c='C1', linestyle='--')

		OE = self.OE
		a_str = '%.2f' % OE[0]
		if OE[1] < 1e-4:
			e_str = '%.2e' % OE[1]
		else:
			e_str = '%.4f' % OE[1]
		inc_str = '%.2f' % np.degrees(OE[2])
		LAN_str = '%.2f' % np.degrees(OE[3])
		omega_str = '%.2f' % np.degrees(OE[4])
		nu_str = '%.2f' % np.degrees(OE[5])
		labels = {r'$a$': [a_str,' km'], r'$e$': [e_str,''], r'$i$': [inc_str,degree_str],
					r'$\Omega$': [LAN_str,degree_str], r'$\omega$': [omega_str,degree_str],
					r'$\nu$': [nu_str,degree_str]}
		#

		title_OEs = ''
		for key in labels:
			OE_str, unit_str = labels[key]
			label = key + ': ' + OE_str + unit_str
			ax.plot(np.nan, np.nan, '.', c='w', label=label)

		legend = ax.legend(loc='upper right')
		for text in legend.get_texts():
			text.set_position((-32,0))
		legend.set_draggable(True)

		# title = title + '\n' + title_OEs
		# ax.set_title(title)
		fig.tight_layout()

		if return_fig_ax:
			return fig, ax

		fig.show()


	# def plot_ground_track(JD=None):
	# 	t0 = self.t0
	# 	T = self.T
	# 	if t1 is None:
	# 		t1 = t0
	# 	if t2 is None:
	# 		t2 = t0 + T

	# 	t = np.linspace(t1,t2,N)
	# 	r, v = self.propagate(t)

	# 	if JD is None:
	# 		R_ECI_ECF = np.
	# 	R_ECI_ECF = get_R_ECI_ECF(JD)



def TLE_to_orb(tle_str):
	from tletools import TLE
	tle_lines = tle_str.strip().splitlines()
	tle = TLE.from_lines(*tle_lines)
	orb_ap = tle.to_orbit()
	a, ecc, inc, raan, argp, nu = orb_ap.classical()
	a, e, inc, LAN, omega, nu = [x.value for x in orb_ap.classical()]
	inc = np.radians(inc)
	LAN = np.radians(LAN)
	omega = np.radians(omega)
	nu = np.radians(nu)
	orb = Orbit(a, e, inc, LAN, omega, nu, propagator='SPE+frozen', warn=False)
	JD0 = tle.epoch.jd
	return orb, JD0


def get_nodal_crossing(r_ecf, JD, split_ascending=True, tr_lla_ecf=None):

	from scipy.interpolate import CubicSpline
	from leocat.utils.math import newton_raphson, wrap, unwrap

	if tr_lla_ecf is None:
		from leocat.utils.geodesy import ecf_to_lla
		lon, lat, alt = ecf_to_lla(r_ecf.T[0], r_ecf.T[1], r_ecf.T[2])
	else:
		lon, lat, alt = tr_lla_ecf.transform(r_ecf.T[0], r_ecf.T[1], r_ecf.T[2], direction='inverse')

	if split_ascending:
		index = np.where((lat[:-1] < 0) & (lat[1:] > 0))[0].astype(int)
	else:
		index = np.where((lat[:-1] > 0) & (lat[1:] < 0))[0].astype(int)

	t = (JD-JD[0])*86400
	clat = CubicSpline(t,lat)
	dclat = clat.derivative()
	t_node, dt_node = newton_raphson(t[index], clat, dclat)
	b = (t_node >= np.min(t)) & (t_node <= np.max(t))
	t_node = t_node[b]

	clon = CubicSpline(t,unwrap(lon))
	lon_node = wrap(clon(t_node))
	lat_node = clat(t_node)

	clon = CubicSpline(t,unwrap(lon))
	lon_node = wrap(clon(t_node))
	lat_node = clat(t_node)

	JD_node = JD[0] + t_node/86400

	return lon_node, JD_node



def orb_to_tracks(orb, num_tracks, JD1, dt_track=1.0):

	propagator = orb.propagator
	if not (propagator == 'kepler' or propagator == 'SPE+frozen'):
		import warnings
		warnings.warn('orbit is not periodic with nodal period')

	Tn = orb.get_period('nodal')
	tn_beg = np.arange(num_tracks)*Tn
	tn_end = tn_beg + Tn

	dt_intp = dt_track # sec
	N_track = int(Tn/dt_intp)
	dt_intp = Tn/N_track
	t_track_init = np.arange(0,N_track)*dt_intp + dt_intp/2

	r_track = []
	t_track = []
	for j in range(num_tracks):
		t_track0 = t_track_init + Tn*j
		r_eci, v_eci = orb.propagate(t_track0)
		JD_track = JD1 + t_track0/86400
		r_ecf = convert_ECI_ECF(JD_track, r_eci)
		r_track.append(r_ecf)
		t_track.append(t_track0)

	r_track = np.array(r_track)
	t_track = np.array(t_track)

	return t_track, r_track



def M2nu(M, e, tol=1e-12):
	from scipy.optimize import newton
	f = lambda E, M: E - e*np.sin(E) - M
	fp = lambda E, M: 1 - e*np.cos(E)
	try:
		E_est = newton(f, M, fprime=fp, args=(M,), rtol=tol)
	except RuntimeError:
		import warnings
		warnings.warn('Newton failed to converge, approximating E = M instead')
		E_est = M
	nu_est = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E_est/2.0))
	return nu_est

def nu2M(nu,e):
	E = 2*np.arctan(np.tan(nu/2)/(np.sqrt((1+e)/(1-e))))
	M = E - e*np.sin(E)
	return M


def RV2OE(r_vec, v_vec, mu=398600.4418):

	"""
	Vallado, D., "Fundamentals of Astrodynamics and Applications", version 4

	Does not handle rectilinear orbit or retrograde equatorial
		can do orbits, parabolic, or hyperbolic trajectories
	r_vec and v_vec can either be single vectors or Nx3 matrices
	returns OE = [a,e,inc,LAN,omega,nu]

	"""
	tol = 1e-10
	from leocat.utils.math import mag, unit, dot, divide, check_proj
	if r_vec.shape != v_vec.shape:
		raise ValueError('r must have same shape as v')

	single = 0
	if len(r_vec.shape) == 1:
		r_vec = np.array([r_vec])
		v_vec = np.array([v_vec])
		single = 1

	h_vec = np.cross(r_vec,v_vec)
	n_vec = np.cross([0,0,1],h_vec)

	h = mag(h_vec)
	if (h < tol).any():
		raise ValueError('orbit is rectilinear (h ~ 0)')

	r = mag(r_vec)
	v = mag(v_vec)
	arg1 = v**2 - mu/r
	arg2 = dot(r_vec,v_vec)
	if not single:
		e_vec_x = (arg1*r_vec.T[0] - arg2*v_vec.T[0])/mu
		e_vec_y = (arg1*r_vec.T[1] - arg2*v_vec.T[1])/mu
		e_vec_z = (arg1*r_vec.T[2] - arg2*v_vec.T[2])/mu
		e_vec = np.transpose([e_vec_x,e_vec_y,e_vec_z])
	else:
		e_vec = (arg1*r_vec - arg2*v_vec)/mu
	e = mag(e_vec)

	energy = v**2/2 - mu/r
	bp = np.abs(e-1) < tol # parabolic orbit
	a = divide(-mu,2*energy) # may div by zero for parab
	a[bp] = np.inf
	p = h**2/mu

	# h != 0 (non-rectilinear)
	proj = check_proj(divide(h_vec.T[2],h))
	inc = np.arccos(proj)
	# inc[np.isnan(inc)] = 0.0 # h = 0, rectilinear
	# inc[h < 1e-10] = 0.0
	# b_eq = (inc < tol) | (np.abs(inc-np.pi) < tol)

	# h != [0,0,1] (inc != 0 or 180, non-rectilinar)
	n_hat = unit(n_vec)
	LAN = np.arctan2(n_hat.T[1],n_hat.T[0])
	# LAN[np.isnan(LAN)] = 0.0 # inc = 0, equatorial
	# LAN[(inc < 1e-10) | (h < 1e-10)] = 0.0
	# LAN[b_eq] = 0.0

	# e != 0, inc != 0
	# b_c = e < tol
	e_hat = unit(e_vec)
	proj = check_proj(dot(n_hat,e_hat))
	omega = np.arccos(proj)
	bk = e_hat.T[2] < 0.0
	omega[bk] = -omega[bk]
	# omega[np.isnan(omega)] = 0.0
	# omega[b_c | b_eq] = 0.0

	# e != 0
	r_hat = unit(r_vec)
	proj = check_proj(dot(e_hat,r_hat))
	nu = np.arccos(proj)
	bk = dot(r_vec,v_vec) < 0.0
	nu[bk] = -nu[bk]
	# nu[np.isnan(nu)] = 0.0
	# nu[b_c] = 0.0

	b_c = e < tol
	b_eq = (inc < tol) # | (np.abs(inc-np.pi) < tol)
	# print('e', e)
	# print('b_c', b_c)
	# print('inc', inc)
	# print('b_eq', b_eq)
	b_c_eq = b_c & b_eq
	b_c_inc = b_c & (~b_eq)
	b_e_eq = (~b_c) & b_eq
	# b_c_inc = b_c & (~b_c_eq)
	# b_e_eq = b_eq & (~b_c_eq)

	if b_c_eq.any():
		# circular equatorial
		# print('b_c_eq')
		LAN[b_c_eq] = 0.0
		omega[b_c_eq] = 0.0
		r_hat_c_eq = r_hat[b_c_eq]
		proj = check_proj(r_hat_c_eq.T[0])
		nu_c_eq = np.arccos(proj)
		nu_c_eq[r_hat_c_eq.T[1] < 0] = -nu_c_eq[r_hat_c_eq.T[1] < 0]
		nu[b_c_eq] = nu_c_eq

	if b_c_inc.any():
		# circular inclined
		# print('b_c_inc')
		omega[b_c_inc] = 0.0
		n_hat_c_inc = n_hat[b_c_inc]
		r_hat_c_inc = r_hat[b_c_inc]
		proj = check_proj(dot(n_hat_c_inc,r_hat_c_inc))
		nu_c_inc = np.arccos(proj)
		nu_c_inc[r_hat_c_inc.T[2] < 0] = -nu_c_inc[r_hat_c_inc.T[2] < 0]
		nu[b_c_inc] = nu_c_inc

	if b_e_eq.any():
		# elliptical equatorial
		# print('b_e_eq')
		LAN[b_e_eq] = 0.0
		e_hat_e_eq = e_hat[b_e_eq]
		proj = check_proj(e_hat_e_eq.T[0])
		omega_e_eq = np.arccos(proj)
		omega_e_eq[e_hat_e_eq.T[1] < 0] = -omega_e_eq[e_hat_e_eq.T[1] < 0]
		omega[b_e_eq] = omega_e_eq


	if single:
		OE = np.array([a[0],e[0],inc[0],LAN[0],omega[0],nu[0]])
	else:
		OE = np.transpose([a,e,inc,LAN,omega,nu])

	return OE


def OE2RV(OE, mu=398600.4418, p=None):
	"""
	Vallado, D., "Fundamentals of Astrodynamics and Applications", version 4

	Does not handle rectilinear orbit or retrograde equatorial
		can do orbits, parabolic, or hyperbolic trajectories
		for e ~ 1 (parabolic), must input p (p = 2*radius at periapsis for parabolic)
		if e > 1 (hyperbolic), a must be negative but not enforced
	OE can either be single vectors or an Nx6 matrix
	returns r, v as 3x1 or Nx3 time-series vectors

	"""
	from leocat.utils.math import matmul, R1, R3

	tol = 1e-10
	if not (type(OE) is np.ndarray):
		OE = np.array(OE)

	single = 0
	if len(OE.shape) == 1:
		a, e, inc, LAN, omega, nu = OE
		single = 1
	else:
		a, e, inc, LAN, omega, nu = OE.T

	# a, e, inc, LAN, omega, nu = OE
	if p is None:
		p = a*(1-e**2)

	Z = 0.0
	if not single:
		Z = np.zeros(len(a))

	r = p/(1+e*np.cos(nu))
	r_pqw = np.transpose([r*np.cos(nu),r*np.sin(nu),Z])

	v0 = np.sqrt(mu/p)
	v_pqw = np.transpose([-v0*np.sin(nu),v0*(e + np.cos(nu)),Z])

	R_LAN = R3(LAN)
	R_inc = R1(inc)
	R_omega = R3(omega)
	R_313 = R_LAN @ (R_inc @ R_omega)

	# print('LAN')
	# print(LAN)
	# print(R_LAN)
	# print('inc')
	# print(inc)
	# print(R_inc)
	# print('omega')
	# print(omega)
	# print(R_omega)
	# print('R_313')
	# print(R_313)

	# print(R_313)
	# print(inc)
	# print(omega)
	# print('single', single)

	if not single:
		r_eci = matmul(R_313,r_pqw)
		v_eci = matmul(R_313,v_pqw)
	else:
		r_eci = (R_313 @ r_pqw.T).T
		v_eci = (R_313 @ v_pqw.T).T

	return r_eci, v_eci



def ode_kepler_J2(t, y, R, mu, J2):

	# R = R_earth
	r, v = y[0:3], y[3:6]
	rx, ry, rz = r
	r_dot = v
	r_mag2 = np.linalg.norm(r)**2
	J3 = 0 #-0.0000025327

	v_dotx = -(15*J3*mu*rx*R**3*r_mag2*rz - 35*J3*mu*rx*R**3*rz**3 + 3*J2*mu*rx*R**2*r_mag2**2 - 15*J2*mu*rx*R**2*r_mag2*rz**2 + 2*mu*rx*r_mag2**3)/(2*r_mag2**(9/2))
	v_doty = -(15*J3*mu*ry*R**3*r_mag2*rz - 35*J3*mu*ry*R**3*rz**3 + 3*J2*mu*ry*R**2*r_mag2**2 - 15*J2*mu*ry*R**2*r_mag2*rz**2 + 2*mu*ry*r_mag2**3)/(2*r_mag2**(9/2))
	v_dotz = -(- 3*J3*mu*R**3*r_mag2**2 + 30*J3*mu*R**3*r_mag2*rz**2 - 35*J3*mu*R**3*rz**4 + 9*J2*mu*R**2*r_mag2**2*rz - 15*J2*mu*R**2*r_mag2*rz**3 + 2*mu*r_mag2**3*rz)/(2*r_mag2**(9/2))

	v_dot = np.array([v_dotx, v_doty, v_dotz])

	y_dot = np.hstack([r_dot, v_dot])
	return y_dot


def ode_kepler(t, y, R, mu):

	# R = R_earth
	r, v = y[0:3], y[3:6]
	rx, ry, rz = r
	r_dot = v
	r_mag2 = np.linalg.norm(r)**2
	J3 = 0 #-0.0000025327
	J2 = 0

	v_dotx = -(15*J3*mu*rx*R**3*r_mag2*rz - 35*J3*mu*rx*R**3*rz**3 + 3*J2*mu*rx*R**2*r_mag2**2 - 15*J2*mu*rx*R**2*r_mag2*rz**2 + 2*mu*rx*r_mag2**3)/(2*r_mag2**(9/2))
	v_doty = -(15*J3*mu*ry*R**3*r_mag2*rz - 35*J3*mu*ry*R**3*rz**3 + 3*J2*mu*ry*R**2*r_mag2**2 - 15*J2*mu*ry*R**2*r_mag2*rz**2 + 2*mu*ry*r_mag2**3)/(2*r_mag2**(9/2))
	v_dotz = -(- 3*J3*mu*R**3*r_mag2**2 + 30*J3*mu*R**3*r_mag2*rz**2 - 35*J3*mu*R**3*rz**4 + 9*J2*mu*R**2*r_mag2**2*rz - 15*J2*mu*R**2*r_mag2*rz**3 + 2*mu*r_mag2**3*rz)/(2*r_mag2**(9/2))

	v_dot = np.array([v_dotx, v_doty, v_dotz])

	y_dot = np.hstack([r_dot, v_dot])
	return y_dot

def get_inc_sso(a_range_sso, e=0.0):
	arg1 = -2*a_range_sso**(7/2) * LAN_dot_SSO * (1-e**2)**2
	arg2 = 3*R_earth**2 * J2 * np.sqrt(MU)
	inc_range_sso = np.arccos(arg1/arg2)
	return inc_range_sso

def get_a_sso(inc_range_sso, e=0.0):
	arg1 = 3*R_earth**2*J2*np.sqrt(MU)*np.cos(inc_range_sso)
	arg2 = 2*LAN_dot_SSO*(1-e**2)**2
	return (-arg1/arg2)**(2/7)

def get_frozen_ecc(a, inc, J2=0.00108248, J3=-0.0000025327):
	# Vallado ch. 11 sec. 4
	#	approximate frozen eccentricity for omega=90 or 270 (Earth)
	e0 = -0.5 * J3/J2 * (R_earth/a) * np.sin(inc)
	return e0

def get_nodal_day(a, e, inc, radians=True, return_LAN_dot=False):
	LAN_dot = get_LAN_dot(a, e, inc, radians=radians)
	Dn = 2*np.pi / (W_EARTH - LAN_dot) # sec
	if not return_LAN_dot:
		return Dn
	else:
		return Dn, LAN_dot


def get_LAN_dot(a, e, inc, radians=True, mu=398600.4418, J2=0.00108248):
	p = a*(1-e**2)
	n = np.sqrt(mu/a**3)
	if not radians:
		inc = np.radians(inc)
	return -3*n*J2/2 * (R_earth/p)**2 * np.cos(inc)


def get_omega_dot(a, e, inc, radians=True, mu=398600.4418, J2=0.00108248):
	# units in km, s
	n = np.sqrt(mu/a**3)

	if not radians:
		inc = np.radians(inc)

	omega_dot = -3/4 * n * (R_earth/a)**2 * J2 * (1/(1-e**2)**2) * (1-5*np.cos(inc)**2)
	return omega_dot


def get_M_dot(a, e, inc, radians=True, mu=398600.4418, J2=0.00108248):
	# units in km, s
	n = np.sqrt(mu/a**3)

	if not radians:
		inc = np.radians(inc)	

	arg = 1 - 3/4 * (R_earth/a)**2 * J2 * (1/(1-e**2)**(3/2)) * (1 - 3*np.cos(inc)**2)
	M_dot = n*arg
	return M_dot

def get_repeat_dlon(k_day2rep, k_rev2rep, deg=False):
	dlam_rev = 2*np.pi*k_day2rep/k_rev2rep # Sq
	dlam_pass = 2*np.pi/k_rev2rep # Sd
	d_rev = dlam_rev * R_earth
	d_pass = dlam_pass * R_earth
	dlon_rev = np.degrees(dlam_rev)
	dlon_pass = np.degrees(dlam_pass)
	if not deg:
		return d_rev, d_pass
	else:
		return dlon_rev, dlon_pass
		

def Walker(walker_name,h,inc,P,F,e=0.00001,omega=0):
	"""
	# https://en.wikipedia.org/wiki/Satellite_constellation
	walker_name - base name for walker constellation
	h - height in km (above eq. radius 6378.137 km)
	inc - inclination of planes in deg.
	N - number of satellites
	P - number of equally spaced planes
	F - number of satellites per plane
	"""
	R_earth = 6378.137
	dLAN = 360.0/P
	dnu = 360.0/F

	digits_P = len(str(P-1))
	digits_F = len(str(F-1))

	orbit_params_walker = []
	for i in range(P):
		for j in range(F):
			i_str = str(i).zfill(digits_P)
			j_str = str(j).zfill(digits_F)
			name = walker_name + '_%s-%s' % (i_str,j_str)
			LAN = dLAN*i
			nu = dnu*j
			orbit_params = {'a': R_earth + h, 'e': e, 'inc': inc, 'LAN': LAN, 'omega': omega, 'nu': nu,
						'name': name, 'DIR': None, 'propagator': 'kepler', 'epoch': None,
						'k_rev2rep': None, 'k_day2rep': None, 'sso': None}
			#
			# key = np.round(LAN,2),np.round(nu,2)
			# orbit_params_walker[key] = orbit_params
			orbit_params_walker.append(orbit_params)

	return orbit_params_walker



#######################################################
# start astro functions
#	to be moved to astro.py


def solar_elev(lon, lat, JD, R_ECI_ECF=None, positive=False, mean_sun=False):

	# single_value = 0
	# if not (type(lon) is np.ndarray):
	# 	single_value = 1
	# 	lon = np.array([lon])
	# 	lat = np.array([lat])
	# 	JD = np.array([JD])
	# 	if R_ECI_ECF is not None:
	# 		R_ECI_ECF = np.array([R_ECI_ECF])

	r_ecf = lla_to_ecf(lon, lat, np.zeros(lon.shape)) # km
	if R_ECI_ECF is None:
		R_ECI_ECF = get_R_ECI_ECF_GMST(JD)

	R_ECF_ECI = np.transpose(R_ECI_ECF, axes=(0,2,1))

	r_eci = matmul(R_ECF_ECI, r_ecf) # km
	r_sun = solar_pos_approx(JD) # km

	if mean_sun:
		s_eci = r_sun
		s_ecf_xy = matmul(R_ECI_ECF, s_eci)
		s_ecf_xy.T[2] = 0.0
		r_sun_xy = (s_ecf_xy.T / np.linalg.norm(s_ecf_xy,axis=1)).T * R_earth
		r_sun = matmul(R_ECF_ECI, r_sun_xy)

	z_hat = (r_eci.T / np.linalg.norm(r_eci, axis=1)).T # up, geocentric
	s_hat = (r_sun.T / np.linalg.norm(r_sun, axis=1)).T # towards sun

	proj_sun = np.einsum('ij,ij->i', z_hat, s_hat) # ut.dot
	proj_sun[proj_sun > 1.0] = 1.0
	proj_sun[proj_sun < -1.0] = -1.0
	angle = np.arccos(proj_sun) * 180/np.pi
	elev = 90.0 - angle
	if positive:
		elev[proj_sun <= 0] = 0.0 # invalid angle, below horizon

	# if single_value:
	# 	elev = elev[0]

	return elev


def solar_pos_approx(JD, return_lon_eclip=False):

	# position in km from Earth at JD
	# sun position
	# 	<1% relative error in pos
	# Fundamentals of Astrodynamics and Applications by David A. Vallado
	# 	section 5.1, algorithm 29

	AU = 149597870700 # meters, wikipedia

	UT1 = (JD - 2451545.0) / 36525
	lam = 280.460 + 36000.771*UT1 # deg
	TDB = UT1 # approx
	M = 357.5291092 + 35999.05034*TDB # deg
	lam_eclip = lam + 1.914666471*np.sin(np.radians(M)) + 0.019994643*np.sin(np.radians(2*M)) # deg
	if return_lon_eclip:
		return wrap(np.radians(lam_eclip), radians=True, two_pi=True)
	r_mag = 1.000140612 - 0.016708617*np.cos(np.radians(M)) - 0.000139589*np.cos(np.radians(2*M)) # AU
	eps = 23.439291 - 0.013004*2*TDB # deg

	rx = r_mag*np.cos(np.radians(lam_eclip))
	ry = r_mag*np.cos(np.radians(eps))*np.sin(np.radians(lam_eclip))
	rz = r_mag*np.sin(np.radians(eps))*np.sin(np.radians(lam_eclip))

	r = np.transpose([rx, ry, rz]) * AU / 1e3

	return r


def sun_lon_eclip(JD):
	# JD in days
	# outputs in radians

	r_sun = solar_pos_approx(JD) # relative to Earth
	r_sun_unit = (r_sun.T / np.linalg.norm(r_sun,axis=1)).T

	x_hat = np.array([1,0,0]) # ECI
	y_hat = np.array([0,1,0])

	proj_x = np.dot(r_sun_unit, x_hat)
	proj_x[proj_x > 1] = 1
	proj_x[proj_x < -1] = -1
	angle = np.arccos(proj_x) * 180/np.pi

	proj_y = np.dot(r_sun_unit, y_hat)
	proj_y[proj_y > 1] = 1
	proj_y[proj_y < -1] = -1
	angle[proj_y < 0] = -angle[proj_y < 0] + 360

	LE = angle

	return np.radians(LE)


def lunar_elev(lon, lat, JD, R_ECI_ECF=None, positive=False, relative=True):

	# single_value = 0
	# if not type(lon) is np.ndarray:
	# 	single_value = 1
	# 	lon = np.array([lon])
	# 	lat = np.array([lat])
	# 	JD = np.array([JD])
	# 	if R_ECI_ECF is not None:
	# 		R_ECI_ECF = np.array([R_ECI_ECF])

	r_ecf = lla_to_ecf(lon, lat, np.zeros(lon.shape)) # km
	if R_ECI_ECF is None:
		R_ECI_ECF = get_R_ECI_ECF_GMST(JD) # not validated

	R_ECF_ECI = np.transpose(R_ECI_ECF, axes=(0,2,1))

	r_eci = matmul(R_ECF_ECI, r_ecf) # km
	# r_sun = solar_pos_approx(JD) # km
	r_moon = lunar_pos_approx(JD) # km, in ECI

	# Need relative pos of moon wrt given lon/lat (pt)
	#	not for solar_elev b/c sun is so far
	#	elevation varies by ~1 deg
	if relative:
		r_moon = r_moon - r_eci

	# if mean_sun:
	# 	s_eci = r_sun
	# 	s_ecf_xy = matmul(R_ECI_ECF, s_eci)
	# 	s_ecf_xy.T[2] = 0.0
	# 	r_sun_xy = (s_ecf_xy.T / np.linalg.norm(s_ecf_xy,axis=1)).T * R_earth
	# 	r_sun = matmul(R_ECF_ECI, r_sun_xy)

	z_hat = (r_eci.T / np.linalg.norm(r_eci, axis=1)).T # up, geocentric
	m_hat = (r_moon.T / np.linalg.norm(r_moon, axis=1)).T # towards moon

	proj_moon = np.einsum('ij,ij->i', z_hat, m_hat) # ut.dot
	proj_moon[proj_moon > 1.0] = 1.0
	proj_moon[proj_moon < -1.0] = -1.0
	angle = np.arccos(proj_moon) * 180/np.pi
	elev = 90.0 - angle
	if positive:
		elev[proj_moon <= 0] = 0.0 # invalid angle, below horizon

	# if single_value:
	# 	elev = elev[0]

	return elev



def lunar_pos_approx(JD, return_lon_eclip=False, R_earth=6378.137):

	"""
	<1% relative error

	"""
	def sin_deg(angle):
		return np.sin(np.radians(angle))

	def cos_deg(angle):
		return np.cos(np.radians(angle))

	# AU = 149597870700 # meters, wikipedia
	# R_earth = 6378.137 # slight difference from Vallado

	JD_TDB = JD # approx, use algorithm 16 Vallado to find TDB
	T_TDB = (JD_TDB - 2451545.0) / 36525.0

	# all in degrees
	lam_eclip = 218.32 + 481267.8813*T_TDB + \
				6.29*sin_deg(134.9 + 477198.85*T_TDB) - \
				1.27*sin_deg(259.2 - 413335.38*T_TDB) + \
				0.66*sin_deg(235.7 + 890534.23*T_TDB) + \
				0.21*sin_deg(269.9 + 954397.70*T_TDB) - \
				0.19*sin_deg(357.5 + 35999.05*T_TDB) - \
				0.11*sin_deg(186.6 + 966404.05*T_TDB)
	#
	if return_lon_eclip:
		return wrap(np.radians(lam_eclip), radians=True, two_pi=True)

	phi_eclip = 5.13*sin_deg(93.3 + 483202.03*T_TDB) + \
				0.28*sin_deg(228.2 + 960400.87*T_TDB) - \
				0.28*sin_deg(318.3 + 6003.18*T_TDB) - \
				0.17*sin_deg(217.6 - 407332.20*T_TDB)
	#
	p = 0.9508 + 0.0518*cos_deg(134.9 + 477198.85*T_TDB) + \
		0.0095*cos_deg(259.2 - 413335.38*T_TDB) + \
		0.0078*cos_deg(235.7 + 890534.23*T_TDB) + \
		0.0028*cos_deg(269.9 + 954397.70*T_TDB)
	#
	eps = 23.439291 - 0.0130042*T_TDB - 1.64e-7*T_TDB**2 + 5.04e-7*T_TDB**3
	dist_moon = R_earth/sin_deg(p)

	rx = dist_moon * (cos_deg(phi_eclip)*cos_deg(lam_eclip))
	ry = dist_moon * (cos_deg(eps)*cos_deg(phi_eclip)*sin_deg(lam_eclip) - sin_deg(eps)*sin_deg(phi_eclip))
	rz = dist_moon * (sin_deg(eps)*cos_deg(phi_eclip)*sin_deg(lam_eclip) + cos_deg(eps)*sin_deg(phi_eclip))

	if type(rx) is np.ndarray:
		r_moon = np.transpose([rx,ry,rz])
	else:
		r_moon = np.array([rx,ry,rz])

	return r_moon


def get_lunar_p(JD):
	def sin_deg(angle):
		return np.sin(np.radians(angle))
	def cos_deg(angle):
		return np.cos(np.radians(angle))

	# from lunar_pos_approx
	JD_TDB = JD # approx, use algorithm 16 Vallado to find TDB
	T_TDB = (JD_TDB - 2451545.0) / 36525.0
	p = 0.9508 + 0.0518*cos_deg(134.9 + 477198.85*T_TDB) + \
		0.0095*cos_deg(259.2 - 413335.38*T_TDB) + \
		0.0078*cos_deg(235.7 + 890534.23*T_TDB) + \
		0.0028*cos_deg(269.9 + 954397.70*T_TDB) # deg
	#
	return p

def lunar_phase(JD):
	# returns lunar phase in deg (rel to geocentric)
	# Astronomical Algorithms 1992 and Vallado 5.3.4
	# 	Astr. Alg. eq. 48.2
	# 	Astr. Alg. eq. 48.3
	from leocat.utils.geodesy import cart_to_RADEC

	r_sun = solar_pos_approx(JD)
	RA_sun, DEC_sun = cart_to_RADEC(r_sun) # deg
	r_moon = lunar_pos_approx(JD)
	RA_moon, DEC_moon = cart_to_RADEC(r_moon) # deg

	def sin_deg(angle):
		return np.sin(np.radians(angle))
	def cos_deg(angle):
		return np.cos(np.radians(angle))

	arg1 = sin_deg(DEC_sun)*sin_deg(DEC_moon)
	arg2 = cos_deg(DEC_sun)*cos_deg(DEC_moon)*cos_deg(RA_sun-RA_moon)
	E = np.arccos(arg1 + arg2) * 180/np.pi # deg, # "elongation" phi in Astr. Alg. eq. 48.2
	# phase = 180.0 - E # deg, similar to Astr. Alg. eq. 48.4.., neglects lunar latitude

	# Astr. Alg. eq. 48.3, includes lunar latitude
	phi = np.radians(E)
	r_sun_mag, r_moon_mag = mag(r_sun), mag(r_moon)
	y = r_sun_mag * np.sin(phi)
	x = r_moon_mag - r_sun_mag*np.cos(phi)
	# arg = r_sun_mag * np.sin(phi) / (r_moon_mag - r_sun_mag*np.cos(phi))
	# phase = np.arctan(arg) * 180/np.pi # deg, lunar phase
	phase = np.arctan2(y,x) * 180/np.pi # deg, lunar phase

	return phase


def lunar_illumination_fraction(phase, radians=False):
	# percent of lunar surface illuminated by sun w.r.t. center of Earth
	#	input phase as deg.
	# Astronimal Algorithms 1992 eq. 48.1
	if not radians:
		phase_rad = np.radians(phase)
	else:
		phase_rad = phase
	p_illum = (1 + np.cos(phase_rad))/2.0
	return p_illum


def surface_illumination(elev, JD, body='sun', phase=None, return_log=False):

	# Returns illumination on Earth's surface from given body
	#	body either sun or moon (considers lunar phases)
	#	illumination in lux, or luminous flux lumens/m^2
	# Vallado Ch.5 Sec. 5.3.4
	# elev in deg, should not be an absolute value

	import pandas as pd
	if body == 'sun':
		l0_coefs = [13.84, 2.70, 2.88, 2.88, 3.05, 3.74]
		l1_coefs = [262.72, 12.17, 21.81, 22.26, 13.28, 3.97]
		l2_coefs = [1447.42, -431.69, -258.11, -207.64, -45.98, -4.07]
		l3_coefs = [2797.93, -1899.83, -858.36, 1034.30, 64.33, 1.47]
		elev_thresholds = [-91, -18, -12, -5, -0.8, 5, 20, 90]

	elif body == 'moon':
		l0_coefs = [-2.79, -2.58, -1.95]
		l1_coefs = [24.27, 12.58, 4.06]
		l2_coefs = [-252.95, -42.58, -4.24]
		l3_coefs = [1321.29, 59.06, 1.56]
		elev_thresholds = [-91, -0.8, 5, 20, 90]

		def sin_deg(angle):
			return np.sin(np.radians(angle))
		def cos_deg(angle):
			return np.cos(np.radians(angle))

		# from lunar_pos_approx
		JD_TDB = JD # approx, use algorithm 16 Vallado to find TDB
		T_TDB = (JD_TDB - 2451545.0) / 36525.0
		p = 0.9508 + 0.0518*cos_deg(134.9 + 477198.85*T_TDB) + \
			0.0095*cos_deg(259.2 - 413335.38*T_TDB) + \
			0.0078*cos_deg(235.7 + 890534.23*T_TDB) + \
			0.0028*cos_deg(269.9 + 954397.70*T_TDB) # deg
		#

		if phase is None:
			phase = lunar_phase(JD)
		else:
			phase = np.full(elev.shape, phase)


	elev[elev > 90] = 90.0
	elev[elev < -90] = -90.0
	x = elev/90.0
	L = np.full(elev.shape, np.nan)
	df = pd.DataFrame({'elev': elev})
	df['bin'] = pd.cut(df['elev'], elev_thresholds)
	index_bin = df.groupby('bin',observed=False).indices
	for j,key in enumerate(index_bin):
		index = index_bin[key]
		i = j-1
		if len(index) == 0 or i < 0:
			continue

		elev_bin = elev[index]
		l0 = l0_coefs[i]
		l1 = l1_coefs[i]
		l2 = l2_coefs[i]
		l3 = l3_coefs[i]

		x_idx = x[index]
		L1 = l0 + l1*x_idx + l2*x_idx**2 + l3*x_idx**3
		if body == 'moon':
			L2 = -8.68e-3*phase[index] - 2.2e-9*phase[index]**4
			L3 = 2*np.log10(p[index]/0.951)
			L1 = L1 + L2 + L3

		L[index] = L1

	if not return_log:
		M = 10**L
		M[np.isnan(L)] = 0.0
		return M
	else:
		return L


def surface_illumination_approx(elev, phase=None, return_log=False):

	# Vallado Ch.5 Sec. 5.3.4
	# elev in deg, should not be an absolute value
	#	removed variation from JD
	#	lunar illum approx depends only on elev and phase

	eps = 1e-6
	import pandas as pd
	if phase is None:
		l0_coefs = [13.84, 2.70, 2.88, 2.88, 3.05, 3.74]
		l1_coefs = [262.72, 12.17, 21.81, 22.26, 13.28, 3.97]
		l2_coefs = [1447.42, -431.69, -258.11, -207.64, -45.98, -4.07]
		l3_coefs = [2797.93, -1899.83, -858.36, 1034.30, 64.33, 1.47]
		elev_thresholds = [-91, -18-eps, -12, -5, -0.8, 5, 20, 90]

	else:
		l0_coefs = [-2.79, -2.58, -1.95]
		l1_coefs = [24.27, 12.58, 4.06]
		l2_coefs = [-252.95, -42.58, -4.24]
		l3_coefs = [1321.29, 59.06, 1.56]
		elev_thresholds = [-91, -0.8-eps, 5, 20, 90]
		if not (phase is np.ndarray):
			phase = np.full(elev.shape, phase)


	elev[elev > 90] = 90.0
	elev[elev < -90] = -90.0
	x = elev/90.0
	L = np.full(elev.shape, np.nan)
	df = pd.DataFrame({'elev': elev})
	df['bin'] = pd.cut(df['elev'], elev_thresholds)
	index_bin = df.groupby('bin',observed=False).indices
	for j,key in enumerate(index_bin):
		index = index_bin[key]
		i = j-1
		if len(index) == 0 or i < 0:
			continue

		elev_bin = elev[index]
		l0 = l0_coefs[i]
		l1 = l1_coefs[i]
		l2 = l2_coefs[i]
		l3 = l3_coefs[i]

		x_idx = x[index]
		L1 = l0 + l1*x_idx + l2*x_idx**2 + l3*x_idx**3
		if not (phase is None):
			L2 = -8.68e-3*phase[index] - 2.2e-9*phase[index]**4
			# L3 = 2*np.log10(p[index]/0.951)
			L1 = L1 + L2 #+ L3

		L[index] = L1

	if not return_log:
		M = 10**L
		M[np.isnan(L)] = 0.0
		return M
	else:
		return L

# end astro functions
###########################################################

def get_GMST(JD, angle=True):
	"""
	Vallado v4, algorithm 15
		JD technically based on UT1
		for the ex, same as our def.
	Due to precision,
		time accurate to 0.1 sec
		angle accurate to 1e-4 deg.
	returns either hrs or deg
		relative to vernal equinox
	"""
	T = (JD - 2451545.0)/36525.0
	h = 876600.0 * 3600
	GMST = 67310.54841 + (h + 8640184.812866)*T + 0.093104*T**2 - 6.2e-6*T**3
	GMST = GMST % 86400
	if angle:
		return GMST / 240 # deg
	else:
		return GMST / 3600 # hrs

def get_LST(lon, JD, angle=True):
	"""
	Vallado, algorithm 15
	LST = local sidereal time
	returns either hrs or deg
		relative to vernal equinox
	"""
	GMST_angle = get_GMST(JD, angle=True) # deg
	LST = wrap(GMST_angle + lon) # deg
	if not angle:
		LST = LST * 24/360 # hrs
	return LST


def get_R_ECI_ECF_GMST(JD, inverse=False):
	# return rot for ECI to ECF
	# r_ecf = R_ECI_ECF @ r_eci
	theta_GMST = get_GMST(JD, angle=True) # deg
	theta_GMST = np.radians(theta_GMST)
	if type(JD) is np.ndarray:
		R_ECI_ECF = R3(-theta_GMST)
	else:
		R_ECI_ECF = R3(-theta_GMST)
	if inverse:
		try:
			# vectorized
			R_ECF_ECI = np.transpose(R_ECI_ECF, axes=(0,2,1))
		except ValueError:
			# single matrix
			R_ECF_ECI = R_ECI_ECF.T
		return R_ECF_ECI
	return R_ECI_ECF

def convert_ECI_ECF(JD, r_eci, v_eci=None):

	scalar = 0
	if not (type(JD) is np.ndarray):
		JD = np.array(JD)
		r_eci = np.array(r_eci)
		if v_eci is not None:
			v_eci = np.array(v_eci)
	if len(JD.shape) == 0:
		JD = np.array([JD])
		r_eci = np.array([r_eci])
		if v_eci is not None:
			v_eci = np.array([v_eci])
		scalar = 1

	R_ECI_ECF = get_R_ECI_ECF_GMST(JD)
	r_ecf = matmul(R_ECI_ECF,r_eci)
	if v_eci is None:
		if scalar:
			return r_ecf[0]
		else:
			return r_ecf
	else:
		pole_axis = R_ECI_ECF[:,:,2]
		v_ecf = matmul(R_ECI_ECF,v_eci - np.cross(W_EARTH*pole_axis, r_eci)) # validated against CubicSpline
		# return r_ecf, v_ecf
		if scalar:
			return r_ecf[0], v_ecf[0]
		else:
			return r_ecf, v_ecf


def convert_ECF_ECI(JD, r_ecf, v_ecf=None):
	# scalar = 0
	# if not (type(JD) is np.ndarray):
	# 	JD = np.array(JD)
	# 	r_eci = np.array(r_eci)
	# 	if v_eci is not None:
	# 		v_eci = np.array(v_eci)
	# if len(JD.shape) == 0:
	# 	JD = np.array([JD])
	# 	r_eci = np.array([r_eci])
	# 	if v_eci is not None:
	# 		v_eci = np.array([v_eci])
	# 	scalar = 1

	R_ECF_ECI = get_R_ECI_ECF_GMST(JD, inverse=True)
	r_eci = matmul(R_ECF_ECI,r_ecf)
	if v_ecf is None:
		return r_eci
	else:
		# pole_axis = R_ECI_ECF[:,:,2]
		pole_axis = R_ECF_ECI[:,:,2] # only correct for simple spinner
		v_eci = matmul(R_ECF_ECI,v_ecf) + np.cross(W_EARTH*pole_axis,r_eci)
		return r_eci, v_eci


def lon_to_MLST(lon_MLST, JD, R_ECI_ECF=None):
	"""
	lon_MLST - lon of output MLST
	JD - scalar or vector
	outputs MLST in hours
		MLST = mean local solar time

	"""
	if R_ECI_ECF is None:
		R_ECI_ECF = get_R_ECI_ECF_GMST(JD)

	# MLST_deg = (MLST-12)/24 * 360 # deg
	s_eci = solar_pos_approx(JD)
	if type(JD) is np.ndarray:
		s_ecf_xy = matmul(R_ECI_ECF, s_eci)
	else:
		s_ecf_xy = R_ECI_ECF @ s_eci

	s_ecf_xy.T[2] = 0.0
	r_sun_xy = unit(s_ecf_xy) * R_earth
	lon_s, lat_s, _ = ecf_to_lla(r_sun_xy.T[0], r_sun_xy.T[1], r_sun_xy.T[2])

	# MLST_deg = (MLST-12)/24 * 360 # deg
	# lon_MLST = ut.wrap(lon_s + MLST_deg, radians=False)
	lon_MLST = wrap(lon_MLST) # wraps to +/-180
	MLST_deg = lon_MLST - lon_s
	MLST = np.round(MLST_deg * 24/360 + 12, 12) % 24 # hrs

	return MLST # hrs


def MLST_to_LAN(MLST, JD):
	# MLST in hours, return LAN in radians
	lon = MLST_to_lon(MLST, JD)
	GMST = get_GMST(JD) # deg
	LAN0 = np.radians(GMST+lon)
	return LAN0

def LAN_to_MLST(LAN, JD):
	GMST = get_GMST(JD) # deg
	lon_MLST = np.degrees(LAN)-GMST
	MLST = lon_to_MLST(lon_MLST, JD)
	return MLST


def MLST_to_lon(MLST, JD, R_ECI_ECF=None):
	"""
	MLST - mean local solar time (hrs)
	JD - scalar or vector
	outputs lon at which MLST occurs (deg)

	"""

	if R_ECI_ECF is None:
		R_ECI_ECF = get_R_ECI_ECF_GMST(JD)

	s_eci = solar_pos_approx(JD)
	if type(JD) is np.ndarray:
		s_ecf_xy = matmul(R_ECI_ECF, s_eci)
	else:
		s_ecf_xy = R_ECI_ECF @ s_eci

	s_ecf_xy.T[2] = 0.0
	r_sun_xy = unit(s_ecf_xy) * R_earth
	lon_s, lat_s, _ = ecf_to_lla(r_sun_xy.T[0], r_sun_xy.T[1], r_sun_xy.T[2])

	MLST_deg = (MLST-12)/24 * 360 # deg
	lon_MLST = wrap(lon_s + MLST_deg)

	return lon_MLST # deg


def beta_numerical(r, v, JD, body='sun'):

	from leocat.utils.math import dot
	h_hat = unit(np.cross(r,v))
	if body == 'sun':
		r_body = solar_pos_approx(JD)
	elif body == 'moon':
		r_body = lunar_pos_approx(JD)
	r_body_unit = unit(r_body)
	proj = dot(h_hat,r_body_unit)
	proj[proj > 1] = 1
	proj[proj < -1] = -1
	angle = np.arccos(proj) * 180/np.pi
	beta = 90 - angle
	return beta

	# h = np.cross(r, v)
	# h_unit = (h.T / np.linalg.norm(h,axis=1)).T

	# # sun vector
	# if body == 'sun':
	# 	r_sun_rgt = solar_pos_approx(JD)
	# elif body == 'moon':
	# 	r_sun_rgt = lunar_pos_approx(JD)
	# r_sun_rgt_unit = (r_sun_rgt.T / np.linalg.norm(r_sun_rgt, axis=1)).T
	# proj = np.sum(h_unit * r_sun_rgt_unit, axis=1)

	# proj[proj > 1] = 1
	# proj[proj < -1] = -1

	# angle = np.arccos(proj) * 180/np.pi
	# beta = 90 - angle

	# return beta

def beta_analytic(LAN, inc, JD, body='sun'):
	# all inputs in radians
	# output in degrees
	# Vallado v4 section 5.3
	from leocat.utils.geodesy import cart_to_RADEC

	if body == 'sun':
		r_sun = solar_pos_approx(JD)
	elif body == 'moon':
		r_sun = lunar_pos_approx(JD)
	RA, DEC = cart_to_RADEC(r_sun) # deg
	RA_rad, DEC_rad = np.radians((RA,DEC))

	beta = np.arcsin(np.cos(DEC_rad)*np.sin(inc)*np.sin(LAN-RA_rad) + np.sin(DEC_rad)*np.cos(inc))
	return np.degrees(beta)


def illum_frac(beta, elev_threshold):
	# beta and elev_threshold in deg
	arg = np.cos(np.radians(90-elev_threshold)) / np.cos(np.radians(np.abs(beta)))
	arg[arg > 1] = 1
	f = 1/np.pi * np.arccos(arg)
	return f

def point_in_view(r, u_hat):
	# shadow
	proj = np.dot(r, u_hat)
	rs = r - np.transpose([proj*u_hat[0], proj*u_hat[1], proj*u_hat[2]])
	b = (proj < 0) & (mag(rs) < R_earth)
	return ~b