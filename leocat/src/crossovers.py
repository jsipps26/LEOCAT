
import numpy as np
from scipy.interpolate import CubicSpline, LSQUnivariateSpline
import copy
from tqdm import tqdm

from leocat.utils.const import *
from leocat.utils.orbit import orb_to_tracks
from pyproj import CRS, Transformer


class Path:

	# Object with lon, lat, ECEF position, and ECEF derivatives (for batch LS)

	def __init__(self, lon, lat, t, id0):
		self.lon = lon
		self.lat = lat
		self.t = t
		self.id = id0

		self.set_projection()
		r_ecef = self.tr_lla_ecf.transform(lon, lat, np.zeros(lon.shape))
		r_ecef = np.transpose([r_ecef[0], r_ecef[1], r_ecef[2]])
		self.x = r_ecef.T[0]
		self.y = r_ecef.T[1]
		self.z = r_ecef.T[2]

		self.update()

	def set_projection(self):
		crs_lla = CRS.from_proj4(PROJ4_LLA).to_3d()
		crs_ecf = CRS.from_proj4(PROJ4_ECF).to_3d()
		tr_lla_ecf = Transformer.from_crs(crs_lla, crs_ecf)
		self.tr_lla_ecf = tr_lla_ecf

	def update(self):

		# fit ECEF position
		x, y, z, t = self.x, self.y, self.z, self.t
		cx = CubicSpline(t, x)
		cy = CubicSpline(t, y)
		cz = CubicSpline(t, z)

		# fit ECEF position derivative (velocity in ECEF)
		#	may not be robust/stable, may have noise (see below)
		dcx = cx.derivative()
		dcy = cy.derivative()
		dcz = cz.derivative()

		if 1:
			# if RGT motion is smooth, can approximate by
			# series of lines rather than cubic functions
			#	~n^2 speedup

			n = int(len(t)/10)
			tau_low = np.linspace(t[0]+1,t[-1]-1,n)

			cx = LSQUnivariateSpline(t, cx(t), tau_low, k=1)
			cy = LSQUnivariateSpline(t, cy(t), tau_low, k=1)
			cz = LSQUnivariateSpline(t, cz(t), tau_low, k=1)

			dcx = LSQUnivariateSpline(t, dcx(t), tau_low, k=1)
			dcy = LSQUnivariateSpline(t, dcy(t), tau_low, k=1)
			dcz = LSQUnivariateSpline(t, dcz(t), tau_low, k=1)

		if 0:
			tau = np.linspace(t[0]-1000, t[-1]+1000, 10000)

			fig, ax = make_fig()
			ax.plot(t, x, '.')
			ax.plot(tau, cx(tau), 'k')
			ax.plot(t, y, '.')
			ax.plot(tau, cy(tau), 'k')
			ax.plot(t, z, '.')
			ax.plot(tau, cz(tau), 'k')
			fig.show()

			fig, ax = make_fig()
			drdt = np.diff(x)/np.diff(t)
			ax.plot(t[:-1], drdt, '.')
			ax.plot(tau, dcx(tau), 'k')
			drdt = np.diff(y)/np.diff(t)
			ax.plot(t[:-1], drdt, '.')
			ax.plot(tau, dcy(tau), 'k')
			drdt = np.diff(z)/np.diff(t)
			ax.plot(t[:-1], drdt, '.')
			ax.plot(tau, dcz(tau), 'k')
			fig.show()

			pause()
			plt.close('all')

		self.cx = cx
		self.cy = cy
		self.cz = cz
		self.dcx = dcx
		self.dcy = dcy
		self.dcz = dcz



class CrossoverEstimator:
	def __init__(self, orb, num_tracks, JD1):
		self.orb = orb
		self.JD1 = JD1
		self.num_tracks = num_tracks

		self.load_tracks()
		self.set_projection()
		self.build_tracks()

	def load_tracks(self):
		orb, num_tracks, JD1 = self.orb, self.num_tracks, self.JD1
		t_track, r_track = orb_to_tracks(orb, num_tracks, JD1)
		self.t_track = t_track
		self.r_track = r_track

	def set_projection(self):
		crs_lla = CRS.from_proj4(PROJ4_LLA).to_3d()
		crs_ecf = CRS.from_proj4(PROJ4_ECF).to_3d()
		tr_lla_ecf = Transformer.from_crs(crs_lla, crs_ecf)
		self.tr_lla_ecf = tr_lla_ecf

	def build_tracks(self):

		# test_tracks = track_info
		# test_tracks = read_pickle('test_tracks.pkl') # load test track dataset
		# JD_all = test_tracks['JD']
		# rx_all = test_tracks['rx']
		# ry_all = test_tracks['ry']
		# rz_all = test_tracks['rz']

		tr_lla_ecf = self.tr_lla_ecf

		JD_all = self.t_track
		rx_all = self.r_track[:,:,0]
		ry_all = self.r_track[:,:,1]
		rz_all = self.r_track[:,:,2]

		tr_lla_ecf = self.tr_lla_ecf
		JD0 = self.JD1

		tracks = []
		# for f_num in iterator:
		for f_num in range(len(JD_all)):
			JD, rx, ry, rz = JD_all[f_num], rx_all[f_num], ry_all[f_num], rz_all[f_num]
			# tau = (JD-JD[0])*86400
			tau = JD
			lon_rgt, lat_rgt, alt_rgt = tr_lla_ecf.transform(rx, ry, rz, direction='inverse')
			# t = (JD-JD0)*86400
			t = JD

			p = Path(lon_rgt, lat_rgt, t, f_num)
			tracks.append(p)

		self.tracks = tracks

	def node_loc(self, lat, search=[0.4,0.6]):
		# defaults to descending node
		#	search = [0,0.2] or [0.8,1] for ascending
		i1 = int(len(lat)*search[0])
		i2 = int(len(lat)*search[1])
		lat_half = lat[i1:i2+1]
		j_min = np.argmin(np.abs(lat_half)) # should be at equator
		index = np.arange(len(lat))
		return index[i1:i2+1][j_min]


	def find_initial_conditions(self, debug=0):
		tracks = self.tracks

		crosstracks = {}

		match_type1 = None
		match_type2 = None
		find_index = True

		N = len(tracks)
		M = len(tracks[0].t)

		for i in range(N):
		# for i in [99]:
			# descending
			p1 = tracks[i]

			# descending node longitude
			k_min = self.node_loc(p1.lat, search=[0.4,0.6])
			lon1_eq = p1.lon[k_min]
			lat1_eq = p1.lat[k_min]
			dlat_min = np.abs(np.max(np.diff(p1.lat))/2)
			if np.abs(lat1_eq) > dlat_min:
				print('warning: lat1_eq', lat1_eq)
				if 0:
					fig, ax = make_fig()
					ax.plot(p1.lon, p1.lat, '.')
					ax.plot(lon1_eq, lat1_eq, 'rx')
					fig.show()

					pause()
					plt.close(fig)

			for j in range(N):
				if i == j:
					continue

				p2 = tracks[j]

				# L1, L2 = p2.lon[0], p2.lon[-1]
				L2, L1 = p2.lon[0], p2.lon[-1] # L1 < L2 in direction
				dL1 = (L1-lon1_eq)
				dL1 = (dL1 + 180) % 360 - 180
				dL2 = (L2-lon1_eq)
				dL2 = (dL2 + 180) % 360 - 180

				if not (dL1 <= 0 and dL2 >= 0):
					crosstype = 1
					title = 'type 1'
					index = np.linspace(0,M-1,3).astype(int)
				else:
					crosstype = 2
					title = 'type 2'
					index = np.linspace(0,M-1,5).astype(int)

				if i in crosstracks:
					crosstracks[i].append([j, crosstype, dL1, dL2])
				else:
					crosstracks[i] = [[j, crosstype, dL1, dL2]]

				if debug > 1:
					fig = plt.figure()
					ax = fig.add_subplot(111,projection='3d')
					ax.plot(p1.x, p1.y, p1.z, '.')
					ax.plot(p2.x, p2.y, p2.z, '.')
					set_axes_equal(ax)
					ax.set_title(title)
					fig.show()

					pause()
					plt.close(fig)

				# if crosstype == 2:
				# 	sys.exit()

				if find_index:

					g1 = []
					g2 = []
					for q in range(len(index)-1):
						k1, k2 = index[q], index[q+1]
						k_mid = int(np.mean([k1,k2]))
						g1.append(k_mid)
						g2.append(k_mid)

					g1, g2 = np.array((g1, g2))


					"""
					Sort g1 and g2 so that you have initial
					conditions that are close to each other

					for i in tracks:
						j_vec, guess1, guess2 = tracks[i]
						...

					"""

					match = []
					for ii, g10 in enumerate(g1):
						r1 = np.array([p1.cx(p1.t[g10]), p1.cy(p1.t[g10]), p1.cz(p1.t[g10])])
						d = []
						for jj, g20 in enumerate(g2):
							r2 = np.array([p2.cx(p2.t[g20]), p2.cy(p2.t[g20]), p2.cz(p2.t[g20])])
							d.append(np.linalg.norm(r2-r1))
						jj = np.argmin(d)
						match.append([g1[ii],g2[jj]])

					match = np.array(match)

					if crosstype == 1 and match_type1 is None:
						match_type1 = copy.deepcopy(match)
					if crosstype == 2 and match_type2 is None:
						match_type2 = copy.deepcopy(match)

					# if not (match_type1 is None and match_type2 is None):
					if not match_type1 is None and not match_type2 is None:
						find_index = False

		self.match_type1 = match_type1
		self.match_type2 = match_type2
		return crosstracks


	def f(self, beta, p1, p2):
		f0 = np.array([ p1.cx(beta[0]) - p2.cx(beta[1]),
						p1.cy(beta[0]) - p2.cy(beta[1]),
						p1.cz(beta[0]) - p2.cz(beta[1]) ])
		#
		return f0

	def J(self, beta, p1, p2):
		J0 = np.array([ [p1.dcx(beta[0]), -p2.dcx(beta[1])],
						[p1.dcy(beta[0]), -p2.dcy(beta[1])],
						[p1.dcz(beta[0]), -p2.dcz(beta[1])] ])
		#
		return J0

	#

	def find_crossovers(self, crosstracks, verbose=2):

		tracks = self.tracks
		tr_lla_ecf = self.tr_lla_ecf

		match_type1 = self.match_type1
		match_type2 = self.match_type2


		N = len(tracks)
		M = len(tracks[0].t)

		cross_data = {}
		iterator = range(N)
		if verbose > 0:
			print('finding crossovers..')
		if verbose > 1:
			iterator = tqdm(range(N))

		for i in iterator:
		# for i in tqdm(range(N)):
		# for i in range(N):
			C = np.array(crosstracks[i]).astype(int)
			p1 = tracks[i]

			for vec in C:
				j, crosstype, dL1, dL2 = vec
				p2 = tracks[j]
				match = match_type1
				alpha = 1
				max_iter = 10
				if crosstype == 2:
					match = match_type2

					d_rev_deg = np.abs(p1.lon[-1] - p1.lon[0])
					z1 = np.abs(dL1)/d_rev_deg
					z2 = np.abs(dL2)/d_rev_deg
					z = np.min([z1,z2])

					# modify initial guess by how small dL1 or dL2 are
					factor = 2*z
					m0 = int((M-1)/4/2)
					m0_z = int(factor * m0)
					dm = 0 # m0_z - m0
					match = np.array([ [m0_z, match[0,1]+dm],
									[match[1,0]+dm, m0_z],
									[match[2,0]+dm,M-1-m0_z],
									[M-1-m0_z,match[3,1]+dm] ])
					#

					# alpha = np.max([0.1,2*z])
					# val = int(10/alpha)*2
					# max_iter = val

				err = 0
				cross = []
				for m in match:
					m1, m2 = m
					tg1, tg2 = float(p1.t[m1]), float(p2.t[m2])
					beta = np.array([tg1, tg2])
					tol = 1e-3
					k = 0
					while k < max_iter:
						J0 = self.J(beta, p1, p2)
						f0 = self.f(beta, p1, p2)
						dbeta = np.linalg.inv(J0.T @ J0) @ J0.T @ (-f0)
						beta = beta + alpha*dbeta
						k += 1
						if np.linalg.norm(dbeta) < tol or k == max_iter:
							if k == max_iter:
								print('warning: max_iter', i, j)
								err = 1
							break

					c0 = np.array([p1.cx(beta[0]), p1.cy(beta[0]), p1.cz(beta[0])])
					t_c = beta[0]

					# lon_c, lat_c, _ = cm_ut.ecf_to_lla(*c0)
					lon_c, lat_c, _ = tr_lla_ecf.transform(c0[0], c0[1], c0[2], direction='inverse')


					if i in cross_data:
						if j in cross_data[i]:
							cross_data[i][j].append([t_c, lon_c, lat_c])
						else:
							cross_data[i][j] = [[t_c, lon_c, lat_c]]
					else:
						cross_data[i] = {}
						cross_data[i][j] = [[t_c, lon_c, lat_c]]

					# key = (i,j)
					# if key in cross_data:
					# 	cross_data[key].append([t_c, lon_c, lat_c])
					# else:
					# 	cross_data[key] = [[t_c, lon_c, lat_c]]

					cross.append(c0)

				cross = np.array(cross)

				if err:
					fig = plt.figure()
					ax = fig.add_subplot(111, projection='3d')
					ax.plot(p1.x, p1.y, p1.z, '.')
					ax.plot(p2.x, p2.y, p2.z, '.')
					for c in cross:
						ax.plot(c[0], c[1], c[2], 'rx')

					set_axes_equal(ax)
					for m in match:
						m1, m2 = m
						tg1, tg2 = p1.t[m1], p2.t[m2]
						c1 = np.array([p1.cx(tg1), p1.cy(tg1), p1.cz(tg1)])
						c2 = np.array([p2.cx(tg2), p2.cy(tg2), p2.cz(tg2)])
						dc = c2-c1
						ax.plot(c1[0], c1[1], c1[2], 'b.')
						ax.plot(c2[0], c2[1], c2[2], 'b.')
						ax.plot([c1[0], c1[0]+dc[0]], 
								[c1[1], c1[1]+dc[1]],
								[c1[2], c1[2]+dc[2]], 'k')
						#

					ax.set_title('%d\n%d, %d' % (max_iter, i,j))
					fig.show()

					pause()
					plt.close(fig)


		#
		return cross_data


	def get_access(self, cross_data, rounding_precision=3):

		tr_lla_ecf = self.tr_lla_ecf
		# JD_all = self.t_track
		# rx_all = self.r_track[:,:,0]
		# ry_all = self.r_track[:,:,1]
		# rz_all = self.r_track[:,:,2]
		tracks = self.tracks
		N = len(tracks)

		t_data = {}
		for j in range(N):
			for i in range(j):
				c_ij = np.array(cross_data[i][j])
				c_ji = np.array(cross_data[j][i])
				t_ij = c_ij.T[0]
				t_ji = c_ji.T[0]

				pos_ij = c_ij[:,1:]
				pos_ji = c_ji[:,1:]

				# sort by longitude
				# not robust, can fail if crossovers are
				# estimated within round-off of each other
				ij_idx = pos_ij[:,0].argsort()
				ji_idx = pos_ji[:,0].argsort()
				# ij_idx = np.arange(len(pos_ij))
				# ji_idx = np.arange(len(pos_ji))

				dp = np.max(np.abs(pos_ij[ij_idx]-pos_ji[ji_idx]))
				if dp > 1e-6:
					print(dp)
					pause()

				t1 = t_ij[ij_idx]
				t2 = t_ji[ji_idx]
				# dt0 = t_ji[ji_idx]-t_ij[ij_idx]
				pos0 = pos_ij[ij_idx]
				for k in range(len(pos0)):
					pos0_rnd = np.round(pos0[k],rounding_precision)
					key = (pos0_rnd[0],pos0_rnd[1])
					if not (key in t_data):
						t_data[key] = []
					t_data[key].append([t1[k], t2[k]])

		keys = list(t_data.keys())
		L = []
		t1_vec, t2_vec = [], []
		lon_vec = []
		lat_vec = []
		for key in keys:
			L.append(len(t_data[key]))
			t1_vec.append(t_data[key][0][0])
			t2_vec.append(t_data[key][0][1])
			lon_vec.append(key[0])
			lat_vec.append(key[1])
		L = np.array(L)
		if (L > 1).any():
			# only 1 crossover per location
			# cannot have multiple times at a single crossover
			print('error: length > 1')
		t1_vec = np.array(t1_vec)
		t2_vec = np.array(t2_vec)
		lon_vec = np.array(lon_vec)
		lat_vec = np.array(lat_vec)

		return lon_vec, lat_vec, t1_vec, t2_vec



