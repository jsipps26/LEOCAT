
import numpy as np

from leocat.utils.const import *

from pyproj import CRS, Transformer
from leocat.utils.geodesy import DiscreteGlobalGrid, ellipsoidal_dist
from leocat.utils.cov import project_pos, get_access_interval, SC_frame
from leocat.utils.orbit import convert_ECI_ECF
from leocat.utils.math import mag
from scipy.interpolate import CubicSpline
# import leocat.src.coverage.mesh as cov_mesh
import leocat.src.mesh as cov_mesh

from leocat.utils.index import cantor_pairing, unique_index, hash_cr_DGG, hash_xy_DGG
from pandas import DataFrame

class Instrument:
	def __init__(self, FOV_CT, FOV_AT=None):
		is_rectangular = True
		if FOV_AT is None:
			is_rectangular = False
		self.is_rectangular = is_rectangular

		self.FOV_CT = FOV_CT
		self.FOV_AT = FOV_AT

	def get_FOV_geometry(self):
		is_rectangular = self.is_rectangular
		FOV_AT, FOV_CT = self.FOV_AT, self.FOV_CT # in deg
		if is_rectangular:
			"""
			"Swath level"
			Increase swath envelope fidelity
			level 1 - circular/max bound
			level 2 - tighter bound to footprint shape
			level 3 - polygon intersections (not implemented)

			Up-front slightly increased runtime, but
			overall removes ~25% of unnecessary mesh
			and grid points, which, with high density
			gridding, is a signficant change
				ideally is sqrt(2)/2 faster for rectangular

			Can be less effective with periodic off-nadir
			maneuvering

			"""
			gamma_x = np.radians(FOV_AT)
			gamma_y = np.radians(FOV_CT)
			fp_geom = np.array([[0,0],[gamma_x,0],[gamma_x,gamma_y],[0,gamma_y]]) # no end-pt

		else:
			# circular
			#	not done the best
			#	could do lvl 1 without knowing fp_geom, b/c circular
			gamma_y = np.radians(FOV_CT)
			angle = np.linspace(0,2*np.pi,25)[:-1] # no end-pt
			fp_geom_x = np.arctan(np.tan(gamma_y/2)*np.cos(angle))
			fp_geom_y = np.arctan(np.tan(gamma_y/2)*np.sin(angle))
			fp_geom = np.transpose([fp_geom_x,fp_geom_y])

		fp_geom = fp_geom - np.mean(fp_geom,axis=0)
		return fp_geom


class SwathEnvelope:
	"""
	Problems
	JD1 tied to orbit, not given JD1
	must modify t_orb I think

	"""
	def __init__(self, orb, Inst, res, JD1, JD2):
		self.orb = orb
		self.Inst = Inst
		self.space_params = {'res': res}
		self.time_params = {'JD1': JD1, 'JD2': JD2}

		self.set_space_params()
		self.propagate_ground_track()

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
		Inst = self.Inst

		T_sim = (JD2-JD1)*86400
		N_orb = int(T_sim/dt_orb)+1
		t_orb = np.linspace(0,T_sim,N_orb)
		dt_orb = t_orb[1]-t_orb[0]
		JD_orb = t_orb/86400 + JD1

		r_eci_sc, v_eci_sc = orb.propagate(t_orb)
		r_ecf_sc, v_ecf_sc = convert_ECI_ECF(JD_orb, r_eci_sc, v_eci_sc)

		LVLH = True
		R_FOV = SC_frame(r_ecf_sc, v_ecf_sc, off_nadir_C_vec, off_nadir_A_vec, LVLH=LVLH) # Nx3x3
		fp_geom = Inst.get_FOV_geometry()
		swath_level = 1
		if Inst.is_rectangular:
			swath_level = 2
		w_l_ecf, w_r_ecf = cov_mesh.get_swath_envelope(t_orb, r_ecf_sc, R_FOV, fp_geom, level=swath_level)
		w_ev = ellipsoidal_dist(w_l_ecf, w_r_ecf, tr_lla_ecf)

		self.w_l_ecf = w_l_ecf
		self.w_r_ecf = w_r_ecf
		self.w_ev = w_ev

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

		# self.cr = cr
		# self.cv = cv
		self.v_ecf_mag_max = np.max(mag(v_ecf))

		self.time_params['T_sim'] = T_sim
		self.time_params['dt_orb'] = dt_orb
		self.time_params['t_orb'] = t_orb


	def get_lonlat(self, res_factor_CT=0.25, res_factor_AT=0.25, spherical=False):
		# res_factor_CT = 0.25
		# res_factor_AT = 0.25

		t_orb = self.time_params['t_orb']
		T_sim = self.time_params['T_sim']
		w_l_ecf, w_r_ecf = self.w_l_ecf, self.w_r_ecf
		w_ev = self.w_ev
		tr_lla_ecf = self.space_params['tr_lla_ecf']
		DGG = self.space_params['DGG']
		v_ecf_mag_max = self.v_ecf_mag_max
		res = self.space_params['res']

		cw_left = CubicSpline(t_orb, w_l_ecf)
		cw_right = CubicSpline(t_orb, w_r_ecf)

		delta_t = res_factor_AT*res/v_ecf_mag_max

		# t_mesh_min, t_mesh_max = np.min(t), np.max(t)
		t_mesh_min, t_mesh_max = 0.0, T_sim
		Nt = int((t_mesh_max-t_mesh_min)/delta_t) + 1
		t_intp = np.linspace(t_mesh_min,t_mesh_max,Nt)
		mesh_edge_l = cw_left(t_intp)
		mesh_edge_r = cw_right(t_intp)

		dz = res*res_factor_CT
		w_max = np.max(w_ev)
		Nz = int(w_max/dz)+2
		tau_c = np.linspace(0,1,Nz)

		# vectorize both axes
		V1 = mesh_edge_r
		V2 = mesh_edge_l
		dV = V2-V1
		Mx = np.outer(dV.T[0],tau_c) # all combinations
		Mx = (Mx.T + V1.T[0]).T
		My = np.outer(dV.T[1],tau_c)
		My = (My.T + V1.T[1]).T
		Mz = np.outer(dV.T[2],tau_c)
		Mz = (Mz.T + V1.T[2]).T

		r_mesh = np.transpose([Mx.flatten(),
							My.flatten(),
							Mz.flatten()])
		#
		# index_mesh = np.tile(np.arange(len(t_intp)), (len(tau_c),1)).T.flatten()
		"""
		Cannot index back to t_intp with multiple revs
		Must classify revolution number prior
			makes problems at start/end of revs, as was
			dealt with via merging, which produces terrible
			algorithm process

		This technique -can- find unique lon/lats covered,
		which is still valuable since those are not known
		a priori for SimpleCoverage/AnalyticCoverage.

		There is potential to just apply some NR-based DCA
		on mesh points instead of GPs, then classify revs
		by a time threshold... but it is not robust, and
		would likely not be any faster than just doing 
		AnalyticCoverage and refining that thru VFPD.

		"""

		xm, ym, _ = tr_lla_ecf.transform(r_mesh.T[0], r_mesh.T[1], r_mesh.T[2], direction='inverse')
		cols, rows = hash_cr_DGG(xm, ym, DGG)

		cols_adj, rows_adj = cols.min()-1, rows.min()-1
		cols_nat = cols-cols_adj
		rows_nat = rows-rows_adj
		cp = cantor_pairing(cols_nat,rows_nat)
		cp_index = unique_index(cp)
		c_uq, r_uq = cols[cp_index], rows[cp_index]

		lon, lat = hash_xy_DGG(c_uq, r_uq, DGG)
		# if spherical:
		# 	phi = np.radians(lat)
		# 	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
		# 	lat_c = np.degrees(phi_c)
		# 	lat = lat_c

		return lon, lat


