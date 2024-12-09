

import numpy as np
from leocat.utils.const import *
from leocat.utils.geodesy import DiscreteGlobalGrid
from leocat.utils.index import hash_cr_DGG
from leocat.utils.math import rad
from leocat.cov import get_coverage
from leocat.post import shift_nu, shift_LAN_orbit, trim_time, \
						get_shift_dt_nu, shift_LAN
#
from copy import deepcopy

from tqdm import tqdm


class ConstellationShell:
	def __init__(self, orb, swath, JD1, JD2, LAN_shifts=[], nu_shifts=[]):
		self.JD1 = JD1
		self.JD2 = JD2
		self.orb = orb
		self.swath = swath
		self.LAN_shifts = LAN_shifts
		self.nu_shifts = nu_shifts

	def get_access(self, verbose=2, approx=True, res=None, alpha=0.25, lon=None, lat=None, \
					fix_noise=True):
		#
		"""
		Assuming you always want snap-to-grid

		if nothing is input:
			goes to a default res
			builds DGG

		elif res has value, and lon/lat are None:
			build DGG using given res

		elif res is None, and lon/lat have value:
			error
			lon/lat must be assumed to be on the grid
			If not, there's no DGG which is fine on
			its own, but if you always want snap-to-grid,
			you -must- have a res or no lla input.

		"""

		JD1, JD2 = self.JD1, self.JD2
		orb = self.orb
		swath = self.swath
		LAN_shifts = self.LAN_shifts
		nu_shifts = self.nu_shifts

		res_exists = res is not None
		lonlat_exists = (lon is not None) and (lat is not None)

		if res_exists and lonlat_exists:
			DGG = DiscreteGlobalGrid(A=res**2)
		elif res_exists and not lonlat_exists:
			DGG = DiscreteGlobalGrid(A=res**2)
		elif not res_exists and lonlat_exists:
			if approx:
				raise Exception('Cannot snap to grid without res specified.')
		elif not res_exists and not lonlat_exists:
			res = alpha*swath
			if res > 100:
				res = 100
			DGG = DiscreteGlobalGrid(A=res**2)

		if ((lon is None) and not (lat is None)) or \
			(not (lon is None) and (lat is None)):
			raise Exception('If inputting lon/lat, must input both lon/lat.')


		CST = {}
		if approx:

			"""
			In approx, two options: shift longitudes or shift keys
			1. Shift longitudes
			Simpler option - given dLAN/dnu, can determine what longitudes
			a copy of the primary satellite would cover with different OEs
			- ideal for global coverage

			2. Shift keys
			More complicated - given dLAN/dnu, determine how keys in the
			primary satellites' access dictionary should change such that
			they point to the same lon/lats for all satellites
			- ideal for regional, point, or lines of lon/lat

			"""

			if not lonlat_exists:
				w_approx = swath
				w_true = None
				if fix_noise:
					# w_approx = swath + 2*res
					w_approx = swath + res*np.sqrt(2)
					w_true = swath

				# DGG = DiscreteGlobalGrid(A=res**2)
				Tn = orb.get_period('nodal')
				JD1_buffer = JD1 - Tn/86400
				JD2_buffer = JD2 + Tn/86400
				orb_buf = deepcopy(orb)
				orb_buf.propagate_epoch((JD1_buffer-JD1)*86400, reset_epoch=True)
				lon_buf, lat_buf, t_access_buf = get_coverage(orb_buf, w_approx, JD1_buffer, JD2_buffer, \
													res=res, verbose=verbose, lon=lon, lat=lat)
				#

				t1_epoch = (JD1-JD1_buffer)*86400
				t2_epoch = (JD2-JD1_buffer)*86400
				t_access = trim_time(t_access_buf, t1_epoch, t2_epoch, time_shift=-t1_epoch)

				CST[0] = {'lon': lon_buf, 'lat': lat_buf, 't_access': t_access, \
							'orb': orb, 'LAN_shift': 0.0, 'nu_shift': 0.0}
				#
				for i in range(len(LAN_shifts)):
					LAN_shift = LAN_shifts[i]
					nu_shift = nu_shifts[i]
					lon_cst, lat_cst, t_access_cst, orb_cst = \
						shift_nu(nu_shift, lon_buf, lat_buf, t_access_buf, orb, JD1, JD2, JD1_buffer, \
									DGG=DGG, LAN_shift=LAN_shift, w_true=w_true)
					#
					CST[i+1] = {'lon': lon_cst, 'lat': lat_cst, 't_access': t_access_cst, \
								'orb': orb_cst, 'LAN_shift': LAN_shift, 'nu_shift': nu_shift}
					#

			else:
				"""
				ROI given
					must have res/DGG b/c need to make longitudes
					for latitude extents

				Ideas
				Maybe can use lon/lat original instead of lon/lat_cst
				b/c you're just shifting and then unshifting
					The meaningful difference comes from t_access indexing,
					not from lon/lat shifts
					Shifting twice increases noise regardless
					Not shifting at all will produce zero noise, but keys
					can still be offset, effectively making noise

				First sat should have space trimmed to ROI
				Can trim space via cr_to_key or re-run get_coverage
					Trimming space probably more efficient, and
					will line up with original data everything is
					copied from which makes results less likely to 
					be inconsistent

					Could maybe do intersect_2d since t_access is
					literally a subset of t_access_buf, can find
					intersection of lon/lat_extents and lon/lat, 
					although you'd still need to iterate over the
					lon/lats of the ROI.. maybe wouldn't be any
					faster.

				Likely do not need to input DGG into LAN_shift b/c 
				lon_preshift is already hashed in hash_cr_DGG...
					I think you do need DGG; making dc_shift, you
					only shift 1 line of latitude at a time by a 
					constant. If input lon is on a DGG, output
					shift must be on the DGG. Otherwise, using 
					hash_cr_DGG, you can get holes in swath b/c
					floating shift is not on-grid.

				May be a way to index cr_to_key without set intersection
				(effectively).. although I think you'd always need to
				iterate thru points in the ROI, which is likely more
				expensive than the tuple stuff is anyway.

				Could use re-indexing as an opportunity to index
				w.r.t. cantor pairing, then you remove the need to
				redo intersection calcs after processing

				"""

				w_approx = swath
				w_true = None
				if fix_noise:
					# w_approx = swath + 2*res
					w_approx = swath + res*np.sqrt(2)
					w_true = swath

				lon_extents, lat_extents = DGG.get_lonlat()
				dlat = DGG.dlat
				lat_min, lat_max = np.min(lat)-dlat/2, np.max(lat)+dlat/2
				by = (lat_min < lat_extents) & (lat_extents < lat_max)
				lon_extents, lat_extents = lon_extents[by], lat_extents[by]

				# DGG = DiscreteGlobalGrid(A=res**2)
				Tn = orb.get_period('nodal')
				JD1_buffer = JD1 - Tn/86400
				JD2_buffer = JD2 + Tn/86400
				orb_buf = deepcopy(orb)
				orb_buf.propagate_epoch((JD1_buffer-JD1)*86400, reset_epoch=True)
				lon_buf, lat_buf, t_access_buf = get_coverage(orb_buf, w_approx, JD1_buffer, JD2_buffer, \
													res=res, verbose=verbose, lon=lon_extents, lat=lat_extents)
				#
				t1_epoch = (JD1-JD1_buffer)*86400
				t2_epoch = (JD2-JD1_buffer)*86400
				# t_access = trim_time(t_access_buf, t1_epoch, t2_epoch, time_shift=-t1_epoch)
				# possible trim_space here for t_access
				#	otherwise t_access/lon/lat_buf are for extents region, not ROI
				#	could just re-run with true time/space for first sat

				# CST[0] = {'lon': lon_buf, 'lat': lat_buf, 't_access': t_access, \
				# 			'orb': orb, 'LAN_shift': 0.0, 'nu_shift': 0.0}
				# #
				# _, _, t_access = get_coverage(orb, w_approx, JD1, JD2, \
				# 									res=res, verbose=verbose, lon=lon, lat=lat)
				# #
				# CST[0] = {'lon': lon, 'lat': lat, 't_access': t_access, \
				# 			'orb': orb, 'LAN_shift': 0.0, 'nu_shift': 0.0}
				# #

				keys = np.array(list(t_access_buf.keys()))
				cols, rows = hash_cr_DGG(lon_extents[keys], lat_extents[keys], DGG)
				cr_tuple = tuple(zip(tuple(cols),tuple(rows)))
				cr_to_key = dict(zip(cr_tuple,keys))


				cols1, rows1 = hash_cr_DGG(lon, lat, DGG)
				t_access_subset = {}
				for j in range(len(cols1)):
					cr0 = (cols1[j], rows1[j])
					if cr0 in cr_to_key:
						key = cr_to_key[cr0]
						t_access_subset[j] = t_access_buf[key]
				#
				t_access = trim_time(t_access_subset, t1_epoch, t2_epoch, time_shift=-t1_epoch)

				CST[0] = {'lon': lon, 'lat': lat, 't_access': t_access, \
							'orb': orb, 'LAN_shift': 0.0, 'nu_shift': 0.0}
				#

				iterator = range(len(LAN_shifts))
				if verbose:
					iterator = tqdm(iterator)

				# for i in range(len(LAN_shifts)):
				for i in iterator:
					LAN_shift = LAN_shifts[i]
					nu_shift = nu_shifts[i]
					dt_nu = get_shift_dt_nu(nu_shift, orb)
					LAN_shift_nu_rad = -orb.get_LAN_dot()*dt_nu + W_EARTH*dt_nu # rad
					LAN_shift_nu = np.degrees(LAN_shift_nu_rad)
					lon_preshift = shift_LAN(-(LAN_shift + LAN_shift_nu), lon, lat, DGG=DGG, orb=None)

					cols1, rows1 = hash_cr_DGG(lon_preshift, lat, DGG)
					t_access_subset = {}
					for j in range(len(cols1)):
						cr0 = (cols1[j], rows1[j])
						if cr0 in cr_to_key:
							key = cr_to_key[cr0]
							t_access_subset[j] = t_access_buf[key]
					#
					_, _, t_access_cst, orb_cst = \
						shift_nu(nu_shift, lon_preshift, lat, t_access_subset, orb, JD1, JD2, JD1_buffer, \
									DGG=DGG, LAN_shift=LAN_shift, w_true=w_true)
					#
					lon_cst, lat_cst = lon, lat
					CST[i+1] = {'lon': lon_cst, 'lat': lat_cst, 't_access': t_access_cst, \
								'orb': orb_cst, 'LAN_shift': LAN_shift, 'nu_shift': nu_shift}
					#


		else:
			# run true simulations for each constellation member
			lon0, lat0, t_access = get_coverage(orb, swath, JD1, JD2, \
											res=res, verbose=verbose, lon=lon, lat=lat)
			#
			CST[0] = {'lon': lon0, 'lat': lat0, 't_access': t_access, \
						'orb': orb, 'LAN_shift': 0.0, 'nu_shift': 0.0}
			#
			for i in range(len(LAN_shifts)):
				LAN_shift = LAN_shifts[i]
				nu_shift = nu_shifts[i]
				orb_cst = deepcopy(orb)
				nu_new = (orb.nu + rad(nu_shift)) % (2*np.pi)
				orb_cst.set_OEs(orb.a, orb.e, orb.inc, orb.LAN, orb.omega, nu_new)
				if LAN_shift != 0.0:
					# shift_LAN_orbit compensates for changes in MLST, if any
					orb_cst = shift_LAN_orbit(LAN_shift, orb_cst)

				lon_cst, lat_cst, t_access_cst = \
					get_coverage(orb_cst, swath, JD1, JD2, \
								res=res, verbose=verbose, lon=lon, lat=lat)
				#
				CST[i+1] = {'lon': lon_cst, 'lat': lat_cst, 't_access': t_access_cst, \
							'orb': orb_cst, 'LAN_shift': LAN_shift, 'nu_shift': nu_shift}
				#


		return CST





