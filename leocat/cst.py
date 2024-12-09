

import numpy as np
from leocat.utils.const import *
from leocat.utils.geodesy import DiscreteGlobalGrid
from leocat.utils.math import rad
from leocat.cov import get_coverage
from leocat.post import shift_nu, shift_LAN_orbit, trim_time

from copy import deepcopy


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





