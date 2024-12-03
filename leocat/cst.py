

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

	def get_access(self, verbose=2, approx=True, res=None, alpha=0.25, lon=None, lat=None):

		JD1, JD2 = self.JD1, self.JD2
		orb = self.orb
		swath = self.swath
		LAN_shifts = self.LAN_shifts
		nu_shifts = self.nu_shifts

		if res is None:
			res = alpha*swath

		CST = {}
		if approx:
			DGG = DiscreteGlobalGrid(A=res**2)
			Tn = orb.get_period('nodal')
			JD1_buffer = JD1 - Tn/86400
			JD2_buffer = JD2 + Tn/86400
			orb_buf = deepcopy(orb)
			orb_buf.propagate_epoch((JD1_buffer-JD1)*86400, reset_epoch=True)
			lon, lat, t_access_buf = get_coverage(orb_buf, swath, JD1_buffer, JD2_buffer, \
												res=res, verbose=verbose, lon=lon, lat=lat)
			#

			t1_epoch = (JD1-JD1_buffer)*86400
			t2_epoch = (JD2-JD1_buffer)*86400
			t_access = trim_time(t_access_buf, t1_epoch, t2_epoch, time_shift=-t1_epoch)

			CST[0] = {'lon': lon, 'lat': lat, 't_access': t_access, 'orb': orb}
			for i in range(len(LAN_shifts)):
				LAN_shift = LAN_shifts[i]
				nu_shift = nu_shifts[i]
				lon_cst, lat_cst, t_access_cst, orb_cst = \
					shift_nu(nu_shift, lon, lat, t_access_buf, orb, JD1, JD2, JD1_buffer, \
								DGG=DGG, LAN_shift=LAN_shift)
				#
				CST[i+1] = {'lon': lon_cst, 'lat': lat_cst, 't_access': t_access_cst, 'orb': orb_cst}

		else:
			# run true simulations for each constellation member
			lon, lat, t_access = get_coverage(orb, swath, JD1, JD2, \
											res=res, verbose=verbose, lon=lon, lat=lat)
			#
			CST[0] = {'lon': lon, 'lat': lat, 't_access': t_access, 'orb': orb}
			for i in range(len(LAN_shifts)):
				LAN_shift = LAN_shifts[i]
				nu_shift = nu_shifts[i]
				orb_cst = deepcopy(orb)
				nu_new = (orb.nu + rad(nu_shift)) % (2*np.pi)
				orb_cst.set_OEs(orb.a, orb.e, orb.inc, orb.LAN, orb.omega, nu_new)
				if LAN_shift != 0.0:
					# shift_LAN_orbit compensates for changes in MLST, if any
					orb_cst = shift_LAN_orbit(LAN_shift, orb_cst)

				lon_cst, lat_cst, t_access_cst = get_coverage(orb_cst, swath, JD1, JD2, res=res, verbose=verbose, lon=lon, lat=lat)
				CST[i+1] = {'lon': lon_cst, 'lat': lat_cst, 't_access': t_access_cst, 'orb': orb_cst}


		return CST





