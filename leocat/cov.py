
import numpy as np

from leocat.utils.const import *
from leocat.src.simple_cov import SimpleCoverage
from leocat.src.bt import AnalyticCoverage
from leocat.src.ssc import SmallSwathCoverage
from leocat.src.swath import SwathEnvelope
from leocat.src.fpt import Satellite, Instrument # WIP

from leocat.utils.cov import get_num_obs, get_revisit


def vector_to_access(t_total, index):
	from pandas import DataFrame
	df = DataFrame({'index': index})
	index_indices = df.groupby('index',sort=False).indices
	t_access = {}
	for key in index_indices:
		idx = index_indices[key]
		t_access[key] = t_total[idx]
	return t_access


def access_to_vector(t_access):
	t_total = []
	index = []
	for key in t_access:
		tau = t_access[key]
		t_total.append(tau)
		index.append(np.full(tau.shape, key))
	t_total = np.concatenate(t_total)
	index = np.concatenate(index)
	return t_total, index


def combine_coverage(lons, lats, t_access_list, DGG):
	"""
	Assuming all lons/lats are on the same DGG,
	combine into one t_access
		Re-indexes the final t_access to reflect
		lon/lat_total

	"""
	from leocat.utils.index import hash_cr_DGG, hash_xy_DGG

	t_access_cr = {}
	for i in range(len(lons)):
		lon, lat = lons[i], lats[i]
		t_access = t_access_list[i]
		cols, rows = hash_cr_DGG(lon, lat, DGG)
		for key in t_access:
			c, r = cols[key], rows[key]
			if not ((c,r) in t_access_cr):
				t_access_cr[(c,r)] = []
			t_access_cr[(c,r)].append(t_access[key])

	cr = np.array(list(t_access_cr.keys()))
	lon_total, lat_total = hash_xy_DGG(cr.T[0], cr.T[1], DGG)

	t_access_total = {}
	for j,(c,r) in enumerate(t_access_cr):
		t_access_total[j] = np.sort(np.concatenate(t_access_cr[(c,r)]))

	# for (c,r) in t_access_cr:
	# 	t_access_cr[(c,r)] = np.sort(np.concatenate(t_access_cr[(c,r)]))
	# cr = np.array(list(t_access_cr.keys()))
	# lon_total, lat_total = hash_xy_DGG(cr.T[0], cr.T[1], DGG)

	return lon_total, lat_total, t_access_total


def get_coverage(orb, swath, JD1, JD2, verbose=2, res=None, alpha=0.25, lon=None, lat=None):

	from leocat.src.bt import AnalyticCoverage

	simulation_period = (JD2-JD1)*86400
	Tn = orb.get_period('nodal')
	num_revs = simulation_period / Tn
	if res is None:
		res = alpha*swath # dx = alpha*w as in EGPA

	C = 2*np.pi*R_earth
	area_cov = num_revs * swath * C
	A_earth = 4*np.pi*R_earth**2

	lonlat_exists = False
	if not (lon is None) and not (lat is None):
		lonlat_exists = True

	if not lonlat_exists:
		if area_cov < A_earth:
			# If less duplicate area covered than A_earth,
			# find lon/lats directly
			from leocat.src.swath import Instrument, SwathEnvelope
			from leocat.utils.cov import swath_to_FOV
			if verbose > 1:
				print('preprocessing lon/lats..')
			FOV_CT = swath_to_FOV(swath, alt=orb.get_alt(), radians=False)
			Inst = Instrument(FOV_CT)
			SE = SwathEnvelope(orb, Inst, res, JD1, JD2)
			lon, lat = SE.get_lonlat()

		else:
			from leocat.utils.geodesy import DiscreteGlobalGrid
			# if more, use global grid
			if verbose > 1:
				print('using global lon/lats..')
			DGG = DiscreteGlobalGrid(A=res**2)
			lon, lat = DGG.get_lonlat()

	# spherical lat for AC
	phi = np.radians(lat)
	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
	lat_c = np.degrees(phi_c)
	AC = AnalyticCoverage(orb, swath, lon, lat_c, JD1, JD2)
	t_access = AC.get_access(verbose=verbose, accuracy=2)


	return lon, lat, t_access



