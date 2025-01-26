
import numpy as np

from leocat.utils.const import *
from leocat.src.simple_cov import SimpleCoverage
from leocat.src.bt import AnalyticCoverage
from leocat.src.ssc import SmallSwathCoverage
from leocat.src.swath import SwathEnvelope
from leocat.src.fpt import Satellite, Instrument # WIP

from leocat.utils.cov import get_num_obs, get_revisit



def vector_to_access(vector, index):
	"""
	Convert vectorized access time (or other quality) into
	dictionary format.

	:param vector: Vectorized access time
	:type vector: numpy.ndarray[float]
	:param index: Indices of access time (matching to dict keys)
	:type index: numpy.ndarray[int]

	:return: Access time/quality in dictionary format
	:rtype: dict[int: numpy.ndarray[float]]

	"""

	from pandas import DataFrame
	df = DataFrame({'index': index})
	index_indices = df.groupby('index',sort=False).indices
	access = {}
	for key in index_indices:
		idx = index_indices[key]
		access[key] = vector[idx]
	return access


def access_to_vector(access):
	"""
	Convert dictionary format access time (or other quality) into
	vectorized format.

	:param access: Dictionary of access times/quality
	:type access: dict[int: numpy.ndarray[float]]
	:return: 
		- **vector (numpy.ndarray[float])** - Access time/quality values
		- **index (numpy.ndarray[int])** - Access key/indices
	:rtype: tuple(numpy.ndarray, numpy.ndarray)

	"""

	vector = []
	index = []
	for key in access:
		tau = access[key]
		vector.append(tau)
		index.append(np.full(tau.shape, key))
	vector = np.concatenate(vector)
	index = np.concatenate(index)
	return vector, index


def combine_coverage(lons, lats, t_access_list, DGG):
	"""
	Combine access time from multiple coverage calculations
	(e.g. from multiple satellites). This is useful for aggregation
	access times of individual satellites to make access times 
	for an entire constellation.

	:param lons: List of longitudes of each access calculation.
	:type lons: list[numpy.ndarray[float]]
	:param lats: List of latitudes of each access calculation.
	:type lats: list[numpy.ndarray[float]]
	:param t_access_list: List of accesses for each calculation.
	:type t_access_list: list[dict[int: numpy.ndarray[float]]]
	:param DGG: Discrete global grid object specifying spatial grid.
	:type DGG: DiscreteGlobalGrid

	:return: 
		- **lon_total (numpy.ndarray[float])** - Longitudes accessed by any access element
		- **lat_total (numpy.ndarray[float])** - Latitudes accessed by any access element
		- **t_access_total (dict[int: numpy.ndarray[float]])** - Combined access dictionary
	:rtype: tuple(numpy.ndarray, numpy.ndarray, dict)

	Additional Notes
	-----------------
	
	All lons/lats must be on the same DGG.

	This function re-indexes the final access dictionary to reflect
	the subset of lons/lats that cover each other.

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

	"""
	This function simplifies coverage calculation to a single routine,
	assuming nadir-pointing observation. Grid resolution is assumed 
	based on swath size unless specified (see Additional Notes).
	
	**Required Parameters**

	:param orb: Orbit of satellite.
	:type orb: Orbit
	:param swath: Swath size (km)
	:type swath: float
	:param JD1: Starting Julian date of simulation.
	:type JD1: float
	:param JD2: Ending Julian date of simulation.
	:type JD2: float

	**Optional Parameters**

	:param verbose: Set verbosity, 0, 1, or 2
	:type verbose: int
	:param res: Grid resolution (km)
	:type res: float
	:param alpha: If res and/or lon/lat unspecified, res = alpha*swath, default value: 0.25, typical alpha ~ 0.25 to 0.1
	:type alpha: float
	:param lon: Longitude of coverage grid 
					(not necessarily on DGG)
	:type lon: numpy.ndarray[float]
	:param lat: Latitude of coverage grid 
					(not necessarily on DGG)

	:return: 
		- **lon (numpy.ndarray[float])** - 
			Longitudes in spatial extent (not necessarily accessed)
		- **lat (numpy.ndarray[float])** - 
			Latitudes in spatial extent (not necessarily accessed)
		- **t_access (dict[int: numpy.ndarray[float]])** - 
			Access times dictionary for this satellite
	:rtype: tuple(numpy.ndarray, numpy.ndarray, dict)


	Additional Notes
	-----------------
	
	Assuming circular orbits only. Otherwise, use SimpleCoverage.

	lon and lat must both be specified or neither be specified.

	If res is None (default), res (grid resolution) is given the value \
	res = alpha*swath

	If lon/lat are not specified, auto-generate lon/lat by either \n
	1. Use SwathEnvelope to build lon/lat only where swath exists
	- occurs if swath coverage is small relative to area of Earth \n
	2. Create a DGG globally
	- occurs if swath coverage is large relative to area of Earth

	"""

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

	# spherical lat for AnalyticCoverage
	phi = np.radians(lat)
	phi_c = np.arctan((R_earth_pole/R_earth)**2 * np.tan(phi))
	lat_c = np.degrees(phi_c)
	AC = AnalyticCoverage(orb, swath, lon, lat_c, JD1, JD2)
	t_access = AC.get_access(verbose=verbose, accuracy=2)


	return lon, lat, t_access



