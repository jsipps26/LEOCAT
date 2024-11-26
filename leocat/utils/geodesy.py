

import numpy as np
from pyproj import Proj, Transformer
from leocat.utils.math import dot, unit, mag
from leocat.utils.index import hash_index

ecf = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
tr_lla_to_ecf = Transformer.from_proj(lla, ecf)

class DiscreteGlobalGrid:
	def __init__(self, N=None, A=None, m_list=[3], lon_band_option=1, ROI=[-180,180,-90,90],
						a=6378.137, f=1/298.257223563):
		self.a = a
		self.f = f
		self.ROI = ROI
		b = a*(1-f)
		e = np.sqrt(1 - (b/a)**2)
		# https://www.math.auckland.ac.nz/Research/Reports/Series/539.pdf (Cotes 1714, pg. 169-171)
		# https://www.jpz.se/Html_filer/wgs_84.html
		# https://analyticphysics.com/Mathematical%20Methods/Surface%20Area%20of%20an%20Ellipsoid.htm
		A_earth = 2*np.pi*a**2 + np.pi*b**2/e * np.log((1+e)/(1-e))
		# A_earth2 = ellipsoid_area(-180,180,-90,90) # same as above
		self.b = b
		self.A_earth = A_earth

		if N is None and A is None:
			raise Exception('Either grid cell area A or approx. number of points N must be specified')

		elif N is None:
			# area
			# self.A = A
			pass

		elif A is None:
			# number of points
			A = A_earth/N

		self.A = A
		self.res = np.sqrt(A)
		self.generate(A, ROI, m=m_list[0], lon_band_option=lon_band_option)


	def generate(self, A, ROI, m=3, lon_band_option=1, debug=0):

		ROI_global = True
		if ROI is not None:
			lon_min, lon_max, lat_min, lat_max = ROI
			if not (lon_max-lon_min == 360.0 and lat_max-lat_min == 180.0):
				ROI_global = False

		
		dlat0 = ellipsoid_band_lat2(-90, m*A) + 90
		n_lat = int(180/dlat0)
		dlat = 180/n_lat # geodetic change
		# print('n_lat', n_lat)

		if ROI_global:
			n_lat_half = n_lat//2 + 1
			lat_edge = -90 + np.arange(n_lat_half)*dlat # only works b/c n_lat_half
			lat_lower = lat_edge[:-1]
			lat_upper = lat_edge[1:]
			cols = ellipsoid_area(-180,180,lat_lower,lat_upper) / A
			cols_int = np.round(cols).astype(int)

			if n_lat % 2 == 0:
				# even
				cols_total = np.hstack((cols, np.flip(cols)))
				num_cols_lat = np.hstack((cols_int, np.flip(cols_int)))

			else:
				# odd
				lat_lower_eq = lat_edge[-1]
				lat_upper_eq = -lat_edge[-1]
				cols_eq = ellipsoid_area(-180,180,lat_lower_eq,lat_upper_eq) / A
				cols_eq_int = int(np.round(cols_eq))
				cols_total = np.hstack((cols, cols_eq, np.flip(cols)))
				num_cols_lat = np.hstack((cols_int, cols_eq_int, np.flip(cols_int)))

			n_lat_ROI = n_lat
			r_min, r_max = 0, n_lat-1
			c_min_lat = np.zeros(n_lat).astype(int)
			c_max_lat = num_cols_lat-1

			if lon_band_option == 1:
				lon_min_lat = np.full(n_lat_ROI,-180.0)
			elif lon_band_option == 2:
				lon_min_lat = np.random.uniform(-180,180,n_lat_ROI)

			r_min_offset = r_min
			r_max_offset = r_max

		else:
			# Subset ROI may not work in extreme cases, like meter-level ROIs
			#	or if resolution is too large for the ROI, or the swath is
			#	larger than the ROI

			# Find min/max bounds of workspace
			#	in rows and columns
			r_min = hash_index(lat_min+dlat/2, -90, dlat)
			r_max = hash_index(lat_max-dlat/2, -90, dlat)
			# print(r_min, r_max, dlat, n_lat)
			n_lat_ROI = r_max-r_min+1
			# lon_min_lat = np.full(n_lat_ROI,-180.0)
			if lon_band_option == 1:
				lon_min_lat = np.full(n_lat_ROI,-180.0)
			# elif lon_band_option == 2:
			# 	lon_min_lat = np.random.uniform(-180,180,n_lat_ROI)

			lat_lower = -90 + np.arange(r_min,r_max+1)*dlat
			lat_upper = lat_lower + dlat
			lat_lower = np.round(lat_lower,12)
			lat_upper = np.round(lat_upper,12)
			cols = ellipsoid_area(-180,180,lat_lower,lat_upper) / A
			num_cols_lat = np.round(cols).astype(int)
			dlon = 360.0/num_cols_lat
			c_min_lat = hash_index(lon_min+dlon/2, lon_min_lat, dlon)
			c_max_lat = hash_index(lon_max-dlon/2, lon_min_lat, dlon)

			# Find min/max bounds of accessible space
			#	made larger to compensate for mesh that may access
			#	col/rows out of bounds
			k_ROI = 1
			lat_diff = lat_max-lat_min
			lat_min_offset = lat_min-lat_diff*k_ROI # np.max([-90,lat_min-lat_diff*k_ROI])
			if lat_min_offset < -90:
				lat_min_offset = -90
			lat_max_offset = lat_max+lat_diff*k_ROI # np.min([lat_max+lat_diff*k_ROI,90])
			if lat_max_offset > 90:
				lat_max_offset = 90
			r_min_offset = hash_index(lat_min_offset+dlat/2, -90, dlat)
			r_max_offset = hash_index(lat_max_offset-dlat/2, -90, dlat)
			lat_lower = -90 + np.arange(r_min_offset,r_max_offset+1)*dlat
			lat_upper = lat_lower + dlat
			# r_min_offset = hash_index(lat_min_offset, -90, dlat)
			# r_max_offset = hash_index(lat_max_offset, -90, dlat)
			# lat_lower = -90 + np.arange(r_min_offset,r_max_offset+1)*dlat
			# lat_upper = lat_lower + dlat
			# b_lat = ((lat_lower >= -90) & (lat_lower <= 90)) & \
			# 		((lat_upper >= -90) & (lat_upper <= 90))
			# #
			# lat_lower, lat_upper = lat_lower[b_lat], lat_upper[b_lat]
			# r_min_offset = hash_index(lat_lower[0], -90, dlat)
			# r_max_offset = hash_index(lat_upper[-1], -90, dlat)
			# print(lat_lower)
			# print(lat_upper)
			# print(lat_min_offset) #, -90 + r_min_offset*dlat)
			# print(lat_max_offset) #, -90 + r_max_offset*dlat + dlat)
			lat_lower = np.round(lat_lower,12)
			lat_upper = np.round(lat_upper,12)
			cols = ellipsoid_area(-180,180,lat_lower,lat_upper) / A
			num_cols_lat = np.round(cols).astype(int)
			if lon_band_option == 1:
				lon_min_lat = np.full(len(num_cols_lat),-180.0)
			# elif lon_band_option == 2:
			# 	lon_min_lat = np.random.uniform(-180,180,len(num_cols_lat))

			# dlon = 360.0/num_cols_lat
			# c_min_lat = hash_index(lon_min+dlon/2, lon_min_lat, dlon)
			# c_max_lat = hash_index(lon_max-dlon/2, lon_min_lat, dlon)


		self.num_cols_lat = num_cols_lat
		self.lon_min_lat = lon_min_lat
		self.r_min = r_min
		self.r_max = r_max
		self.r_min_offset = r_min_offset
		self.r_max_offset = r_max_offset
		self.c_min_lat = c_min_lat
		self.c_max_lat = c_max_lat

		self.n_lat = n_lat
		self.dlon = 360.0/num_cols_lat
		self.dlat = dlat


	def get_params(self):
		params = {'num_cols_lat': self.num_cols_lat, 'lon_min_lat': self.lon_min_lat,
					'r_min': self.r_min, 'r_max': self.r_max,
					'c_min_lat': self.c_min_lat, 'c_max_lat': self.c_max_lat,
					'dlon': self.dlon, 'dlat': self.dlat, 'n_lat': self.n_lat,
					'r_min_offset': self.r_min_offset, 'r_max_offset': self.r_max_offset}
		#
		return params


	def get_lonlat(self):
		DGG_params = self.get_params()

		dlat = DGG_params['dlat']
		r_min_offset = DGG_params['r_min_offset']
		r_max_offset = DGG_params['r_max_offset']
		n_lat = DGG_params['n_lat']

		dlon_lat = DGG_params['dlon']
		c_min_lat = DGG_params['c_min_lat']
		c_max_lat = DGG_params['c_max_lat']
		lon_min_lat = DGG_params['lon_min_lat']
		lat0 = -90 + np.arange(r_min_offset,r_max_offset+1)*dlat + dlat/2
		lon, lat = [], []
		for j in range(n_lat):
			lon0 = lon_min_lat[j] + np.arange(c_min_lat[j],c_max_lat[j]+1)*dlon_lat[j] + dlon_lat[j]/2
			lon.append(lon0)
			lat.append(np.full(len(lon0),lat0[j]))
		lon = np.concatenate(lon)
		lat = np.concatenate(lat)

		return lon, lat






def set_projection_res(res_km, projection):
	if projection == 'LLA': # DO NOT MODIFY THIS
		res = km_to_deg(res_km)
	else:
		res = res_km
	return res

def km_to_deg(res_km, a=6378.137):
	# res_deg, _, _ = ev_direct(0, 0, 90, res_km, unit='km', radians=False)
	res_deg = np.degrees(res_km/a)
	return res_deg

def deg_to_km(res_deg, a=6378.137):
	# res_km = ev_inverse(0, 0, res_deg, 1e-3, dist_only=True, radians=False, unit='km')
	res_km = np.radians(res_deg)*a
	return res_km

def poly_grid_cell_map(x_GP, y_GP, res_x, res_y, N_side=1):

	if not isinstance(res_x,np.ndarray):
		res_x = np.full(x_GP.shape,res_x)
	if not isinstance(res_y,np.ndarray):
		res_y = np.full(y_GP.shape,res_y)

	poly_grid = []
	poly_ecef = []
	for k in range(len(x_GP)):
		poly_grid0 = poly_grid_cell(x_GP[k], y_GP[k], res_x[k], res_y[k], N_side=N_side)
		poly_grid.append(poly_grid0)
		# poly_ecef0 = tr_lla_ecef.transform(poly_grid0.T[0], poly_grid0.T[1], np.zeros(len(poly_grid0)))
		# poly_ecef0 = np.transpose([poly_ecef0[0],poly_ecef0[1],poly_ecef0[2]])
		# poly_ecef.append(poly_ecef0)

	return poly_grid


def poly_grid_cell(x0,y0,res_x,res_y,N_side=10):
	gc_corner1 = np.array([x0 - res_x/2, y0 - res_y/2])
	gc_corner2 = np.array([x0 + res_x/2, y0 - res_y/2])
	gc_corner3 = np.array([x0 + res_x/2, y0 + res_y/2])
	gc_corner4 = np.array([x0 - res_x/2, y0 + res_y/2])
	poly = []
	# N_side = 10
	for j in range(N_side): # 1 to 2
		gc = j/N_side * (gc_corner2 - gc_corner1) + gc_corner1
		poly.append(gc)
	for j in range(N_side): # 2 to 3
		gc = j/N_side * (gc_corner3 - gc_corner2) + gc_corner2
		poly.append(gc)
	for j in range(N_side): # 3 to 4
		gc = j/N_side * (gc_corner4 - gc_corner3) + gc_corner3
		poly.append(gc)
	for j in range(N_side): # 4 to 1
		gc = j/N_side * (gc_corner1 - gc_corner4) + gc_corner4
		poly.append(gc)
	poly.append(poly[0])
	poly = np.array(poly)

	# poly_ecef = tr_grid_ecef.transform(poly.T[0], poly.T[1], np.zeros(len(poly)))
	# poly_ecef = np.transpose([poly_ecef[0],poly_ecef[1],poly_ecef[2]])

	return poly


def convert_latitude(lat_input, lat_type_in, lat_type_out, a=6378.137, f=1/298.257223563):
	"""
	Input/output in degrees

	Converts geocentric to geodetic latitude
	Also has eqn relating geocentric, geodetic, and parametric latitude:

		(b/a)*tan(lat_geodetic) = tan(lat_parametric) = (a/b)*tan(lat_geocentric)

	"""

	b = a*(1-f)
	single_value = 0
	if type(lat_input) is not np.ndarray:
		lat_input = np.array([lat_input]).astype(float)
		if len(lat_input) == 1:
			single_value = 1

	lat_input = np.copy(lat_input)
	invalid = (lat_input > 90) | (lat_input < -90)
	lat_input[invalid] = np.nan
	if lat_type_in == lat_type_out:
		return lat_input

	if lat_type_in == 'geodetic' and lat_type_out == 'geocentric':
		lat = lat_input
		phi = np.radians(lat)
		phi_c = np.arctan((b/a)**2 * np.tan(phi))
		lat_c = np.degrees(phi_c)
		lat_output = lat_c

	elif lat_type_in == 'geocentric' and lat_type_out == 'geodetic':
		lat_c = lat_input
		phi_c = np.radians(lat_c)
		phi = np.arctan((a/b)**2 * np.tan(phi_c))
		lat = np.degrees(phi)
		lat_output = lat

	else:
		raise Exception('Invalid conversion, check lat_type_in and lat_type_out')


	if single_value:
		lat_output = lat_output[0]

	return lat_output


def ellipsoid_area(lon1, lon2, lat1, lat2, a=6378.137, f=1/298.257223563):

	gamma = 1.0000030087366867

	lat_c1 = convert_latitude(lat1, 'geodetic', 'geocentric')
	lat_c2 = convert_latitude(lat2, 'geodetic', 'geocentric')

	lam1 = np.radians(lon1)
	lam2 = np.radians(lon2)
	phi_c1 = np.radians(lat_c1)
	phi_c2 = np.radians(lat_c2)

	b = a*(1-f)
	e = np.sqrt(1 - (b/a)**2)
	k = np.sqrt((1-e**2)/e**2)

	coef = b**2 * (lam2-lam1) / (k*e**2)
	arg2 = np.arctan(np.sin(phi_c2)/k)
	arg1 = np.arctan(np.sin(phi_c1)/k)
	A = gamma * coef * (arg2-arg1)

	return A


def ellipsoid_band_lat2(lat1, A, dlon=360, a=6378.137, f=1/298.257223563):

	# Given lat1, what is lat2 such that area is A?

	gamma = 1.0000030087366867

	lat_c1 = convert_latitude(lat1, 'geodetic', 'geocentric')
	# lat_c2 = convert_latitude(lat2, 'geodetic', 'geocentric')

	dlam = np.radians(dlon)
	phi_c1 = np.radians(lat_c1)

	b = a*(1-f)
	e = np.sqrt(1 - (b/a)**2)
	k = np.sqrt((1-e**2)/e**2)

	arg1 = np.arctan(np.sin(phi_c1)/k)
	arg2 = A*k*e**2 / (gamma * b**2 * dlam)
	arg3 = k*np.tan(arg1 + arg2)
	phi_c2 = np.arcsin(arg3)

	lat_c2 = np.degrees(phi_c2)
	lat2 = convert_latitude(lat_c2, 'geocentric', 'geodetic')

	return lat2

def radius_at_latitude(lat, perpendicular=False, a=6378.137, f=1/298.257223563):
	# geodetic latitude
	# https://en.wikipedia.org/wiki/Earth_radius#Geocentric_radius

	b = a*(1-f)
	phi = np.radians(lat)
	arg1 = (a**2 * np.cos(phi))**2 + (b**2 * np.sin(phi))**2
	arg2 = (a*np.cos(phi))**2 + (b*np.sin(phi))**2
	radius = np.sqrt(arg1/arg2)

	if perpendicular:
		lat_c = convert_latitude(lat, 'geodetic', 'geocentric')
		phi_c = np.radians(lat_c)
		radius_perp = radius * np.cos(phi_c)
		return radius_perp

	return radius








def ev_inverse(lon1, lat1, lon2, lat2, max_iter=3, dist_only=False, radians=True, unit='m'):

	"""
	all lon/lat in radians
	distance in meters

	Ellipsoidal Vincenty eqns
	https://en.wikipedia.org/wiki/Vincenty%27s_formulae
	http://www.movable-type.co.uk/scripts/latlong-vincenty.html

	"""
	if not radians:
		lon1 = np.radians(lon1)
		lat1 = np.radians(lat1)
		lon2 = np.radians(lon2)
		lat2 = np.radians(lat2)

	a = 6378137.0
	f = 1/298.257223563
	b = (1-f)*a # 6356752.314245

	# lon1 = 0.0 * np.pi/180
	# lat1 = 45.0 * np.pi/180
	# lon2 = 45.0 * np.pi/180
	# lat2 = 60.0 * np/pi/180

	U1 = np.arctan((1-f)*np.tan(lat1))
	U2 = np.arctan((1-f)*np.tan(lat2))

	cos = np.cos
	sin = np.sin

	L = lon2 - lon1

	lam = L

	for i in range(max_iter):
	    sin_sig = np.sqrt( (cos(U2)*sin(lam))**2 + (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(lam))**2 )
	    cos_sig = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(lam)
	    sig = np.arctan2(sin_sig, cos_sig)

	    sin_alpha = (cos(U1)*cos(U2)*sin(lam)) / sin_sig
	    cos_alpha2 = 1 - sin_alpha**2
	    cos_2sigm = cos_sig - (2*sin(U1)*sin(U2)) / cos_alpha2
	    C = f/16 * cos_alpha2 * (4 + f*(4 - 3*cos_alpha2))

	    arg = sig + C*sin_alpha * (cos_2sigm + C*cos_sig * (-1 + 2*cos_2sigm**2))
	    lam_new = L + (1-C)*f*sin_alpha * arg
	    dlam = abs(lam_new - lam)
	    
	    lam = lam_new

	u2 = cos_alpha2 * (a**2 - b**2) / b**2
	A = 1 + u2/16384.0 * (4096 + u2*(-768 + u2*(320 - 175*u2)))
	B = u2/1024.0 * (256 + u2*(-128 + u2*(74 - 47*u2)))

	arg_sig = cos_sig * (-1 + 2*cos_2sigm**2) - B/6*cos_2sigm * (-3 + 4*sin_sig**2) * (-3 + 4*cos_2sigm**2)
	dsig = B * sin_sig * (cos_2sigm + 1/4*B*arg_sig)

	d = b*A*(sig - dsig)
	if unit == 'km':
		d = d/1e3

	if dist_only:
		return d

	alpha1 = np.arctan2( cos(U2)*sin(lam), cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(lam) )
	alpha2 = np.arctan2( cos(U1)*sin(lam), -sin(U1)*cos(U2) + cos(U1)*sin(U2)*cos(lam) )

	if not radians:
		alpha1 = np.degrees(alpha1)
		alpha2 = np.degrees(alpha2)

	return d, alpha1, alpha2


def ev_direct(L1, phi1, alpha1, s, tol=1e-12, max_iter=5, radians=True, unit='m'):

	"""
	Distance in meters

	Ellipsoidal Vincenty eqns
	https://en.wikipedia.org/wiki/Vincenty%27s_formulae
	http://www.movable-type.co.uk/scripts/latlong-vincenty.html

	ALL UNITS IN RADIANS and meters
	input: 	longitude (L1), latitude (phi1), azimuth (alpha1), distance (s) (meters)
	output: point at end of geodesic (L2, phi2, alpha2)

	"""

	if not radians:
		L1 = np.radians(L1)
		phi1 = np.radians(phi1)
		alpha1 = np.radians(alpha1)

	if unit == 'km':
		s = s*1e3

	a = 6378137.0
	f = 1/298.257223563
	b = (1-f)*a # 6356752.314245

	U1 = np.arctan((1-f)*np.tan(phi1))
	sig1 = np.arctan2(np.tan(U1), np.cos(alpha1))
	sin_alpha = np.cos(U1)*np.sin(alpha1)
	cos_alpha2 = (1 - sin_alpha**2)

	u2 = cos_alpha2 * (a**2 - b**2) / b**2
	A = 1 + u2/16384.0 * (4096 + u2*(-768 + u2*(320 - 175*u2)))
	B = u2/1024.0 * (256 + u2*(-128 + u2*(74 - 47*u2)))

	sig = s/(b*A)
	# dsig_prev = 1e12
	for i in range(max_iter):
		two_sigm = 2*sig1 + sig

		cos_sig = np.cos(sig)
		sin_sig = np.sin(sig)
		cos_2sigm = np.cos(two_sigm)

		arg_sig = cos_sig * (-1 + 2*cos_2sigm**2) - B/6*cos_2sigm * (-3 + 4*sin_sig**2) * (-3 + 4*cos_2sigm**2)
		dsig = B * sin_sig * (cos_2sigm + 1/4*B*arg_sig)

		sig = s/(b*A) + dsig
		# print(abs(dsig - dsig_prev))
		# if abs(dsig - dsig_prev) < tol:
		# 	break

		# dsig_prev = dsig

	
	# cos_sig = np.cos(sig)
	# sin_sig = np.sin(sig)
	# cos_alpha1 = np.cos(alpha1)
	# sin_alpha1 = np.sin(alpha1)

	arg = np.sqrt( sin_alpha**2 + (np.sin(U1)*np.sin(sig) - np.cos(U1)*np.cos(sig)*np.cos(alpha1))**2 )
	phi2 = np.arctan2( np.sin(U1)*np.cos(sig) + np.cos(U1)*np.sin(sig)*np.cos(alpha1), (1-f)*arg )
	lam = np.arctan2( np.sin(sig)*np.sin(alpha1), np.cos(U1)*np.cos(sig) - np.sin(U1)*np.sin(sig)*np.cos(alpha1) )
	C = f/16 * cos_alpha2 * (4 + f*(4 - 3*cos_alpha2))

	two_sigm = 2*sig1 + sig
	cos_2sigm = np.cos(two_sigm)
	arg_L = sig + C*np.sin(sig)*( cos_2sigm + C*np.cos(sig)*(-1 + 2*cos_2sigm**2) )
	L = lam - (1-C)*f*sin_alpha*( arg_L )
	L2 = L + L1
	alpha2 = np.arctan2( sin_alpha, -np.sin(U1)*np.sin(sig) + np.cos(U1)*np.cos(sig)*np.cos(alpha1) )

	if radians:
		return L2, phi2, alpha2
	else:
		return np.degrees(L2), np.degrees(phi2), np.degrees(alpha2)



def sin_edge(y_sin, side):
	if not isinstance(y_sin, np.ndarray):
		y_sin = np.array(y_sin)
	x_sin = np.zeros(y_sin.shape)
	_, lat = sin_to_lla(x_sin, y_sin)
	if side == 'left':
		lon = -180*np.ones(lat.shape)
	elif side == 'right':
		lon = 180*np.ones(lat.shape)
	x_edge, y_edge = lla_to_sin(lon, lat)
	return x_edge

def sin_bounds_func():
	fx_lim_left = lambda y: sin_edge(y, side='left')
	fx_lim_right = lambda y: sin_edge(y, side='right')
	return fx_lim_left, fx_lim_right

def lla_to_sin(lon, lat, lon_cent=0.0):
	# https://pubs.usgs.gov/pp/1395/report.pdf
	a = 6378137.0/1e3
	f = 1/298.257223563
	e = np.sqrt(2*f-f**2)
	b = a*np.sqrt(1-e**2)

	lam = np.radians(lon)
	phi = np.radians(lat)
	lam0 = np.radians(lon_cent)

	x_sin = a*(lam-lam0)*np.cos(phi) / np.sqrt( 1-e**2*np.sin(phi)**2 )
	arg1 = 1-e**2/4 - 3*e**4/64 - 5*e**6/256
	arg2 = 3*e**2/8 + 3*e**4/32 + 45*e**6/1024
	arg3 = 15*e**4/256 + 45*e**6/1024
	arg4 = 35*e**6/3072
	y_sin = a*(arg1*phi - arg2*np.sin(2*phi) + arg3*np.sin(4*phi) - arg4*np.sin(6*phi))

	return x_sin, y_sin


def sin_to_lla(x_sin, y_sin, lon_cent=0.0):
	# https://pubs.usgs.gov/pp/1395/report.pdf
	a = 6378137.0/1e3
	f = 1/298.257223563
	e = np.sqrt(2*f-f**2)
	b = a*np.sqrt(1-e**2)

	M = y_sin
	lam0 = np.radians(lon_cent)

	mu = M/( a*(1-e**2/4 - 3*e**4/64 - 5*e**6/256) )
	e1 = (1 - np.sqrt(1-e**2)) / (1 + np.sqrt(1-e**2))
	arg1 = 3*e1/2 - 27*e1**3/32
	arg2 = 21*e1**2/16 - 55*e1**4/32
	arg3 = 151*e1**3/96
	arg4 = 1097*e1**4/512
	phi = mu + arg1*np.sin(2*mu) + arg2*np.sin(4*mu) + arg3*np.sin(6*mu) + arg4*np.sin(8*mu)
	lam = lam0 + x_sin*np.sqrt(1 - e**2*np.sin(phi)**2) / (a*np.cos(phi))

	lon = np.degrees(lam)
	lat = np.degrees(phi)

	return lon, lat


def ecf_to_lla(x, y, z):
	# x, y, z in km
	# output alt in km
	lon, lat, alt = tr_lla_to_ecf.transform(x*1e3, y*1e3, z*1e3, direction='inverse')
	alt = alt/1e3 # km
	return lon, lat, alt


def lla_to_ecf(lon, lat, alt):
	# alt in km,
	# returns in km
	alt = alt * 1e3
	x, y, z = tr_lla_to_ecf.transform(lon, lat, alt)
	r_ecf = np.transpose([x, y, z]) / 1e3 # km
	return r_ecf

def spherical_dist(r0, r_est, R_earth=6378.137):
	r_est_unit = (r_est.T / np.linalg.norm(r_est, axis=1)).T
	r0_unit = (r0.T / np.linalg.norm(r0, axis=1)).T
	proj_arc = dot(r_est_unit, r0_unit)
	proj_arc[proj_arc < -1] = -1
	proj_arc[proj_arc > 1] = 1
	angle = np.arccos(proj_arc)
	d_est = R_earth*angle # arc length
	return d_est

def spherical_proj(r1, r2, fwd0):
	dist0 = spherical_dist(r1,r2)
	fwd_vec = np.cross(r2,r1) # nominally along fwd0
	sign = np.ones(len(r1))
	b_sign = dot(fwd0,fwd_vec) < 0
	sign[b_sign] = -sign[b_sign]
	s_proj0 = sign*dist0
	return s_proj0

def geodesic_curvature(r_ecf, v_ecf, a_ecf):
	# https://people.math.wisc.edu/~angenent/561/kgsolutions.pdf
	# 	track curvature across the ellipsoid (approx due to n_hat)
	n_hat = unit(r_ecf) # approx as spherical normal
	kappa_geo = dot(a_ecf, np.cross(n_hat,v_ecf)) / mag(v_ecf)**3
	return np.abs(kappa_geo)
	
def ellipsoidal_dist(r0, r_est, tr_lla_ecef):
	lon1, lat1, alt1 = tr_lla_ecef.transform(r0.T[0], r0.T[1], r0.T[2], direction='inverse')
	lon2, lat2, alt2 = tr_lla_ecef.transform(r_est.T[0], r_est.T[1], r_est.T[2], direction='inverse')
	d_est_ev = ev_inverse(np.radians(lon1), np.radians(lat1), 
							np.radians(lon2), np.radians(lat2), dist_only=True)
	#
	return d_est_ev/1e3

def RADEC_to_cart(RA, DEC):
	"""
	RA and DEC are in deg
	Either np.arrays or floats
	"""
	alpha = np.radians(RA)
	delta = np.radians(DEC)
	x = np.cos(delta) * np.cos(alpha)
	y = np.cos(delta) * np.sin(alpha)
	z = np.sin(delta)
	r = np.transpose([x,y,z])
	return r

def cart_to_RADEC(r):
	"""
	r is an nx3 matrix:
		x, y, z == r.T[0], r.T[1], r.T[2]
	or for one position, it could be
	an array or list of length 3:
		x0, y0, z0 == r[0], r[1], r[2]

	"""
	x, y, z = r.T[0], r.T[1], r.T[2]
	if len(r.shape) == 1:
		r_mag = np.linalg.norm(r)
	else:
		r_mag = np.linalg.norm(r,axis=1) #np.sqrt(x**2 + y**2 + z**2 )
	dec = np.arcsin(z/r_mag)*180/np.pi
	ra = (np.arctan2(y,x)*180/np.pi)
	if type(ra) == np.ndarray:
		ra[ra < 0] = ra[ra < 0] + 360
	else:
		if ra < 0:
			ra += 360
	return ra, dec