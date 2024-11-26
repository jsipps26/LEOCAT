
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict


def merge_intervals(intervals):
	if not intervals:
		return []

	# Sort intervals by their start value
	# intervals.sort(key=lambda x: x[0])

	merged = [intervals[0]]  # Initialize merged list with the first interval

	for i in range(1, len(intervals)):
		# Compare the current interval with the last interval in the merged list
		if intervals[i][0] <= merged[-1][1]:  # Overlapping intervals
			# Merge by updating the end of the last interval in the merged list
			merged[-1][1] = max(merged[-1][1], intervals[i][1])
		else:
			# No overlap, add the interval to the merged list
			merged.append(intervals[i])

	return merged
	
	
def intervals_intersection(A, B, return_A_idx=False):
	"""
	A and B must be sorted by column

	"""
	intersections = []
	A_idx = []
	i, j = 0, 0
	
	while i < len(A) and j < len(B):
		# Current intervals in A and B
		a_start, a_end = A[i]
		b_start, b_end = B[j]
		
		# Find intersection, if any
		intersection_start = max(a_start, b_start)
		intersection_end = min(a_end, b_end)
		
		# If there's an overlap, add it to the result list
		if intersection_start <= intersection_end:
			intersections.append([intersection_start, intersection_end])
			if return_A_idx:
				A_idx.append(i)
		
		# Move to the next interval in A or B depending on which ends earlier
		if a_end < b_end:
			i += 1
		else:
			j += 1
		
	intersections = np.array(intersections)

	if return_A_idx:
		return intersections, np.array(A_idx)

	return intersections

	
def contains_points(poly_vert, points):
	from matplotlib import path as path_mpl
	xv_vec, yv_vec = points.T
	p_mpl = path_mpl.Path(poly_vert)
	b = p_mpl.contains_points(np.hstack((xv_vec[:,np.newaxis],yv_vec[:,np.newaxis])))
	return b

def sort_col(x, col=0):
	# https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
	if type(x) != np.ndarray:
		x = np.array(x)
	x_sort = x[x[:,col].argsort()]
	return x_sort

def sort_row(x, row=0):
	# https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
	if type(x) != np.ndarray:
		x = np.array(x)
	x_sort = sort_col(x.T, row).T
	return x_sort

def inner_region(reg1, reg2, equal=True):
	if overlap(reg1, reg2, equal):
		total_reg = sorted([reg1[0], reg1[1], reg2[0], reg2[1]])
		return [total_reg[1], total_reg[2]]
	else:
		return []

def get_bbox_sector_ring(r1, r2, theta1, theta2, radians=True):
	if not radians:
		theta1 = np.radians(theta1)
		theta2 = np.radians(theta2)
		
	theta1_norm = theta1 % (2*np.pi)
	theta2_norm = theta2 % (2*np.pi)
	if np.abs(theta1_norm-theta2_norm) < 1e-8:
		x_min, x_max = -r2, r2
		y_min, y_max = -r2, r2
		return x_min, x_max, y_min, y_max 

	def angle_in_sector(angle):
		angle = angle % (2*np.pi)
		theta1_norm = theta1 % (2*np.pi)
		theta2_norm = theta2 % (2*np.pi)
		if theta1_norm > theta2_norm:
			return angle >= theta1_norm or angle <= theta2_norm
		else:
			return theta1_norm <= angle <= theta2_norm

	points = [[r1*np.cos(theta1), r1*np.sin(theta1)],
			[r1*np.cos(theta2), r1*np.sin(theta2)],
			[r2*np.cos(theta1), r2*np.sin(theta1)],
			[r2*np.cos(theta2), r2*np.sin(theta2)]]
	#
	for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
		if angle_in_sector(angle):
			points.append((r2*np.cos(angle), r2*np.sin(angle)))

	points = np.array(points)

	x_min, x_max = np.min(points.T[0]), np.max(points.T[0])
	y_min, y_max = np.min(points.T[1]), np.max(points.T[1])

	return x_min, x_max, y_min, y_max


def eq_check(reg):
	if reg[1] - reg[0] != 0:
		return reg
	else:
		return []
		

def overlap(reg1, reg2, equal=True):

	# equal=True means that if
	#	reg1 = [10,20]
	#	reg2 = [20,40],
	# output will be 1 since 20's are the same

	reg = sort_col([reg1, reg2])
	reg1_t, reg2_t = reg[0], reg[1]

	b = 0
	if equal:
		if reg1_t[1] < reg2_t[0]:
			return 0
		else:
			return 1
	else:
		if reg1_t[1] <= reg2_t[0]:
			return 0
		else:
			return 1


def hash_xy_DGG(cols, rows, DGG):
	lon_min_lat = DGG.lon_min_lat # vector
	y_origin = -90.0 # Sim.space_params['y_origin'] # scalar
	dlon = DGG.dlon # vector
	dlat = DGG.dlat # scalar
	r_min_offset = DGG.r_min_offset # scalar
	x_GP = hash_value(cols, lon_min_lat[rows-r_min_offset], dlon[rows-r_min_offset])
	y_GP = hash_value(rows, y_origin, dlat)
	return x_GP, y_GP


def hash_cr_DGG(x_GP, y_GP, DGG):
	# Only works for global ROI

	lon_min_lat = DGG.lon_min_lat # vector
	y_origin = -90 # Sim.space_params['y_origin'] # scalar
	dlon = DGG.dlon # vector
	dlat = DGG.dlat # scalar
	r_min_offset = DGG.r_min_offset
	rows = hash_index(y_GP, y_origin, dlat)
	# cols = hash_index(x_GP, lon_min_lat[rows-r_min], dlon[rows-r_min])
	cols = hash_index(x_GP, lon_min_lat[rows-r_min_offset], dlon[rows-r_min_offset])
	return cols, rows


def hash_cr(vec_xy, origin_xy, res_xy, Sim=None):
	b_DGG = False
	if Sim is not None:
		if 'DGG' in Sim.space_params:
			b_DGG = True

	if not b_DGG:
		cols = hash_index(vec_xy[0], origin_xy[0], res_xy[0])
		rows = hash_index(vec_xy[1], origin_xy[1], res_xy[1])
		return cols, rows

	else:
		DGG = Sim.space_params['DGG']
		x_GP, y_GP = vec_xy
		lon_min_lat = DGG.lon_min_lat # vector
		y_origin = Sim.space_params['y_origin'] # scalar
		dlon = DGG.dlon # vector
		dlat = DGG.dlat # scalar
		r_min_offset = DGG.r_min_offset
		rows = hash_index(y_GP, y_origin, dlat)
		# cols = hash_index(x_GP, lon_min_lat[rows-r_min], dlon[rows-r_min])
		cols = hash_index(x_GP, lon_min_lat[rows-r_min_offset], dlon[rows-r_min_offset])
		return cols, rows

		# DGG_params = Sim.space_params['DGG']
		# x_GP, y_GP = vec_xy
		# lon_min_lat = DGG_params['lon_min_lat'] # vector
		# y_origin = Sim.space_params['y_origin'] # scalar
		# dlon = DGG_params['dlon'] # vector
		# dlat = DGG_params['dlat'] # scalar
		# # r_min = DGG_params['r_min'] # scalar
		# r_min_offset = DGG_params['r_min_offset']
		# rows = hash_index(y_GP, y_origin, dlat)
		# # cols = hash_index(x_GP, lon_min_lat[rows-r_min], dlon[rows-r_min])
		# cols = hash_index(x_GP, lon_min_lat[rows-r_min_offset], dlon[rows-r_min_offset])
		# return cols, rows


def hash_xy(vec_cr, origin_xy, res_xy, Sim=None):
	b_DGG = False
	if Sim is not None:
		if 'DGG' in Sim.space_params:
			b_DGG = True

	if not b_DGG:
		x = hash_value(vec_cr[0], origin_xy[0], res_xy[0])
		y = hash_value(vec_cr[1], origin_xy[1], res_xy[1])
		return x, y

	else:
		DGG = Sim.space_params['DGG']
		cols, rows = vec_cr
		lon_min_lat = DGG.lon_min_lat # vector
		y_origin = Sim.space_params['y_origin'] # scalar
		dlon = DGG.dlon # vector
		dlat = DGG.dlat # scalar
		r_min_offset = DGG.r_min_offset # scalar
		x_GP = hash_value(cols, lon_min_lat[rows-r_min_offset], dlon[rows-r_min_offset])
		y_GP = hash_value(rows, y_origin, dlat)
		return x_GP, y_GP

		# DGG_params = Sim.space_params['DGG']
		# # x_GP, y_GP = vec_xy
		# cols, rows = vec_cr
		# lon_min_lat = DGG_params['lon_min_lat'] # vector
		# y_origin = Sim.space_params['y_origin'] # scalar
		# dlon = DGG_params['dlon'] # vector
		# dlat = DGG_params['dlat'] # scalar
		# # r_min = DGG_params['r_min'] # scalar
		# r_min_offset = DGG_params['r_min_offset'] # scalar
		# # x_GP = hash_value(cols, lon_min_lat[rows-r_min], dlon[rows-r_min])
		# x_GP = hash_value(cols, lon_min_lat[rows-r_min_offset], dlon[rows-r_min_offset])
		# y_GP = hash_value(rows, y_origin, dlat)
		# return x_GP, y_GP


def hash_index(val, val0, res):
	if not isinstance(val, np.ndarray):
		val = np.array(val)
	return np.floor((val-val0)/res).astype(int)

def hash_value(cols, x_origin, res):
	# hash_index inverse
	if not isinstance(cols, np.ndarray):
		cols = np.array(cols).astype(int)
	return x_origin + cols*res + res/2

@njit
def is_unique(arr):
	# use dictionary as search, do not sort
	d = Dict.empty(key_type=types.int64, value_type=types.int64)
	for i,val in enumerate(arr):
		if val in d:
			# print(i,len(arr))
			return False
		else:
			d[val] = 0
	return True

@njit
def unique_index(arr):
	# use dictionary as search, do not sort
	#	build L once
	d = Dict.empty(key_type=types.int64, value_type=types.int64)
	L = []
	for i,val in enumerate(arr):
		if not (val in d):
			L.append(i)
			d[val] = 0
	return np.array(L)

def isin_2d(cr1, cr2):
	# isin in 2d with cantor pairing
	cr1_adj = np.min(cr1,axis=0)-1
	cr2_adj = np.min(cr2,axis=0)-1
	cr_adj = np.min(np.vstack((cr1_adj,cr2_adj)),axis=0)

	cr1_nat = cr1 - cr_adj
	cr2_nat = cr2 - cr_adj

	cp1 = cantor_pairing(cr1_nat.T[0],cr1_nat.T[1])
	cp2 = cantor_pairing(cr2_nat.T[0],cr2_nat.T[1])
	b_dup = np.isin(cp1,cp2)

	return b_dup

def unique_2d(c_int, r_int):
	cr_u0 = np.unique(np.array([c_int, r_int]), axis=1).T
	return cr_u0

def cantor_pairing(cols,rows):
	# only works for positive integers
	# return (c + r + 1)*(c + r)//2 + r
	return (cols + rows + 1)*(cols + rows)//2 + rows

	"""
	Max integer with numpy

	In best-case, past int64 max numpy changes int to float.
	It could also change int64 to uint64, but that might
	be conditional. I think expect bad things to happen if
	an int exceeds 2**63-1.

	Python "int" object is unbounded, but that's because it's
	not an int, it's an object. But numpy arrays require a
	basic datatype, so the constraint still exists unless you
	convert cantor_pairing to loop over python int objects.

	Cantor pairing is limited by numerator exceeding uint64. Cols and
	rows cannot both be greater than 2**31-1, but each may be if
	the other is less.

	Numpy does not automatically convert to uint64, so logic below
	forces it, and errors if the numerator is actually above uint64.
	Pratically, this means coverage is limited to >2cm resolution at
	global scale.

	Also: unique_index is limited to int64 cantor pairing values

	"""

	# import warnings

	# # INT_MAX = 2**63 - 1 # int64
	# # UINT_MAX = 2*(2**63) - 1 # uint64

	# c_max_np, r_max_np = np.max(cols), np.max(rows)
	# c_max, r_max = int(c_max_np), int(r_max_np)
	# n = (c_max+r_max+1)*(c_max+r_max) # python int unaffected by uint64 max
	# with warnings.catch_warnings():
	# 	warnings.simplefilter('ignore', RuntimeWarning)
	# 	n_np = int( (c_max_np+c_max_np+1)*(c_max_np+c_max_np) )
	# if n != n_np: # if they aren't the same, numpy datatype is too small
	# 	cols = cols.astype(np.uint64)
	# 	rows = rows.astype(np.uint64)
	# 	c_max_np, r_max_np = np.max(cols), np.max(rows)
	# 	c_max, r_max = int(c_max_np), int(r_max_np)
	# 	n = (c_max+r_max+1)*(c_max+r_max)
	# 	with warnings.catch_warnings():
	# 		warnings.simplefilter('ignore', RuntimeWarning)
	# 		n_np = (c_max_np+r_max_np+np.array(1,dtype=np.uint64))*(c_max_np+r_max_np)
	# 	if n != n_np:
	# 		raise ValueError('cantor pairing exceeds uint64, reduce size of cols or rows')

	# 	return (cols+rows+np.array(1,dtype=np.uint64))*(cols+rows) // np.array(2,dtype=np.uint64) + rows

	# else:
	# 	return (cols + rows + 1)*(cols + rows)//2 + rows


def cantor_pairing_inv(pairs):
	# https://en.wikipedia.org/wiki/Pairing_function
	z = pairs
	w = np.floor( (np.sqrt(8*z+1) - 1)/2 ).astype(int)
	t = (w**2 + w)//2
	r = z - t
	c = w - r
	return c, r

def boolean_cut_2d(lon, lat, bounds):
	[lon_min, lon_max, lat_min, lat_max] = bounds
	b_lon = (lon >= lon_min) & (lon <= lon_max)
	b_lat = (lat >= lat_min) & (lat <= lat_max)
	b = b_lon & b_lat

	return b

def intersect_2d(A, B):

	cA, rA = A.T
	cB, rB = B.T
	# c_min = np.min([np.min(cA),np.min(cB)])-1
	# r_min = np.min([np.min(rA),np.min(rB)])-1
	if len(cA) > 0 and len(cB) > 0:
		c_min = np.min([np.min(cA),np.min(cB)])-1
	elif len(cA) > 0:
		c_min = np.min(cA)-1
	elif len(cB) > 0:
		c_min = np.min(cB)-1
	else:
		c_min = -1

	if len(rA) > 0 and len(rB) > 0:
		r_min = np.min([np.min(rA),np.min(rB)])-1
	elif len(rA) > 0:
		r_min = np.min(rA)-1
	elif len(rB) > 0:
		r_min = np.min(rB)-1
	else:
		r_min = -1

	cA, rA = cA-c_min, rA-r_min
	cB, rB = cB-c_min, rB-r_min

	cp_A = cantor_pairing(cA,rA)
	cp_B = cantor_pairing(cB,rB)

	cp_common, a_inds, b_inds = np.intersect1d(cp_A, cp_B, return_indices=True)
	a_inds = a_inds.astype(int)
	b_inds = b_inds.astype(int)
	# A[a_inds] == B[b_inds]

	return A[a_inds], a_inds, b_inds

	# nrows, ncols = A.shape
	# dtype={'names':['f{}'.format(i) for i in range(ncols)],
	# 		'formats':ncols * [A.dtype]}
	# C, a_inds, b_inds = np.intersect1d(A.view(dtype), B.view(dtype), return_indices=True)
	# try:
	# 	C, a_inds, b_inds = np.intersect1d(A.view(dtype), B.view(dtype), return_indices=True)
	# except ValueError:
	# 	A = sort_col(A)
	# 	B = sort_col(B)
	# 	C, a_inds, b_inds = np.intersect1d(A.view(dtype), B.view(dtype), return_indices=True)
	# return C.view(A.dtype).reshape(-1, ncols), a_inds, b_inds

def get_raster2(x, y, z, res, method=np.nanmean, fill_value=np.nan, x_origin=0, y_origin=0, column_major=True):

	"""
	New version of get_raster
	Cleaner code, but also adjusts for a given origin
	x/y_origin assumed to be zero in old get_raster
	Now, one can specify the origin of an external, global
	grid, then create a raster that is on that particular
	grid, rather than just a raster that has a certain resolution.

	returns a raster object
	column-major xx, yy, zz

	"""
	import pandas as pd
	class raster_struct:
		def __init__(self, xx, yy, zz):
			self.xx = xx
			self.yy = yy
			self.zz = zz

	# def hash_cr(val, val0, res):
	# 	if not isinstance(val, np.ndarray):
	# 		val = np.array(val)
	# 	return np.floor((val-val0)/res).astype(int)

	# x_origin = -53141.321
	# y_origin = 0.521

	# res = 10 #7.33
	# fill_value = np.nan
	dx0 = x_origin - hash_index(x_origin, 0, res)*res
	dy0 = y_origin - hash_index(y_origin, 0, res)*res

	x_min, y_min = np.min(x), np.min(y)
	x_max, y_max = np.max(x), np.max(y)
	x0 = hash_index(x_min, 0, res)*res + dx0
	y0 = hash_index(y_min, 0, res)*res + dy0
	if x0 > x_min:
		x0 -= res
	if y0 > y_min:
		y0 -= res

	c = hash_index(x, x0, res)
	if (c < 0).any():
		# should not occur b/c x0 < x_min
		print('error: c')
	# r = hash_index(y, y0, res)
	c_min, c_max = hash_index([x0+res/2, x_max], x0, res)
	r_min, r_max = hash_index([y0+res/2, y_max], y0, res)
	num_cols = c_max - c_min + 1
	num_rows = r_max - r_min + 1
	zz = np.full((num_cols,num_rows), fill_value)

	c_range = np.arange(num_cols) + c_min
	r_range = np.arange(num_rows) + r_min
	xg = x0 + c_range*res + res/2
	yg = y0 + r_range*res + res/2
	xx, yy = np.meshgrid(xg, yg)
	xx, yy = xx.T, yy.T

	# method = np.size

	# c_rel, r_rel = c-c_min, r-r_min
	# df = pd.DataFrame({'c': c_rel, 'r': r_rel, 'z': z})
	df = pd.DataFrame({'c': hash_index(x, x0, res), 'r': hash_index(y, y0, res), 'z': z})
	df_agg = df.groupby(['c','r']).agg({'z': [method]})
	df_agg.columns = ['z_agg']
	df_agg = df_agg.reset_index()

	zz[df_agg['c'],df_agg['r']] = df_agg['z_agg']

	# print((x0-x_origin)/res) # should be integer
	# print((y0-y_origin)/res)

	if column_major:
		return raster_struct(xx, yy, zz)
		# return xx, yy, zz
	else:
		return raster_struct(xx.T, yy.T, zz.T)
		# return xx.T, yy.T, zz.T


def fill_gaps(raster, num_cells=5, debug=0):

	from scipy.interpolate import NearestNDInterpolator
	import copy
	
	xx, yy, zz = raster.xx, raster.yy, raster.zz

	# num_cells = 5 # must be odd
	dn = int( (num_cells-1)/2 )
	zz_mid = zz[dn:-dn,dn:-dn]
	bb = ~np.isnan(zz_mid)
	# bb = ~np.isnan(zz)
	rows, cols = np.where(~bb)
	rows, cols = rows.astype(int), cols.astype(int)

	for dr in range(-dn,dn+1):
		for dc in range(-dn,dn+1):
			if dr == 0 and dc == 0:
				continue
			bb[rows,cols] = bb[rows,cols] | ~np.isnan(zz[rows+dr+dn,cols+dc+dn])

	#
	"""
	I think the sets are
		valid numbers
		nans that are disallowed
		nans that are allowed
	I think 
		bb = (valid numbers & nans that are disallowed)
		np.isnan(zz) = (nans that are allowed & nans that are disallowed)

		bb & np.isnan(zz) = nans that are disallowed?

	interpolator itself should be generated by valid numbers only (~np.isnan(zz))
	interpolator should only operate on nans that are disallowed (bb & np.isnan(zz))

	"""

	nans_to_remove = bb & np.isnan(zz_mid)

	# mask1 = valid numbers (generate interpolant)
	# mask2 = nans that we need to interpolate over
	mask1 = ~np.isnan(zz) # valid numbers
	mask2 = np.zeros(zz.shape).astype(bool)
	mask2[dn:-dn,dn:-dn] = nans_to_remove

	if debug:
		fig, ax = make_fig()
		ax.pcolormesh(xx, yy, zz, shading='auto')
		ax.plot(xx[mask2], yy[mask2], 'r.', label='nans to interpolate thru')
		ax.legend()
		fig.show()


	xym = np.vstack( (xx[mask1], yy[mask1]) ).T
	zm = zz[mask1]
	f = NearestNDInterpolator(xym, zm)
	zz_intp = copy.deepcopy(zz)
	zz_intp[mask2] = f(xx[mask2], yy[mask2])

	if debug > 1:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.pcolormesh(xx.T, yy.T, zz.T, shading='auto')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title('zz')
		fig.show()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.pcolormesh(xx.T, yy.T, zz_intp.T, shading='auto')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title('zz_intp')
		fig.show()

	# zz = zz_intp
	# return zz_intp
	raster.zz = zz_intp
	return raster


def mask_region(A, B, priority, equal=True):
	
	if A == B:
		if priority == 1:
			return A, []
		elif priority == 2:
			return [], A

	if overlap(A, B, equal):
		# overlap exists
		b = 0
		if equal:
			if B[0] <= A[0] and B[1] >= A[1]:
				b = 1
			elif A[0] <= B[0] and A[1] >= B[1]:
				b = 2
			else:
				b = 3
		else:
			if B[0] < A[0] and B[1] > A[1]:
				b = 1
			elif A[0] < B[0] and A[1] > B[1]:
				b = 2
			else:
				b = 3

		# if B[0] <= A[0] and B[1] >= A[1]:
		if b == 1:
			# full overlap, superset B
			if priority == 1:
				A_new_eq = eq_check([A[0],A[1]])
				B_new = [[B[0],A[0]], [A[1],B[1]]]

				B_new_eq = B_new
				if eq_check(B_new[0]) == []:
					B_new_eq = B_new[1]
					if eq_check(B_new[1]) == []:
						B_new_eq = []

				if eq_check(B_new[1]) == []:
					B_new_eq = B_new[0]
					if eq_check(B_new[0]) == []:
						B_new_eq = []

				return A_new_eq, B_new_eq

			elif priority == 2:
				A_new_eq = []
				B_new_eq = eq_check([B[0],B[1]])
				return A_new_eq, B_new_eq


		# elif A[0] <= B[0] and A[1] >= B[1]:
		elif b == 2:
			# full overlap, superset A
			if priority == 1:
				A_new_eq = eq_check([A[0],A[1]])
				B_new_eq = []
				return A_new_eq, B_new_eq

			elif priority == 2:
				A_new = [[A[0],B[0]], [B[1],A[1]]]
				B_new_eq = eq_check([B[0],B[1]])

				A_new_eq = A_new
				if eq_check(A_new[0]) == []:
					A_new_eq = A_new[1]
					if eq_check(A_new[1]) == []:
						A_new_eq = []

				if eq_check(A_new[1]) == []:
					A_new_eq = A_new[0]
					if eq_check(A_new[0]) == []:
						A_new_eq = []

				return A_new_eq, B_new_eq

		else:
			# partial overlap
			if A[0] < B[0]:
				if priority == 1:
					A_new = [A[0],A[1]]
					B_new = [A[1],B[1]]
					return eq_check(A_new), eq_check(B_new)
				elif priority == 2:
					A_new = [A[0],B[0]]
					B_new = [B[0],B[1]]
					return eq_check(A_new), eq_check(B_new)

			elif B[0] < A[0]:
				if priority == 1:
					A_new = [A[0],A[1]]
					B_new = [B[0],A[0]]
					return eq_check(A_new), eq_check(B_new)
				elif priority == 2:
					A_new = [B[1],A[1]]
					B_new = [B[0],B[1]]
					return eq_check(A_new), eq_check(B_new)

	else:
		# no overlap
		return [], []



def mask_region_multi(region_ids, regions, P, debug=False):


	"""
	This function takes in multiple regions defined by start/end
	times and multiple priorities for each region (can have same
	priority for multiple regions), and outputs the highest
	priority overlap for all regions, i.e.
		if given regions A, B
		A: [10, 20], priority == 1 (P[0] = 1)
		B: [15, 17], priority == 2 (P[1] = 2),
		
		output
		A: [[10, 15], [17, 20]]
		B: [[15, 17]]
		since B has a higher priority than A

	Input:
	1. region_ids:
		- some kind of unique identifier for each region
		- can be number, letter, word, etc, outputs as dictionary keys
	2. regions:
		- multiple regions defined like
			regions = [[10, 20], [15, 17], [30, 40], ...]
			note that second number > first number
	3. P:
		- integer priorities for each region (can be negative)
			P = [1, 1, 3, 2, 3, 1, 5, ...]


	Output:
	1. namelist:
		- dictionary of region_ids with entries as the final regions
		- note that a final region can consist of multiple regions for
			the same region id
		- note further than some regions can be removed if they are
			entirely overlapped by a higher-priority region

	2. regions_all: (mostly for debug)
		- all regions in namelist, sorted such that any overlaps
			would be very visible (there should be no overlaps
			at the end)

	"""


	num_regions = len(region_ids)
	err = 0
	if num_regions != len(regions):
		print('error: len(region_ids) != len(regions)')
		err = 1
	if num_regions != len(P):
		print('error: len(region_ids) != len(P)')
		err = 1

	# check that regions is correctly formatted,
	# and sort each region by least-to-greatest
	# just in case
	for i in range(num_regions):
		if len(regions[i]) != 2:
			err = 1
			print('error: len(regions[%d]) != 2' % i)
			# break
		regions[i] = sorted(regions[i])

		if type(P[i]) != int:
			# break
			p = int(P[i])
			if p != P[i]:
				err = 1
				print('error: type(P[%d]) != int' % i)
			P[i] = p

	if err:
		return {}, []

	############################################
	# # Testing
	# seed = int(sys.argv[1])
	# np.random.seed(seed)
	# num_regions = 75
	# # max_time = 150
	# d_time = 25
	# max_priority = 15

	# P = []
	# regions = []
	# for i in range(num_regions):
	# 	while 1:
	# 		# a = np.random.randint(max_time)
	# 		# b = np.random.randint(max_time)
	# 		# t0, tf = i*5, i*5 + d_time
	# 		t0, tf = i, i + d_time
	# 		# a = np.random.randint(t0,tf)
	# 		# b = np.random.randint(t0,tf)
	# 		a = np.random.rand(1)[0]*(tf-t0) + t0
	# 		b = np.random.rand(1)[0]*(tf-t0) + t0
	# 		if abs(a) > abs(b+1):
	# 			break

	# 	regions.append(sorted([a,b]))
	# 	P.append(np.random.randint(max_priority))

	# a0 = 65
	# # region_ids = [str(unichr(a0+i)) for i in range(num_regions)]
	# region_ids = [str(i) for i in range(num_regions)]

	# namelist, regions_all = rd.mask_region_multi(region_ids, regions, P, debug=False)
	############################################



	# sort by starting time
	regions_p = [[regions[i],P[i],region_ids[i]] for i in range(num_regions)]
	regions_p.sort(key=lambda x: x[0][0])
	regions = [regions_p[i][0] for i in range(num_regions)]
	regions_check = combine_region(regions) # for debug
	P = [regions_p[i][1] for i in range(num_regions)]
	region_ids = [regions_p[i][2] for i in range(num_regions)]


	# increase duplicate entries in P so there is no
	# equal priority
	#	sort P, then increase remaining list if duplicates are found
	#	i.e. given P       = [2, 4, 1, 2, 1, 2, 5, 5, 2]  (after regions time-sorting)
	#			   P_s 	   = [1, 1, 2, 2, 2, 2, 4, 5, 5]  (sort by priority)
	#			   P_s_new = [1, 2, 3, 4, 5, 6, 8, 9, 10] (apply algorithm)
	#			   P_new   = [3, 8, 1, 4, 2, 5, 9, 10, 6] (reverse sort back to original P)

	P0 = list(np.copy(P))
	# print('P', P)
	if len(np.unique(P)) != len(P):
		# some duplicate entries exist

		P_old = list(np.copy(P))
		P_s_index = np.argsort(P_old)	# save indicies for reversing sort later
		P_s0 = list(np.array(P_old)[P_s_index])
		P_s = list(np.copy(P_s0))
		# print('P_s', P_s)

		P_s_new = []
		dp = 0
		for i in range(num_regions):
			if not (P_s[i] in P_s_new):
				P_s_new.append(P_s[i])
			else:
				P_s = P_s[:i] + list(np.array(P_s[i:]) + 1)
				P_s_new.append(P_s[i])

		# print('P_s_new', P_s_new)

		P_new = np.zeros(num_regions).astype(int)
		for i in range(num_regions):
			P_new[P_s_index[i]] = P_s_new[i]
		P_new = list(P_new)

		# print('P_new', P_new)
		P = P_new

	# sys.exit()

	namelist = {}
	# for i, f in enumerate(files):
	# 	namelist[f] = [regions[i]]

	if debug:
		print(regions)
		print(P0)
		print(P)
		print('')


	# loop over all regions and determine how region i is limited
	# by all regions j, where j != i
	# 	this makes reg_limit_p, regions that region i cannot be in

	for i, reg1 in enumerate(regions):

		reg_limit_p = []
		for j, reg2 in enumerate(regions):
			if i == j:
				continue

			if overlap(reg1, reg2):
				if P[j] > P[i]:
					# reg1_new, reg2_new = overlap_priority(reg1, reg2, 2)
					reg1_new, reg2_new = mask_region(reg1, reg2, 2)
					if len(reg2_new) > 0:
						if type(reg2_new[0]) == list:
							# reg2_new == [[num, num], [num, num]]
							reg_limit_p.append([reg2_new[0], P[j]])
							reg_limit_p.append([reg2_new[1], P[j]])
						else:
							# reg2_new == [num, num]
							reg_limit_p.append([reg2_new, P[j]])



		# limit by highest priority first
		reg_limit_p.sort(key=lambda x: x[1], reverse=True)
		reg_limit = [arr[0] for arr in reg_limit_p] # regions that region i cannot be in
		if debug:
			print(i, P[i], reg_limit_p)

		reg1_total = [reg1]
		if len(reg_limit_p) > 0:
			# if limited at all
			k = 0
			while k < len(reg1_total):
				reg1_t = reg1_total[k]
				# if reg1_t == []:
				# 	del reg1_total[k]
				# 	# k -= 1
				# 	# k += 1
				# 	continue

				for reg_lim in reg_limit:
					if overlap(reg1_t, reg_lim):
						# reg1_t, _ = overlap_priority(reg1_t, reg_lim, 2)
						reg1_t, _ = mask_region(reg1_t, reg_lim, 2)
						# print(reg1_t)
						if len(reg1_t) > 0:
							if type(reg1_t[0]) == list:
								# reg1_t == [[num, num], [num, num]] or [[], [num, num]], etc

								if type(reg1_t[1]) != list:
									print('error mask_region_multi: list')

								r = reg1_t
								if r[0] != []:
									reg1_total[k] = r[0]
									reg1_t = r[0]
								else:
									if debug:
										print('delete (1)')
									del reg1_total[k]
									# if len(reg1_total) == 0:
									# 	break
									k -= 1
									break

								if r[1] != []:
									reg1_total.append(r[1])
								else:
									pass

								# reg1_total[k] = reg1_t[0]
								# reg1_total.append(reg1_t[1])
								# reg1_t = reg1_t[0]

							else:
								# reg1_t == [num, num]
								reg1_total[k] = reg1_t
						else:
							# reg1_t == []
							if debug:
								print('delete (2)')
							del reg1_total[k]
							# if len(reg1_total) == 0:
							# 	break
							k -= 1
							break
				k += 1

		else:
			# no limit
			if debug:
				print('no limit')


		if len(reg1_total) > 0:
			namelist[region_ids[i]] = reg1_total

		if debug:
			print(reg1, reg1_total)


	# debug checks
	regions_all = []
	for f in namelist:
		regions = namelist[f]
		for reg in regions:
			regions_all.append(reg)
	regions_all.sort()

	regions_all_cmb = combine_region(regions_all)

	if regions_all_cmb != regions_check:
		# combined regions start == combined regions end
		print('warning mask_region_multi: regions_all != regions_check')

	for i in range(1,len(regions_all)):
		reg1 = regions_all[i-1]
		reg2 = regions_all[i]
		if reg1[1] != reg2[0]:
			b = 1
			# for reg_cmb in regions_all_cmb:
			for j in range(1,len(regions_all_cmb)):
				# if reg1[1] == reg_cmb[1] and reg2[0] == reg_cmb[0]:
				reg_cmb1 = regions_all_cmb[j-1]
				reg_cmb2 = regions_all_cmb[j]
				if reg1[1] == reg_cmb1[1] and reg2[0] == reg_cmb2[0]:
					b = 0
			if b:
				"""
				Final regions should make up a continuous range of time.
				If not, then the separation must be due to separations
				shown in the entire combined set, i.e.
					region_ids = ['A','B','C']
					regions    = [[10,20],[15,17],[30,40]]
					P  		   = [1,2,3]
					output:
						namelist =
						{'A': [[10,15], [17,20]],
						 'B': [[15,17]],
						 'C': [[30,40]]}
						regions_all = 
							[[10,15], [15,17], [17,20], [30,40]]
						regions_all_cmb = 
							[[10,20],[30,40]]
					
				We should expect 10-15, 15-17, 17-20 to be continuous,
				then a separation only at the same place that there
				is separation in the combined set (regions_all_cmb),
				which is evident at 20-30.

				Floating/roundoff error may trigger this warning.
				"""
				print(reg1, reg2)
				print('warning mask_region_multi: reg')


	return namelist, regions_all

	

def find_region(t, dt, dt_buffer=0.0):

	degrade = []
	degrade_index_red = []
	if len(t) > 0:

		# This works by first giving every outlier point a region defined by
		# deg_definition_dt. Then, regions are added to degrade_index, and 
		# each overlapping super region is separated from one-another.
		# Then the first and last points of each super region are used to define
		# the overall degrade at that set of points.

		deg_regions = [[t[0] - dt, t[0] + dt]]
		degrade_index = [[0]]
		for k in range(1, len(t)):
			deg_t = t[k]
			deg_regions.append([deg_t - dt, deg_t + dt])

			# deg_start1 = deg_regions[k-1][0]
			deg_end1 = deg_regions[k-1][1]
			deg_start2 = deg_regions[k][0]
			# deg_end2 = deg_regions[k][1]

			if deg_end1 > deg_start2:
				# degrade regions overlap
				degrade_index[-1].append(k)
			else:
				degrade_index.append([])
				degrade_index[-1].append(k)


		# degrade_index[d] is a list of k-indices, the degrade regions
		# that all overlap, or a single region if len(degrade_index[d]) == 1

		for d in range(len(degrade_index)):
			# handles single point or multiple regions that overlap
			# in single point case, degrade_index[d][0] == degrade_index[d][-1]
			d_start = deg_regions[degrade_index[d][0]][0] 	# "deg_start1", or "deg_start1" in single-point case
			d_end = deg_regions[degrade_index[d][-1]][1]	# "deg_end2", or "deg_end1" in single-point case
			d_start += (dt - dt_buffer)
			d_end += (-dt + dt_buffer)

			degrade.append([d_start, d_end])
			degrade_index_red.append([degrade_index[d][0], degrade_index[d][-1]])
			# fp_outlier_deg_h.write('%d %15.6f %15.6f %15.6f\n' % (i+1, degrade_start, degrade_end, degrade_end - degrade_start))

	return degrade, degrade_index_red


def get_total_region(deg1, deg2):
	# This function compares the degrade ranges and
	# returns a single total region
	
	# assumes start1 < end1, start2 < end2
	if deg1[1] < deg2[0]:
		# non-overlapping regions
		overlap = 0
		return deg1, overlap
	else:
		overlap = 1
		deg = np.array([deg1, deg2])
		c_reg = deg.T
		index_start = np.argmin(c_reg[0])
		index_end = np.argmax(c_reg[1])

		dr = [c_reg[2][index_start], c_reg[3][index_end]]

		deg_new = [c_reg[0][index_start], c_reg[1][index_end],
				   c_reg[2][index_start], c_reg[3][index_end]]

		return deg_new, overlap


def combine_region(degrades):

	"""
	Only used for region times, no region id (such as degrade id)

	degrades is a list of [start, end] degrades
		[[start, end]
		 [start, end]
		 [...]]

	Addition 9/24/19
	- degrades can also be a list of [start, end, dr1, dr2] regions
	- this enables each smaller region to encompass a larger zone,
	  virtually, and the final combined region will only use the end-point
	  dr's
	Format:
		[[start+dr1, end+dr2, dr1, dr2],
		 [start+dr1, end+dr2, dr1, dr2],
		 [...]]
	Example:
		[[0.0, 100.0, -10.0, 10.0],
		 [-1.0, 50.0, -15.0, 25.0]]
		This would produce the combined region
		[[-1.0-15.0, 100.0+10.0]] == [[-16.0, 110.0]]

		Conceptually, if you had several regions that were separated by
		small gaps, you could combine them by artificially adding dr1
		and dr2.

	"""

	# This works by choosing a first degrade,
	# then comparing the ones that come after it to see
	# if they're within the same degrade region.
	# Overlapping regions are combined into one degrade,
	# then a new "first" degrade is chosen, and so on.
	
	n_deg = len(degrades)
	deg_reduced = [] # degrades with no overlapping regions
	if n_deg > 1:

		degrades_tr = np.transpose(degrades)
		if type(degrades_tr) == np.ndarray:
			if degrades_tr.dtype != 'O':
				length = len(degrades_tr)
				if length == 2:
					degrades = [[deg[0], deg[1], 0, 0] for deg in degrades]
				elif length == 4:
					pass
				else:
					print('error: length of regions must be either 2 or 4')
					return degrades
			else:
				print('error: inconsistent lengths in regions')
				return degrades

		degrades.sort() # sort by start time
		i = 0
		while i < n_deg:
			deg_new = degrades[i]
			j = i+1
			while j < n_deg:
				# recursively update the combined degrade
				deg_new, overlap = get_total_region(deg_new, degrades[j])
				if not overlap:
					break
				j += 1

			# append the final modification
			deg_reduced.append(deg_new)
			i = j

		if length == 2:
			deg_reduced = [[deg[0], deg[1]] for deg in deg_reduced]

	elif n_deg == 1:
		deg_reduced = [degrades[0]]

	return deg_reduced

