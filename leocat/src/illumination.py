
import numpy as np
from scipy.optimize import curve_fit
from leocat.fqs.fqs import single_cubic, single_quartic
from numba import njit

from leocat.utils.const import *
from leocat.utils.orbit import surface_illumination_approx, surface_illumination, \
								solar_pos_approx, lunar_pos_approx, lunar_phase, \
								get_lunar_p, get_LAN_dot
#
from leocat.utils.math import R1, R3, R3, unit, mag, matmul, dot

elev_threshold_solar0 = -18.0
elev_threshold_lunar0 = -0.8


def design_illumination(alt_vec, inc_vec, JD0_start, num_years, swath=0.0, 
						body='moon+sun', verbose=1):
#
	from tqdm import tqdm

	if len(alt_vec) == 0:
		return []

	iterator = range(len(alt_vec))
	if verbose == 1:
		iterator = tqdm(iterator)

	zz_out = []
	for I in iterator:
		if verbose > 1:
			print('%d/%d' % (I+1,len(alt_vec)))
		alt = alt_vec[I]
		inc_deg = inc_vec[I]
		verbose0 = verbose-1
		if verbose0 < 0:
			verbose0 = 0
		xx0, yy0, zz0 = orbit_illumination(alt, inc_deg, JD0_start, num_years, \
											verbose=verbose0, debug=0, swath=swath, body=body)
		#
		# zz[I,J] = np.percentile(zz0,p0)
		zz_out.append(zz0)

	return zz_out



def orbit_illumination(alt, inc_deg, JD0_start, num_years, dJD_des=1.0, \
						swath=0.0, Nx=50, Ny=50, body='moon+sun', debug=0,
						verbose=1):
#
	# Non-SSO illumination performance for given alt/inc

	from leocat.orb import LEO
	from tqdm import tqdm
	from leocat.utils.plot import make_fig
	from leocat.utils.time import jd_to_date

	I_GT = GroundTrackIllumination(body=body)

	JD_range = np.linspace(JD0_start, JD0_start + 365, Nx)
	LAN0_range = np.linspace(0, 360, Ny)

	xx, yy = np.meshgrid(JD_range-JD0_start,LAN0_range)
	zz = np.zeros(xx.shape)


	iterator = range(len(xx))
	if verbose and (debug <= 1):
		iterator = tqdm(iterator)

	for I in iterator:
		for J in range(len(yy)):
			# JD1 = JD_range[I]
			JD1 = xx[I,J]
			JD2_des = JD1 + 365*num_years

			LAN0_deg = yy[I,J]
			orb = LEO(alt, inc_deg, LAN=LAN0_deg)

			I_GT.set_ephemeris(orb, JD1, JD2_des, dJD_des)
			I_est = I_GT.predict(w=swath)

			zz[I,J] = np.sum(I_est)

			if debug > 1:
				days = I_GT.JD_bar - JD1
				I_est0 = I_GT.predict(w=0.0)
				fig, ax = make_fig()
				ax.plot(days, I_est0)
				ax.plot(days, I_est)
				fig.show()
				pause()
				plt.close('all')

		# break

	if debug > 0:
		fig, ax = make_fig()
		ax.pcolormesh(xx,yy,zz)
		ax.set_xlabel('Days since %s' % str(jd_to_date(JD0_start)))
		ax.set_ylabel(r'$\Omega_{0}$' + ' (deg)')
		if body == 'moon+sun' or body == 'lunisolar':
			title = 'Total Nighttime Lunar Illumination'
		elif body == 'sun' or body == 'solar':
			title = 'Total Solar Illumination'
		elif body == 'moon' or body == 'lunar':
			title = 'Total Day/Night Lunar Illumination'
		title += ' over %.1f years' % (num_years)
		ax.set_title(title)
		fig.show()

	# z0 = np.percentile(zz,p0)
	# idx = np.where(zz >= z0)

	return xx, yy, zz



def orbit_illumination_SSO(alt, MLST, JD0_start, num_years, dJD_des=1.0, \
						swath=0.0, N=50, body='moon+sun', debug=0,
						verbose=0):
#
	# Non-SSO illumination performance for given alt/inc

	from leocat.orb import LEO_SSO
	from tqdm import tqdm
	from leocat.utils.plot import make_fig
	from leocat.utils.time import jd_to_date


	I_GT = GroundTrackIllumination(body=body)

	JD_range = np.linspace(JD0_start, JD0_start + 365, N)
	# LAN0_range = np.linspace(0, 360, Ny)

	# xx, yy = np.meshgrid(JD_range-JD0_start,LAN0_range)
	# zz = np.zeros(xx.shape)

	iterator = range(len(JD_range))
	if verbose and (debug <= 1):
		iterator = tqdm(iterator)

	z = []

	for I in iterator:
		JD1 = JD_range[I]
		JD2_des = JD1 + 365*num_years

		orb = LEO_SSO(alt, MLST, JD1)
		I_GT.set_ephemeris(orb, JD1, JD2_des, dJD_des)
		I_est = I_GT.predict(w=swath)

		z.append(np.sum(I_est))

		if debug > 1:
			days = I_GT.JD_bar - JD1
			I_est0 = I_GT.predict(w=0.0)
			fig, ax = make_fig()
			ax.plot(days, I_est0)
			ax.plot(days, I_est)
			fig.show()
			pause()
			plt.close('all')

	z = np.array(z)
	days = JD_range - JD0_start
	if debug > 0:
		fig, ax = make_fig()
		ax.plot(days, z)
		ax.set_xlabel('Days since %s' % str(jd_to_date(JD0_start)))
		ax.set_ylabel('Total Illumination')
		if body == 'moon+sun' or body == 'lunisolar':
			title = 'Total Nighttime Lunar Illumination'
		elif body == 'sun' or body == 'solar':
			title = 'Total Solar Illumination'
		elif body == 'moon' or body == 'lunar':
			title = 'Total Day/Night Lunar Illumination'
		title += ' over %.1f years' % (num_years)
		ax.set_title(title)
		fig.show()

	# z0 = np.percentile(z,p0)
	# idx = np.where(z >= z0)

	return days, z




def poly_fit1(x, a, b):
	return a*x + b
def poly_fit2(x, a, b, c):
	return a*x**2 + b*x + c
def poly_fit3(x, a, b, c, d):
	return a*x**3 + b*x**2 + c*x + d
def poly_fit4(x, a, b, c, d, e):
	return a*x**4 + b*x**3 + c*x**2 + d*x + e

def fit_illumination(elev_range, I, fit, func, debug=0):
	x = np.cos(np.radians(90-elev_range))
	y = I

	elev_int = -90.0
	popt, pcov = curve_fit(func, x, y)
	if fit == 1:
		a, b = popt
		x_int = -b/a
		gamma_int = np.arccos(x_int)
		elev_int = 90.0 - np.degrees(gamma_int)

	elif fit == 2:
		a, b, c = popt
		det = b**2 - 4*a*c
		x1 = (-b + np.sqrt(det))/(2*a)
		# x2 = (-b - np.sqrt(det))/(2*a)
		x_int = x1
		gamma_int = np.arccos(x_int)
		elev_int = 90.0 - np.degrees(gamma_int)

	elif fit == 3:
		a, b, c, d = popt
		roots_any = np.array(single_cubic(a,b,c,d))
		roots_real = roots_any.real
		b = np.isreal(roots_any)
		roots = roots_real[b]
		idx = np.argmin(np.abs(roots))
		x_int = roots[idx] # not robust
		gamma_int = np.arccos(x_int)
		elev_int = 90.0 - np.degrees(gamma_int)

	elif fit == 4:
		a, b, c, d, e = popt
		roots_any = np.array(single_quartic(a,b,c,d,e))
		roots_real = roots_any.real
		b = np.isreal(roots_any)
		roots = roots_real[b]
		idx = np.argmin(np.abs(roots))
		x_int = roots[idx] # not robust
		gamma_int = np.arccos(x_int)
		elev_int = 90.0 - np.degrees(gamma_int)


	if debug:
		import matplotlib.pyplot as plt
		from leocat.utils.plot import make_fig
		from leocat.utils.general import pause

		fig, ax = make_fig()
		ax.plot(x, y, '.')
		ax.plot(x, func(x, *popt), '--')
		ax.plot(x_int, func(x_int, *popt), 'rx')
		ax.grid()
		fig.show()

		# sys.exit()
		pause()
		plt.close(fig)

	return popt, elev_int

@njit
def get_M_bounds_numba(p_hat, q_hat, h_hat, c_hat, elev_threshold, M_bounds_lat, dx=0.0, tol=1e-6, psi=0.0):

	cx = (p_hat[0]*c_hat[0] + p_hat[1]*c_hat[1] + p_hat[2]*c_hat[2]) * np.cos(psi)
	cy = (q_hat[0]*c_hat[0] + q_hat[1]*c_hat[1] + q_hat[2]*c_hat[2]) * np.cos(psi)
	cz = (h_hat[0]*c_hat[0] + h_hat[1]*c_hat[1] + h_hat[2]*c_hat[2]) * np.sin(psi)
	if psi != 0.0:
		dx = dx - cz

	c = np.sqrt(cx**2 + cy**2)
	gamma_thres = np.pi/2 - np.radians(elev_threshold)
	gamma1 = np.arccos(c-dx)
	gamma2 = np.arccos(-c-dx)
	gamma_min = np.arccos(c-dx)
	gamma_max = np.arccos(-c-dx)
	if gamma_min > gamma_max:
		gamma_min, gamma_max = gamma_max, gamma_min

	# gamma_min = np.arccos(c-dx)
	# gamma_max = np.arccos(-c-dx)
	# if gamma_max < gamma_min:
	# 	print('warning: M_bounds (1)')
	# 	gamma_min, gamma_max = gamma_max, gamma_min

	# EL0 = 0.0
	# if dx != 0.0:
	# 	gamma0 = np.arccos(-dx)
	# 	EL0 = np.pi/2 - gamma0

	if (np.abs(cx) < tol) & (np.abs(cy) < tol):
		# print('tol')
		check = 0
		if gamma_min > gamma_thres:
			M_bounds = np.array([[0.0,0.0]])
			check += 1
		if gamma_max < gamma_thres:
			M_bounds = np.array([[0.0,2*np.pi]])
			check += 1
		if check == 2:
			print('warning: M_bounds (2)')

		# if elev_threshold > EL0:
		# 	M_bounds = np.array([[0.0,2*np.pi]])
		# else:
		# 	M_bounds = np.array([[0.0,0.0]])
		return M_bounds

	# c = np.sqrt(cx**2 + cy**2)
	# gamma_thres = np.pi/2 - np.radians(elev_threshold)
	arg = (np.cos(gamma_thres)+dx)/c
	if np.abs(arg) > 1:
		# print('arg')
		# print(p_hat)
		# print(q_hat)
		# print(h_hat)
		# print(c_hat)
		# print(elev_threshold)
		# print(M_bounds_lat)
		# print(tol)
		# print(psi)
		# print('')

		# sys.exit()

		"""
		No M s.t. gamma == gamma_thres
		If m_hat == h_hat, we have that condition covered.
		Imagine m_hat == h_hat so all elevations are 90 deg.
		exactly, and EL_thres = 30 deg. Then there is no M 
		s.t. EL=EL_thres. This case is handled b/c my=0. However,
		if m_hat is tilted away from h_hat by 1 deg, then the 
		EL angles will vary from 89 to 91 deg. Again, there is 
		no M s.t. EL = 30 deg (EL_thres). In fact, I think
		m_hat must be tilted off by more than np.abs(EL_thres),
		otherwise there's no solution.

		"""
		# if elev_threshold > EL0:
		# 	M_bounds = np.array([[0.0,2*np.pi]])
		# else:
		# 	M_bounds = np.array([[0.0,0.0]])
		check = 0
		if gamma_min > gamma_thres:
			M_bounds = np.array([[0.0,0.0]])
			check += 1
		if gamma_max < gamma_thres:
			M_bounds = np.array([[0.0,2*np.pi]])
			check += 1
		if check == 2:
			print('warning: M_bounds (3)')

		return M_bounds

	alpha = np.arctan2(cx,cy)
	M1 = (np.pi/2-alpha) % (2*np.pi)
	M2 = (-np.pi/2-alpha) % (2*np.pi)
	# x1 = c*np.sin(alpha+M1) - dx
	# x2 = c*np.sin(alpha+M2) - dx
	x1 = np.cos(gamma1)
	x2 = np.cos(gamma2)
	if x1 > x2:
		# M1 is at maximum
		M_opt = M1
	elif x1 < x2:
		# M2 is at maximum
		M_opt = M2
	else:
		# M1 == M2
		# should not be possible?
		M_opt = M1
	#

	# exact
	angle = np.arcsin(arg)
	M_sol1 = (angle-alpha) % (2*np.pi)
	M_sol2 = ((np.pi-angle)-alpha) % (2*np.pi)
	# M_lower, M_upper = np.sort([M_sol1, M_sol2])
	M_lower, M_upper = M_sol1, M_sol2
	if M_lower > M_upper:
		M_lower, M_upper = M_upper, M_lower

	if M_lower < M_opt < M_upper:
		M_bounds = np.array([[M_lower, M_upper]])
	else:
		M_bounds = np.array([[0.0,M_lower],[M_upper,2*np.pi]])

	if len(M_bounds_lat) > 0:
		M_bounds = mask_intervals_numba(M_bounds, M_bounds_lat)

	return M_bounds


@njit
def integrate_illumination_numba(p_hat, q_hat, h_hat, c_hat, popt, M_bounds, fit, dx=0.0, p_bar_factor=1.0, psi=0.0):
	"""
	p_hat, q_hat, c_hat
	popt_k
	M_bounds
	p_factor_k

	sin 1st to 3rd:
	https://en.wikipedia.org/wiki/List_of_integrals_of_trigonometric_functions
	sin 4th:
	https://www.wolframalpha.com/input?i=integrate+sin%28x%29%5E4+dx

	"""
	# cx = p_hat[0]*c_hat[0] + p_hat[1]*c_hat[1] + p_hat[2]*c_hat[2]
	# cy = q_hat[0]*c_hat[0] + q_hat[1]*c_hat[1] + q_hat[2]*c_hat[2]
	cx = (p_hat[0]*c_hat[0] + p_hat[1]*c_hat[1] + p_hat[2]*c_hat[2]) * np.cos(psi)
	cy = (q_hat[0]*c_hat[0] + q_hat[1]*c_hat[1] + q_hat[2]*c_hat[2]) * np.cos(psi)
	cz = (h_hat[0]*c_hat[0] + h_hat[1]*c_hat[1] + h_hat[2]*c_hat[2]) * np.sin(psi)
	if psi != 0.0:
		dx = dx - cz

	c = np.sqrt(cx**2 + cy**2)
	alpha = np.arctan2(cx,cy)

	a0, a1, a2, a3, a4 = np.zeros(5)
	if fit == 1:
		a1, a0 = popt
	elif fit == 2:
		a2, a1, a0 = popt
	elif fit == 3:
		a3, a2, a1, a0 = popt
	elif fit == 4:
		a4, a3, a2, a1, a0 = popt

	if dx != 0.0:
		d0 = -a1*dx + a2*dx**2 - a3*dx**3 + a4*dx**4
		d1 = -2*a2*dx + 3*a3*dx**2 - 4*a4*dx**3
		d2 = -3*a3*dx + 6*a4*dx**2
		d3 = -4*a4*dx

	int_x1 = 0.0
	int_x2 = 0.0
	int_x3 = 0.0
	int_x4 = 0.0
	I_total = 0.0
	for reg in M_bounds:
		M1, M2 = reg
		arg1 = alpha + M1
		arg2 = alpha + M2

		if fit > 0:
			X1 = -c*np.cos(arg1)
			X2 = -c*np.cos(arg2)
			int_x1 = X2 - X1

		if fit > 1:
			X1 = c**2 * (arg1/2 - 1/4*np.sin(2*arg1))
			X2 = c**2 * (arg2/2 - 1/4*np.sin(2*arg2))
			int_x2 = X2 - X1

		if fit > 2:
			X1 = c**3 * (np.cos(3*arg1)/12 - 3*np.cos(arg1)/4)
			X2 = c**3 * (np.cos(3*arg2)/12 - 3*np.cos(arg2)/4)
			int_x3 = X2 - X1

		if fit > 3:
			X1 = c**4 * (3*arg1/8 - np.sin(2*arg1)/4 + np.sin(4*arg1)/32)
			X2 = c**4 * (3*arg2/8 - np.sin(2*arg2)/4 + np.sin(4*arg2)/32)
			int_x4 = X2 - X1

		#
		int_Phi0 = a0*(M2-M1) + a1*int_x1 + a2*int_x2 + a3*int_x3 + a4*int_x4
		int_Phi = int_Phi0
		if dx != 0.0:
			int_dPhi = d0*(M2-M1) + d1*int_x1 + d2*int_x2 + d3*int_x3
			int_Phi = int_Phi + int_dPhi

		#
		I_total = I_total + int_Phi
		# print(2.1, a0, -c*np.cos(arg1))
		# print(2.2, a1, -c*np.cos(arg2))

	#
	return I_total * p_bar_factor


@njit
def mask_intervals_numba(A, B):
	# Convert list of intervals to a flattened NumPy array for efficient processing
	# if not (type(A) is np.ndarray):
	# 	A = np.array(A)
	# if not (type(B) is np.ndarray):
	# 	B = np.array(B)

	A_flat = A.flatten()
	B_flat = B.flatten()
	# A_flat = A.reshape(-1)
	# B_flat = B.reshape(-1)
	# A_flat, B_flat = A, B
	
	# Placeholder for the results, using a list initially (will convert to array later)
	result_flat = []
	
	# Process each interval in A
	for i in range(0, len(A_flat), 2):
		a_start, a_end = A_flat[i], A_flat[i+1]
		current_intervals = [(a_start, a_end)]
		
		# Check against all intervals in B
		for j in range(0, len(B_flat), 2):
			b_start, b_end = B_flat[j], B_flat[j+1]
			new_intervals = []
			
			for a_start, a_end in current_intervals:
				# Check for overlap and adjust intervals accordingly
				if b_start <= a_end and b_end >= a_start:
					if a_start < b_start:
						new_intervals.append((a_start, min(b_start, a_end)))
					if a_end > b_end:
						new_intervals.append((max(a_start, b_end), a_end))
				else:
					new_intervals.append((a_start, a_end))
			current_intervals = new_intervals
		
		# Flatten the current_intervals for the result
		for start, end in current_intervals:
			result_flat.extend([start, end])
	
	# Convert the flat result list back to a 2D array
	result = np.array(result_flat).reshape(-1, 2)
	return result


# @njit
# def mask_M_bounds_lat(M_bounds, M_bounds_lat):
# 	"""
# 	Just limits M_bounds to within the M_bounds_lat interval

# 	"""
# 	M_bounds_mask = []
# 	M_min, M_max = M_bounds_lat
# 	for reg in M_bounds:
# 		lower = min([M_min,reg[0]])
# 		upper = max([reg[1],M_max])
# 		if lower < upper:
# 			# not inc equality since that's 
# 			# a single pt
# 			reg_new = [lower,upper]
# 			M_bounds_mask.append(reg_new)

# 	M_bounds_mask = np.array(M_bounds_mask)
# 	return M_bounds_mask


@njit
def solar_illumination_numba(p_hat, q_hat, h_hat, s_hat, elev_threshold_solar_fit, popt_solar, fit, M_bounds_lat, psi=0.0):
	Is_int_vec = np.zeros(len(p_hat))
	for k in range(len(p_hat)):
		M_bounds_solar = get_M_bounds_numba(p_hat[k], q_hat[k], h_hat[k], s_hat[k], elev_threshold_solar_fit, M_bounds_lat, psi=psi)
		Is_bar_int_total = integrate_illumination_numba(p_hat[k], q_hat[k], h_hat[k], s_hat[k], popt_solar, M_bounds_solar, fit, psi=psi)
		Is_int_vec[k] = Is_bar_int_total
	return Is_int_vec


@njit
def lunar_illumination_numba(p_hat, q_hat, h_hat, m_hat, elev_threshold_lunar_fit, popt_lunar, fit, dx, p_bar_factor, M_bounds_lat, psi=0.0):
	Im_int_vec = np.zeros(len(p_hat))
	for k in range(len(p_hat)):
		M_bounds_lunar = get_M_bounds_numba(p_hat[k], q_hat[k], h_hat[k], m_hat[k], elev_threshold_lunar_fit[k], M_bounds_lat, dx=dx[k], psi=psi)
		Im_bar_int_total = integrate_illumination_numba(p_hat[k], q_hat[k], h_hat[k], m_hat[k], popt_lunar[k], \
																M_bounds_lunar, fit, dx[k], p_bar_factor[k], psi=psi)
		#
		Im_int_vec[k] = Im_bar_int_total
	return Im_int_vec


@njit
def lunisolar_illumination_numba(p_hat, q_hat, h_hat, m_hat, s_hat, elev_threshold_lunar_fit, elev_threshold_solar,
									popt_lunar, fit, dx, p_bar_factor, M_bounds_lat, psi=0.0):
	#
	"""
	If you mask by latitude band first, then you'll have potentially 2 regions from
	M_bounds_lunar masked with potentially 3 regions from M_bounds_lat, which can
	produce 6 regions
	If you also mask solar, then that's another 2 regions from M_bounds_solar_true and
	3 regions from M_bounds_lat, producing up to 6 regions

	THEN you mask M_bounds_lunar (up to 6) by M_bounds_solar_true (up to 6) which could
	produce 36 regions?

	OR
	you first mask without latitude band, so that's 2 for M_bounds_lunar and M_bounds_solar_true
	then you mask M_bounds_lunar by M_bounds_solar_true making up to 4 regions. Then you
	mask by latitude band, which is up to 4*3 = 12 regions.

	I'm not sure how those are different. I think you might only end up with 12 regions
	even if you did it the other way just because of how they overlap.

	"""
	# print(M_bounds_lat, repr(M_bounds_lat))
	# print(np.empty(0,dtype=np.float64), repr(np.empty(0,dtype=np.float64)))
	# print(np.array([0.0]), repr(np.array([0.0])))
	# print(repr(np.array([],dtype=np.float64)))
	Ims_int_vec = np.zeros(len(p_hat))
	for k in range(len(p_hat)):
		# M_bounds_lunar = get_M_bounds_numba(p_hat[k], q_hat[k], m_hat[k], elev_threshold_lunar_fit[k], M_bounds_lat, dx=dx[k])
		# M_bounds_solar_true = get_M_bounds_numba(p_hat[k], q_hat[k], s_hat[k], elev_threshold_solar, M_bounds_lat)
		M_bounds_lunar = get_M_bounds_numba(p_hat[k], q_hat[k], h_hat[k], m_hat[k], elev_threshold_lunar_fit[k], np.empty(0,dtype=np.float64), dx=dx[k], psi=psi)
		M_bounds_solar_true = get_M_bounds_numba(p_hat[k], q_hat[k], h_hat[k], s_hat[k], elev_threshold_solar, np.empty(0,dtype=np.float64), psi=psi)
		# A_flat = M_bounds_lunar.flatten()
		# B_flat = M_bounds_solar_true.flatten()
		# M_bounds_lunisolar = mask_intervals_numba(A_flat, B_flat)
		M_bounds_lunisolar = mask_intervals_numba(M_bounds_lunar, M_bounds_solar_true)
		if len(M_bounds_lat) > 0:
			M_bounds_lunisolar = mask_intervals_numba(M_bounds_lunisolar, M_bounds_lat)
		Ims_bar_int_total = integrate_illumination_numba(p_hat[k], q_hat[k], h_hat[k], m_hat[k], popt_lunar[k], \
																M_bounds_lunisolar, fit, dx[k], p_bar_factor[k], psi=psi)
		#
		Ims_int_vec[k] = Ims_bar_int_total
	return Ims_int_vec



def get_validation_range(M_start, M_end, JD_start, JD_end, dt_bar, LAN_dot, num, M_bounds_lat):
	M_range = np.linspace(M_start, M_end, num)
	JD_range = np.linspace(JD_start, JD_end, num)
	LAN_range = np.linspace(-dt_bar/2*LAN_dot, dt_bar/2*LAN_dot, num)
	if len(M_bounds_lat) > 0:
		M_range_wrap = M_range % (2*np.pi)
		# b = []
		# for i,val in enumerate(M_range_wrap):
		# 	for reg in M_bounds_lat:
		# 		if not (reg[0] < val < reg[1]):
		# 			idx.append(i)
		# 			break
		# idx = np.array(idx)
		# b_vec = []
		# for reg in M_bounds_lat:
		# 	M_min, M_max = reg
		# 	b_reg = ~((M_range_wrap >= M_min) & (M_range_wrap <= M_max))
		# 	b_vec.append(b_reg)

		b = np.ones(num).astype(bool)
		for reg in M_bounds_lat:
			M_min, M_max = reg
			b_reg = ~((M_range_wrap >= M_min) & (M_range_wrap <= M_max))
			b = b & b_reg

		# b = np.zeros(num).astype(bool)
		# for reg in M_bounds_lat:
		# 	M_min, M_max = reg
		# 	b = b | ((M_range_wrap >= M_min) & (M_range_wrap <= M_max))
		# M_min, M_max = M_bounds_lat
		# M_range_wrap = M_range % (2*np.pi)
		# b = (M_range_wrap >= M_min) & (M_range_wrap <= M_max)
		M_range = M_range[b]
		JD_range = JD_range[b]
		LAN_range = LAN_range[b]
	return M_range, JD_range, LAN_range


def get_track_position(M_range, R_313, LAN_range=None, psi=0.0):
	r_mag = R_earth
	r_pfc = np.transpose([r_mag*np.cos(M_range)*np.cos(psi), 
							r_mag*np.sin(M_range)*np.cos(psi), 
							np.zeros(len(M_range)) + r_mag*np.sin(psi)])
	#
	r_eci = (R_313 @ r_pfc.T).T
	r_eci_J2 = r_eci
	if LAN_range is not None:
		r_eci_J2 = matmul(R3(LAN_range), r_eci)
	return r_eci_J2


# solar_illumination_direct
# lunar_illumination_direct
# solar_illumination_direct_bar
# lunar_illumination_direct_bar

def solar_illumination_direct(r_hat, JD_range, func, elev_threshold_solar_fit, popt_solar, elev_threshold_solar):
	r_gt = R_earth*r_hat
	r_sun = solar_pos_approx(JD_range)
	r_sun_rel = r_sun-r_gt
	c_hat = unit(r_sun_rel)

	x_range = dot(r_hat,c_hat)
	gamma_range = np.arccos(x_range)
	elev_range = 90.0 - np.degrees(gamma_range)
	I_true = surface_illumination(elev_range, JD_range, body='sun', phase=None, return_log=False)
	I_true[elev_range < elev_threshold_solar] = 0.0

	I_fit = func(x_range, *popt_solar)
	I_fit[elev_range < elev_threshold_solar_fit] = 0.0

	return I_true, I_fit


def lunar_illumination_direct(r_hat, JD_range, func, fit_data_popt, fit_data_elev_int, elev_threshold_lunar):
	r_gt = R_earth*r_hat
	r_moon = lunar_pos_approx(JD_range)
	r_moon_rel = r_moon-r_gt
	c_hat = unit(r_moon_rel)
	phase_range = lunar_phase(JD_range)

	x_range = dot(r_hat,c_hat)
	gamma_range = np.arccos(x_range)
	elev_range = 90.0 - np.degrees(gamma_range)
	I_true = surface_illumination(elev_range, JD_range, body='moon', phase=phase_range, return_log=False)
	I_true[elev_range < elev_threshold_lunar] = 0.0

	num = len(JD_range)
	I_fit = np.zeros(num)
	phase_range_int = np.round(phase_range).astype(int)
	popt_lunar = fit_data_popt[phase_range_int]
	elev_int_lunar = fit_data_elev_int[phase_range_int]
	for j in range(num):
		elev_threshold_lunar_fit = elev_threshold_lunar
		if elev_threshold_lunar_fit < elev_int_lunar[j]:
			elev_threshold_lunar_fit = elev_int_lunar[j]
		if elev_range[j] < elev_threshold_lunar_fit:
			continue
		else:
			I_fit[j] = func(x_range[j], *popt_lunar[j])

	p_range = get_lunar_p(JD_range)
	p_factor_range = (p_range/0.951)**2
	I_fit = I_fit * p_factor_range

	return I_true, I_fit


def solar_illumination_direct_bar(p_hat, q_hat, h_hat, c_hat, M_range, JD_bar, func, popt_solar, elev_threshold_solar_fit, elev_threshold_solar, psi=0.0):
	num = len(M_range)
	JD_bar_range = np.full(num,JD_bar)

	cx = np.dot(p_hat,c_hat) * np.cos(psi)
	cy = np.dot(q_hat,c_hat) * np.cos(psi)
	cz = np.dot(h_hat,c_hat) * np.sin(psi)
	c = np.sqrt(cx**2 + cy**2)
	alpha = np.arctan2(cx,cy)

	x_range = c*np.sin(alpha + M_range) + cz
	x_range[x_range < -1] = -1.0
	x_range[x_range > 1] = 1.0
	gamma_range = np.arccos(x_range)
	elev_range = 90.0 - np.degrees(gamma_range)
	I_bar = surface_illumination(elev_range, JD_bar_range, body='sun', phase=None, return_log=False)
	I_bar[elev_range < elev_threshold_solar] = 0.0

	I_bar_fit = func(x_range, *popt_solar)
	I_bar_fit[elev_range < elev_threshold_solar_fit] = 0.0

	return I_bar, I_bar_fit


def lunar_illumination_direct_bar(p_hat, q_hat, h_hat, c_hat, M_range, JD_bar, dx, func, fit_data_popt, fit_data_elev_int, elev_threshold_lunar, psi=0.0):
	num = len(M_range)
	JD_bar_range = np.full(num,JD_bar)

	phase_bar = lunar_phase(JD_bar)
	phase_bar_range = np.full(num,phase_bar)

	# cx = np.dot(p_hat,c_hat)
	# cy = np.dot(q_hat,c_hat)
	cx = np.dot(p_hat,c_hat) * np.cos(psi)
	cy = np.dot(q_hat,c_hat) * np.cos(psi)
	cz = np.dot(h_hat,c_hat) * np.sin(psi)
	c = np.sqrt(cx**2 + cy**2)
	alpha = np.arctan2(cx,cy)

	x_range = c*np.sin(alpha + M_range) - dx + cz
	x_range[x_range < -1] = -1.0
	x_range[x_range > 1] = 1.0
	gamma_range = np.arccos(x_range)
	elev_range = 90.0 - np.degrees(gamma_range)
	I_bar = surface_illumination(elev_range, JD_bar_range, body='moon', phase=phase_bar_range, return_log=False)
	I_bar[elev_range < elev_threshold_lunar] = 0.0

	I_bar_fit = np.zeros(num)
	phase_bar_int = int(np.round(phase_bar))
	popt_lunar = fit_data_popt[phase_bar_int]
	elev_int_lunar = fit_data_elev_int[phase_bar_int]

	elev_threshold_lunar_fit = elev_threshold_lunar
	if elev_threshold_lunar_fit < elev_int_lunar:
		elev_threshold_lunar_fit = elev_int_lunar

	for j in range(num):
		if elev_range[j] < elev_threshold_lunar_fit:
			continue
		else:
			I_bar_fit[j] = func(x_range[j], *popt_lunar)

	p_bar = get_lunar_p(JD_bar)
	p_bar_factor = (p_bar/0.951)**2
	# I_bar_est = I_bar_est * p_bar_factor
	I_bar_fit = I_bar_fit * p_bar_factor

	return I_bar, I_bar_fit



def integrate_illumination_numerical(I, N_orb):
	return (N_orb/2*np.pi * (R_earth/1e3)**2)/len(I) * np.sum(I)
	# 2pi/len(I) = dM in numerical integration


def lat_band_to_M_bounds(lat_band, inc, omega, invert=False):
	# Not strictly correct if psi != 0

	phi_min, phi_max = np.radians(lat_band)
	phi_inc = inc
	if phi_inc > np.pi/2:
		phi_inc = np.pi - phi_inc

	def phi_to_u(phi):
		arg = np.sin(phi)/np.sin(inc)
		u0 = np.arcsin(arg)
		u1 = u0 % (2*np.pi)
		u2 = (np.pi - u0) % (2*np.pi)
		return np.sort([u1,u2])

	def u_to_phi(u):
		# output phi bounded between
		#	[-pi/2,pi/2], no ambiguity
		arg = np.sin(inc)*np.sin(u)
		phi0 = np.arcsin(arg)
		phi1 = phi0
		# phi2 = np.pi - phi0
		# return np.sort([phi1,phi2])
		return phi1

	#
	def merge_regions(b_int):
		intervals = []  # To store the start,end intervals
		in_sequence = False  # Flag to track if we are in a sequence of 1's
		start_index = None  # To remember the start of a sequence
		matrix = np.transpose([np.arange(len(b_int)), b_int])
		
		for index, value in matrix:
			if value == 1 and not in_sequence:
				# Start of a new sequence
				in_sequence = True
				start_index = index
			elif value == 0 and in_sequence:
				# End of the current sequence
				in_sequence = False
				intervals.append([start_index, index - 1])
		
		# If the last value in the matrix is a 1, close the last sequence
		if in_sequence:
			intervals.append([start_index, matrix[-1][0]])
		
		return intervals

	if -phi_inc <= phi_min < phi_inc:
		u_min = phi_to_u(phi_min)
	else:
		u_min = np.array([np.pi,2*np.pi])

	if -phi_inc < phi_max <= phi_inc:
		u_max = phi_to_u(phi_max)
	else:
		u_max = np.array([0.0,np.pi])

	M_min = (u_min - omega) % (2*np.pi)
	M_max = (u_max - omega) % (2*np.pi)
	M_vec = np.sort(np.hstack((M_min,M_max)))
	if M_vec[0] > 0.0:
		M_vec = np.hstack((0.0, M_vec))
	if M_vec[-1] < 2*np.pi:
		M_vec = np.hstack((M_vec, 2*np.pi))

	M_reg = []
	for i in range(1,len(M_vec)):
		M1, M2 = M_vec[i-1], M_vec[i]
		M_reg.append([M1,M2])
	M_reg = np.array(M_reg)
	M_mid = np.mean(M_reg,axis=1) # pt in middle of each region

	# classify regions as within lat bounds or not using M_mid
	#	adjacent regions will be merged
	u_mid = (M_mid + omega) % (2*np.pi)
	phi_mid = u_to_phi(u_mid)
	b = (phi_mid > phi_min) & (phi_mid < phi_max)
	if invert:
		b = ~b

	intervals = merge_regions(b.astype(int))
	M_bounds_lat = []
	for intv in intervals:
		i1, i2 = intv
		M_bounds_lat.append([M_reg[i1][0],M_reg[i2][1]])
	M_bounds_lat = np.array(M_bounds_lat)

	return M_bounds_lat


def get_illum(r_ecf, JD, N_orb, JD1, dJD, body='sun'):

	from leocat.utils.geodesy import ecf_to_lla
	from leocat.utils.orbit import solar_elev, lunar_elev
	from leocat.utils.index import hash_index
	import pandas as pd

	lunisolar = (body == 'lunisolar' or body == 'moon+sun')

	lon, lat, _ = ecf_to_lla(r_ecf.T[0], r_ecf.T[1], r_ecf.T[2])
	if not lunisolar:
		phase = None
		if body == 'sun':
			elev = solar_elev(lon, lat, JD)
		elif body == 'moon':
			phase = lunar_phase(JD)
			elev = lunar_elev(lon, lat, JD)
		Is = surface_illumination(elev, JD, body=body, phase=phase, return_log=False)
		Is = Is * N_orb*2*np.pi

		I_rtn = Is

	else:
		phase = lunar_phase(JD)
		elev_s = solar_elev(lon, lat, JD)
		elev_m = lunar_elev(lon, lat, JD)
		Is = surface_illumination(elev_s, JD, body='sun', phase=None, return_log=False)
		Im = surface_illumination(elev_m, JD, body='moon', phase=phase, return_log=False)
		Im[Is > 0] = 0.0
		Im = Im * N_orb*2*np.pi

		I_rtn = Im


	cols = hash_index(JD, JD1, dJD)
	JD_bar2 = JD1 + np.arange(np.max(cols)+1)*dJD + dJD/2
	# t_bar2 = (JD_bar2-JD1)*86400

	df = pd.DataFrame({'c': cols, 'I': I_rtn})
	df = df.groupby('c').agg({'I': 'mean'})
	df = df.reset_index()
	Is_true = df['I'].to_numpy()

	return Is_true




class GroundTrackIllumination():
	def __init__(self, body='sun', elev_threshold_solar=elev_threshold_solar0,
					elev_threshold_lunar=elev_threshold_lunar0, lat_band=[], fit=3, warn=True):
		#

		body = body.lower()
		if body == 'solar':
			body = 'sun'
		if body == 'lunar':
			body = 'moon'
		if body == 'lunisolar':
			body = 'moon+sun'

		self.warn = warn
		if elev_threshold_solar < elev_threshold_solar0 and self.warn:
			import warnings
			warnings.warn(f'Given solar elev threshold of {elev_threshold_solar} set to minimum of {elev_threshold_solar0} deg')
			elev_threshold_solar = elev_threshold_solar0
		if elev_threshold_lunar < elev_threshold_lunar0 and self.warn:
			import warnings
			warnings.warn(f'Given lunar elev threshold of {elev_threshold_lunar} set to minimum of {elev_threshold_lunar0} deg')
			elev_threshold_lunar = elev_threshold_lunar0

		self.elev_threshold_solar = elev_threshold_solar
		self.elev_threshold_lunar = elev_threshold_lunar
		if len(lat_band) > 0:
			self.lat_band = np.sort(np.array(lat_band))
			if np.ptp(self.lat_band) < 1e-6:
				raise Exception('Latitude band min/max are nearly/exactly equal')
		else:
			self.lat_band = np.array([])

		self.body = body
		self.fit = fit
		self.load_fits()


	def load_fits(self):
		fit = self.fit
		body = self.body

		if fit == 1:
			func = poly_fit1
		elif fit == 2:
			func = poly_fit2
		elif fit == 3:
			func = poly_fit3
		elif fit == 4:
			func = poly_fit4
		self.func = func

		if body == 'moon' or body == 'moon+sun':
			phase_range = np.arange(180+1)
			fit_data_popt = np.zeros((len(phase_range),fit+1))
			fit_data_elev_int = np.zeros(len(phase_range))
			for phase in phase_range:
				elev_range = np.linspace(elev_threshold_lunar0,90)
				I_lunar = surface_illumination_approx(elev_range, phase=phase, return_log=False)
				popt_lunar, elev_int_lunar = fit_illumination(elev_range, I_lunar, fit, func, debug=0)
				fit_data_popt[phase,:] = popt_lunar
				fit_data_elev_int[phase] = elev_int_lunar

			self.fit_data_popt = fit_data_popt
			self.fit_data_elev_int = fit_data_elev_int

		if body == 'sun' or body == 'moon+sun':
			elev_range = np.linspace(elev_threshold_solar0,90)
			I_solar = surface_illumination_approx(elev_range, return_log=False)
			popt_solar, elev_int_solar = fit_illumination(elev_range, I_solar, fit, func, debug=0)

			elev_threshold_solar_fit = float(np.copy(self.elev_threshold_solar))
			if elev_threshold_solar_fit < elev_int_solar:
				elev_threshold_solar_fit = elev_int_solar

			self.elev_threshold_solar_fit = elev_threshold_solar_fit
			self.popt_solar = popt_solar
			self.elev_int_solar = elev_int_solar


	# def set_ephemeris(self, a, e, inc, omega, LAN0, M0, JD1, JD2_des, dJD_des):
	def set_ephemeris(self, orb, JD1, JD2_des, dJD_des):

		self.orb = orb
		e = orb.e
		if e != 0.0 and self.warn:
			import warnings
			warnings.warn('Illumination results may be inaccurate for e != 0.0')

		body = self.body

		a = orb.a
		inc = orb.inc
		omega = orb.omega
		LAN0 = orb.LAN
		M0 = orb.M0

		self.a = a
		self.e = e
		self.inc = inc
		self.omega = omega
		self.LAN0 = LAN0
		self.M0 = M0

		# JD1 = date_to_jd(2023,1,1)
		# JD2_des = JD1 + 16 # upper bound, desired end
		# dJD_des = 1.0 # desired interval (days)

		# n = np.sqrt(MU/a**3)
		# P = calc_period(MU,a=a)
		M_dot = orb.get_M_dot()
		P = orb.get_period('nodal')
		# P = orb.get_period('kepler')
		# LAN_dot = get_LAN_dot(a, e, inc) # +/-5% for all other harmonics
		LAN_dot = orb.get_LAN_dot()
		N_orb = int(np.round(dJD_des*86400/P)) # num orbits to process at once
		if N_orb == 0:
			# print('dJD set N_orb to zero, setting N_orb = 1')
			N_orb = 1

		N_total = int( (JD2_des-JD1)*86400 / P ) # num orbits over entire sim period
		N = int( (JD2_des-JD1)*86400 / (N_orb*P) ) # num separate processes
		N_orb_rem = N_total - N*N_orb # 1 more process for remainder

		dJD = N_orb*P/86400
		JD_bar = JD1 + np.arange(N)*dJD + dJD/2
		JD2 = JD1 + dJD*N
		# JD_start = JD_bar - dJD/2
		# JD_end = JD_bar + dJD/2

		t_bar = 0 + np.arange(N)*(N_orb*P) + (N_orb*P)/2
		# dt_bar = N_orb*P
		# t_start = t_bar - (N_orb*P)/2
		# t_end = t_bar + (N_orb*P)/2

		LAN_bar = LAN0 + LAN_dot*t_bar
		# M_bar = M0 + n*t_bar
		M_bar = M0 + M_dot*t_bar
		self.t_bar = t_bar
		self.M_dot = M_dot

		self.LAN_dot = LAN_dot
		self.P = P
		self.dJD = dJD
		self.JD1 = JD1
		self.JD2 = JD2

		self.N = N
		self.N_orb = N_orb
		self.JD_bar = JD_bar
		self.LAN_bar = LAN_bar
		self.M_bar = M_bar

		R_LAN = R3(LAN_bar)
		R_inc = R1(inc)
		R_omega = R3(omega)
		R_313 = R_LAN @ (R_inc @ R_omega)
		self.R_313 = R_313

		p_hat = R_313[:,:,0]
		h_hat = R_313[:,:,2]
		q_hat = np.cross(h_hat,p_hat)
		self.p_hat = p_hat
		self.q_hat = q_hat
		self.h_hat = h_hat

		lat_band = self.lat_band
		M_bounds_lat = np.array([])
		if len(lat_band) > 0:
			# true only for circular orbit, geocentric latitude
			#	from 1 to 3 regions
			#	should account for psi
			M_bounds_lat = lat_band_to_M_bounds(lat_band, inc, omega, invert=True)
			if len(M_bounds_lat) == 0:
				M_bounds_lat = np.array([])
		self.M_bounds_lat = M_bounds_lat

		if body == 'sun' or body == 'moon+sun':
			r_sun = solar_pos_approx(JD_bar)
			s_hat = unit(r_sun)
			self.s_hat = s_hat

		if body == 'moon' or body == 'moon+sun':
			r_moon = lunar_pos_approx(JD_bar)
			r_moon_rel = r_moon
			m_hat = unit(r_moon_rel)

			phase_bar = lunar_phase(JD_bar)
			phase_bar_int = np.round(phase_bar).astype(int)
			p_bar = get_lunar_p(JD_bar)
			p_bar_factor = (p_bar/0.951)**2
			popt_lunar = self.fit_data_popt[phase_bar_int]
			elev_int_lunar = self.fit_data_elev_int[phase_bar_int]

			elev_threshold_lunar_fit = np.full(JD_bar.shape,self.elev_threshold_lunar)
			b = elev_threshold_lunar_fit < elev_int_lunar
			elev_threshold_lunar_fit[b] = elev_int_lunar[b]

			# dx = np.zeros(len(JD_bar))
			dx = R_earth / mag(r_moon)

			self.m_hat = m_hat
			self.p_bar_factor = p_bar_factor
			self.popt_lunar = popt_lunar
			self.elev_threshold_lunar_fit = elev_threshold_lunar_fit
			self.dx = dx

		# if return_metadata:


	def predict(self, w=0.0, dw=500.0):
		Nz = int(w/dw) + 1
		if Nz == 1:
			return self.predict_CT()
		else:
			CT_offsets = np.linspace(-w/2, w/2, Nz)
			I_est_CT = []
			for j,CT_offset in enumerate(CT_offsets):
				I_est = self.predict_CT(CT_offset=CT_offset)
				I_est_CT.append(I_est)
			I_est_CT = np.array(I_est_CT)
			I_est = np.mean(I_est_CT,axis=0)
			return I_est


	def predict_CT(self, CT_offset=0.0):
		body = self.body
		# if body is None:
		# 	body = self.body
		# else:
		# 	if body == 'solar':
		# 		body = 'sun'
		# 	if body == 'lunar':
		# 		body = 'moon'
		# 	if body == 'lunisolar':
		# 		body = 'moon+sun'

		p_hat = self.p_hat
		q_hat = self.q_hat
		N_orb = self.N_orb
		fit = self.fit

		self.psi = CT_offset / R_earth # single value

		if body == 'sun' or body == 'moon+sun':
			s_hat = self.s_hat
			elev_threshold_solar_fit = self.elev_threshold_solar_fit
			popt_solar = self.popt_solar

		if body == 'moon' or body == 'moon+sun':
			m_hat = self.m_hat
			elev_threshold_lunar_fit = self.elev_threshold_lunar_fit
			popt_lunar = self.popt_lunar
			dx = self.dx
			p_bar_factor = self.p_bar_factor

		M_bounds_lat = self.M_bounds_lat
		if body == 'sun':
			I_int_vec = (R_earth/1e3)**2 * N_orb*solar_illumination_numba(p_hat, q_hat, self.h_hat, s_hat, elev_threshold_solar_fit, popt_solar, fit, M_bounds_lat, self.psi)

		elif body == 'moon':
			I_int_vec = (R_earth/1e3)**2 * N_orb*lunar_illumination_numba(p_hat, q_hat, self.h_hat, m_hat, elev_threshold_lunar_fit, popt_lunar, fit, dx, p_bar_factor, M_bounds_lat, self.psi)

		elif body == 'moon+sun':
			I_int_vec = (R_earth/1e3)**2 * N_orb*lunisolar_illumination_numba( \
								p_hat, q_hat, self.h_hat, m_hat, s_hat, elev_threshold_lunar_fit, \
								self.elev_threshold_solar, popt_lunar, fit, dx, p_bar_factor, M_bounds_lat, psi=self.psi)
			#

		return I_int_vec



	def predict_true(self, w=0.0, dw=500.0):

		from leocat.utils.cov import create_swath_simple
		from leocat.utils.orbit import convert_ECI_ECF
		from leocat.utils.geodesy import ecf_to_lla

		JD1 = self.JD1
		JD2 = self.JD2
		dJD = self.dJD
		N_orb = self.N_orb
		num_days = JD2-JD1

		t_bar = self.t_bar
		JD_bar = self.JD_bar
		orb = self.orb

		orb_p_day = orb.get_nodal_day()/orb.get_period('nodal')

		N = int((JD2-JD1)*orb_p_day*100) + 1
		JD = np.linspace(JD1,JD2,N+2)
		JD = (JD[1]-JD[0])/2 + JD[1:-1]
		t = (JD-JD1)*86400

		r_eci, v_eci = orb.propagate(t)
		r_ecf = convert_ECI_ECF(JD, r_eci)
		rg = unit(r_ecf)*R_earth

		# dz = 500
		Nz = int(w/dw) + 1

		lon, lat, _ = ecf_to_lla(rg.T[0], rg.T[1], rg.T[2])

		if Nz > 1:
			mesh_edge_l, mesh_edge_r = create_swath_simple(unit(r_eci), unit(v_eci), w)
			mesh_edge_l = convert_ECI_ECF(JD, mesh_edge_l)
			mesh_edge_r = convert_ECI_ECF(JD, mesh_edge_r)

			tau_c = np.linspace(0,1,Nz)
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
			index_mesh = np.tile(np.arange(len(t)), (len(tau_c),1)).T.flatten()

			# if 0:
			# 	k1 = 0
			# 	k2 = int(0.00003*len(rg))

			# 	fig, ax = make_fig('3d')
			# 	ax.plot(rg[k1:k2].T[0], rg[k1:k2].T[1], rg[k1:k2].T[2])
			# 	ax.plot(mesh_edge_r[k1:k2].T[0], mesh_edge_r[k1:k2].T[1], mesh_edge_r[k1:k2].T[2])
			# 	ax.plot(mesh_edge_l[k1:k2].T[0], mesh_edge_l[k1:k2].T[1], mesh_edge_l[k1:k2].T[2])
			# 	ax.plot(r_mesh[k1:Nz*k2].T[0], r_mesh[k1:Nz*k2].T[1], r_mesh[k1:Nz*k2].T[2], '.')
			# 	set_axes_equal(ax)
			# 	set_aspect_equal(ax)
			# 	fig.show()

			# 	sys.exit()

			# lon, lat, _ = ecf_to_lla(r_mesh.T[0], r_mesh.T[1], r_mesh.T[2])
			rg = r_mesh
			JD = t[index_mesh]/86400 + JD1

		I_true = get_illum(rg, JD, N_orb, JD1, dJD, body=self.body) * (R_earth/1e3)**2

		return I_true


	def predict_true_CT(self, CT_offset=0.0, n_mult=1000, verbose=1, debug=0):
		body = self.body
		# if body is None:
		# 	body = self.body
		# else:
		# 	if body == 'solar':
		# 		body = 'sun'
		# 	if body == 'lunar':
		# 		body = 'moon'
		# 	if body == 'lunisolar':
		# 		body = 'moon+sun'

		JD_bar = self.JD_bar
		M_bar = self.M_bar
		N_orb = self.N_orb
		dJD = self.dJD

		self.psi = CT_offset / R_earth # single value

		JD_start = JD_bar - dJD/2
		JD_end = JD_bar + dJD/2
		M_start = M_bar - (2*np.pi*N_orb)/2
		M_end = M_bar + (2*np.pi*N_orb)/2

		dt_bar = N_orb*self.P
		LAN_dot = self.LAN_dot

		M_bounds_lat = self.M_bounds_lat
		func = self.func

		a = self.a
		R_LAN = R3(self.LAN_bar)
		R_inc = R1(self.inc)
		R_omega = R3(self.omega)
		R_313 = R_LAN @ (R_inc @ R_omega)

		iterator = range(len(JD_bar))
		if verbose:
			from tqdm import tqdm
			iterator = tqdm(range(len(JD_bar)))

		p_hat = self.p_hat
		q_hat = self.q_hat

		if body == 'sun' or body == 'moon+sun':
			elev_threshold_solar_fit = self.elev_threshold_solar_fit
			popt_solar = self.popt_solar
			s_hat = self.s_hat

		if body == 'moon' or body == 'moon+sun':
			fit_data_popt = self.fit_data_popt
			fit_data_elev_int = self.fit_data_elev_int
			dx = self.dx
			m_hat = self.m_hat
			elev_threshold_lunar_fit = self.elev_threshold_lunar_fit


		num = n_mult*N_orb

		if debug:
			import matplotlib.pyplot as plt
			from leocat.utils.general import pause
			from leocat.utils.plot import make_fig, pro_plot
			pro_plot()

		# Is_est = self.predict(body=body)

		Is_vec = []
		Im_vec = []
		Ims_vec = []
		# r_eci_save = []
		# JD_save = []
		for k in iterator:
			# if debug > 1:
			# 	print('%d/%d' % (k+1,len(JD_bar)))
			# 	if k < 281:
			# 		continue
			# 	print(Is_est[k])

			M_range, JD_range, LAN_range = \
				get_validation_range(M_start[k], M_end[k], JD_start[k], JD_end[k], dt_bar, LAN_dot, num, M_bounds_lat)
			# r_eci = get_orbit_position(a, M_range, R_313[k], LAN_range=None) # keplerian
			r_eci = get_track_position(M_range, R_313[k], LAN_range=LAN_range, psi=self.psi) # J2
			r_hat = unit(r_eci)

			# r_eci_save.append(r_eci)
			# JD_save.append(JD_range)

			if body == 'sun' or body == 'moon+sun':
				Is, Is_fit = solar_illumination_direct(r_hat, JD_range, func, elev_threshold_solar_fit, popt_solar, self.elev_threshold_solar)
				Is_bar, Is_bar_fit = solar_illumination_direct_bar(p_hat[k], q_hat[k], self.h_hat[k], s_hat[k], M_range, JD_bar[k], func, popt_solar, elev_threshold_solar_fit, self.elev_threshold_solar, psi=self.psi)
				Is_total = integrate_illumination_numerical(Is, N_orb*len(M_range)/num)
				Is_bar_fit_total = integrate_illumination_numerical(Is_bar_fit, N_orb*len(M_range)/num)
				Is_vec.append(Is_total)

				if debug:
					M_bounds_solar = get_M_bounds_numba(p_hat[k], q_hat[k], self.h_hat[k], s_hat[k], elev_threshold_solar_fit, M_bounds_lat, psi=self.psi)
					if debug > 1:
						print(M_bounds_solar)

					fig, ax = make_fig()
					ax.plot(M_range, Is, label='True')
					ax.plot(M_range, Is_fit, label='Fit')
					ax.plot(M_range, Is_bar, label='True-Bar')
					ax.plot(M_range, Is_bar_fit, label='Fit-Bar')

					ylim = ax.get_ylim()
					for j in range(N_orb):
						shift = M_start[k] + j*2*np.pi
						for reg in M_bounds_solar:
							ax.plot([reg[0]+shift,reg[0]+shift],ylim,'g--')
							ax.plot([reg[1]+shift,reg[1]+shift],ylim,'r--')

					ax.legend(loc='upper left')
					ax.set_xlabel('M (rad)')
					ax.set_ylabel('Solar Intensity')
					ax.set_title('Sun')
					fig.tight_layout()
					fig.show()


			if body == 'moon' or body == 'moon+sun':
				Im, Im_fit = lunar_illumination_direct(r_hat, JD_range, func, fit_data_popt, fit_data_elev_int, self.elev_threshold_lunar)
				Im_bar, Im_bar_fit = lunar_illumination_direct_bar(p_hat[k], q_hat[k], self.h_hat[k], m_hat[k], M_range, JD_bar[k], dx[k], func, fit_data_popt, fit_data_elev_int, self.elev_threshold_lunar, psi=self.psi)
				Im_total = integrate_illumination_numerical(Im, N_orb*len(M_range)/num)
				Im_bar_fit_total = integrate_illumination_numerical(Im_bar_fit, N_orb*len(M_range)/num)
				Im_vec.append(Im_total)

				if debug:
					M_bounds_lunar = get_M_bounds_numba(p_hat[k], q_hat[k], self.h_hat[k], m_hat[k], elev_threshold_lunar_fit[k], M_bounds_lat, dx=dx[k], psi=self.psi)
					fig, ax = make_fig()
					ax.plot(M_range, Im, label='True')
					ax.plot(M_range, Im_fit, label='Fit')
					ax.plot(M_range, Im_bar, label='True-Bar')
					ax.plot(M_range, Im_bar_fit, label='Fit-Bar')

					ylim = ax.get_ylim()
					for j in range(N_orb):
						shift = M_start[k] + j*2*np.pi
						for reg in M_bounds_lunar:
							ax.plot([reg[0]+shift,reg[0]+shift],ylim,'g--')
							ax.plot([reg[1]+shift,reg[1]+shift],ylim,'r--')

					ax.legend(loc='upper left')
					ax.set_xlabel('M (rad)')
					ax.set_ylabel('Lunar Intensity')
					ax.set_title('Moon')
					fig.tight_layout()
					fig.show()


			if body == 'moon+sun':
				Ims = np.copy(Im)
				Ims_fit = np.copy(Im_fit)
				Ims[Is > 0] = 0.0
				Ims_fit[Is > 0] = 0.0

				Ims_bar = np.copy(Im_bar)
				Ims_bar_fit = np.copy(Im_bar_fit)
				Ims_bar[Is_bar > 0] = 0.0
				Ims_bar_fit[Is_bar > 0] = 0.0

				if debug:
					M_bounds_lunar = get_M_bounds_numba(p_hat[k], q_hat[k], self.h_hat[k], m_hat[k], elev_threshold_lunar_fit[k], np.array([]), dx=dx[k], psi=self.psi)
					M_bounds_solar_true = get_M_bounds_numba(p_hat[k], q_hat[k], self.h_hat[k], s_hat[k], self.elev_threshold_solar, np.array([]), psi=self.psi)
					# M_bounds_lunar = get_M_bounds_numba(p_hat[k], q_hat[k], m_hat[k], elev_threshold_lunar_fit[k], np.array([]), dx=dx[k])
					# M_bounds_solar_true = get_M_bounds_numba(p_hat[k], q_hat[k], s_hat[k], self.elev_threshold_solar, np.array([]))
					# A_flat = M_bounds_lunar.flatten()
					# B_flat = M_bounds_solar_true.flatten()
					# M_bounds_lunisolar = mask_intervals_numba(A_flat, B_flat)
					M_bounds_lunisolar = mask_intervals_numba(M_bounds_lunar, M_bounds_solar_true)
					if len(M_bounds_lat) > 0:
						M_bounds_lunisolar = mask_intervals_numba(M_bounds_lunisolar, M_bounds_lat)


					fig, ax = make_fig()
					ax.plot(M_range, Is, label='True sun')
					ax.plot(M_range, Is_bar, label='True-Bar sun')
					ax2 = ax.twinx()
					im = ax2.plot(M_range, Ims, c='C2')
					ax.plot(np.nan, np.nan, label='True lunisolar', c=im[0].get_c())
					im = ax2.plot(M_range, Ims_bar_fit, c='C3')
					ax.plot(np.nan, np.nan, label='Fit-Bar lunisolar', c=im[0].get_c())
					ax.legend(loc='upper left')

					ylim = ax.get_ylim()
					for j in range(N_orb):
						shift = M_start[k] + j*2*np.pi
						for reg in M_bounds_lunisolar:
							ax.plot([reg[0]+shift,reg[0]+shift],ylim,'g--')
							ax.plot([reg[1]+shift,reg[1]+shift],ylim,'r--')

					ax.set_title('Lunar + Non-Solar')
					ax.set_xlabel('M (rad)')
					ax.set_ylabel('Lunar Intensity')
					fig.tight_layout()
					fig.show()


				Ims_total = integrate_illumination_numerical(Ims, N_orb*len(M_range)/num)
				Ims_bar_fit_total = integrate_illumination_numerical(Ims_bar_fit, N_orb*len(M_range)/num)
				Ims_vec.append(Ims_total)


			if debug:
				pause()
				plt.close('all')

		Is_vec = np.array(Is_vec)
		Im_vec = np.array(Im_vec)
		Ims_vec = np.array(Ims_vec)

		# r_eci_save = np.vstack(r_eci_save)
		# self.r_eci_save = r_eci_save
		# JD_save = np.concatenate(JD_save)
		# self.JD_save = JD_save

		if body == 'sun':
			return Is_vec
		if body == 'moon':
			return Im_vec
		if body == 'moon+sun':
			return Ims_vec #Is_vec, Im_vec, Ims_vec



if __name__ == "__main__":

	from leocat.utils.general import write_pickle

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-alt_vec', nargs='+', required=True)
	parser.add_argument('-inc_vec', nargs='+', required=True)
	parser.add_argument('-JD0_start', required=True)
	parser.add_argument('-num_years', required=True)
	parser.add_argument('-body')
	parser.add_argument('-swath')
	parser.add_argument('-verbose', type=int)
	parser.add_argument('-pid_str')
	parser.add_argument('-OUT_DIR')
	args = parser.parse_args()

	# alt = float(args.alt)
	alt_vec = np.array([float(val) for val in args.alt_vec])
	inc_vec = np.array([float(val) for val in args.inc_vec])
	JD0_start = float(args.JD0_start)
	num_years = float(args.num_years)

	verbose = 0
	if args.verbose:
		verbose = int(args.verbose)

	swath = 0.0
	if args.swath:
		swath = float(args.swath)

	body = 'moon+sun'
	if args.body:
		body = args.body

	pid_str = None
	if args.pid_str:
		pid_str = args.pid_str

	OUT_DIR = None
	if args.OUT_DIR:
		OUT_DIR = args.OUT_DIR


	zz_out = design_illumination(alt_vec, inc_vec, JD0_start, num_years, swath=swath, 
						body=body, verbose=verbose)
	#

	if (OUT_DIR is not None) and (pid_str is not None):
		file = os.path.join(OUT_DIR,f'{pid_str}.pkl')
		write_pickle(file, zz_out)
