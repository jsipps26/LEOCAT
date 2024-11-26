
import numpy as np
from leocat.utils.orbit import Orbit, MLST_to_LAN
from leocat.src.rgt import RepeatGroundTrack



def LEO(alt, inc, e=0.0, LAN=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	a = R_earth + alt
	inc = np.radians(inc)
	LAN = np.radians(LAN)
	omega = np.radians(omega)
	nu = np.radians(nu)
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator,warn=warn)
	return orb

def LEO_SSO(alt, MLST, JD, direction='ascending', e=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	"""
	Technically, the arg of latitude must be zero if direction is 'ascending',
	and u should be 180 deg. if direction is 'descending'. However, the change
	in LAN across a revolution is <0.05 deg., which does not significantly 
	change the true equatorial crossing time.

	If you wanted to compensate, you must find TOF to the ascending/descending
	node given u = omega + nu, then subtract/add TOF to JD1, which specifies
	the LAN at which the satellite crosses the ascending/descending node.

	"""
	if not (propagator == 'SPE' or propagator == 'SPE+frozen'):
		raise Exception('SSO requires J2 perturbation, propagator must be one of: [SPE, SPE+frozen]')
	mu = MU
	J2 = 0.00108248
	a = R_earth + alt
	arg1 = -2*a**(7/2) * LAN_dot_SSO * (1-e**2)**2
	arg2 = 3*R_earth**2 * J2 * np.sqrt(mu)
	inc = np.arccos(arg1/arg2)

	if direction == 'ascending':
		LTAN = MLST % 24
		LTDN = (MLST + 12) % 24
	elif direction == 'descending':
		LTAN = (MLST + 12) % 24
		LTDN = MLST % 24

	if direction == 'descending':
		MLST = (MLST + 12) % 24
	LAN = MLST_to_LAN(MLST, JD)

	omega = np.radians(omega)
	nu = np.radians(nu)
	metadata = {'LTAN': LTAN, 'LTDN': LTDN, 'direction': direction, 'JD': JD}
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator, metadata=metadata, warn=warn)
	return orb


def LEO_RGT(D, R, inc, e=0.0, LAN=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	RGT = RepeatGroundTrack(D,R,propagator=propagator)
	inc = np.radians(inc)
	a = RGT.get_a(inc, e=e)
	LAN = np.radians(LAN)
	omega = np.radians(omega)
	nu = np.radians(nu)
	metadata = {'D': D, 'R': R}
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator, metadata=metadata, warn=warn)
	return orb

def LEO_RGT_SSO(D, R, MLST, JD, direction='ascending', e=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	if not (propagator == 'SPE' or propagator == 'SPE+frozen'):
		raise Exception('SSO requires J2 perturbation, propagator must be one of: [SPE, SPE+frozen]')
	RGT = RepeatGroundTrack(D,R,propagator=propagator)
	a, inc = RGT.get_sso(e=e)
	LAN = RGT.get_sso_LAN(MLST,JD,e=e,direction=direction)

	if direction == 'ascending':
		LTAN = MLST
		LTDN = (MLST + 12) % 24
	elif direction == 'descending':
		LTAN = (MLST + 12) % 24
		LTDN = MLST

	omega = np.radians(omega)
	nu = np.radians(nu)
	metadata = {'D': D, 'R': R, 'LTAN': LTAN, 'LTDN': LTDN, 'direction': direction, 'JD': JD}
	orb = Orbit(a,e,inc,LAN,omega,nu,propagator=propagator, metadata=metadata, warn=warn)
	return orb

def LEO_SSO_RGT(D, R, MLST, JD, direction='ascending', e=0.0, omega=270.0, nu=90.0, propagator='SPE+frozen', warn=True):
	return LEO_RGT_SSO(D, R, MLST, JD, direction=direction, e=e, omega=omega, nu=nu, propagator=propagator, warn=warn)

