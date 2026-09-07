
import matplotlib.pyplot as plt
import numpy as np

from leocat.utils.general import rnp, pause
from leocat.utils.plot import pro_plot, make_fig, plot_sim
pro_plot()
rnp()

from leocat.utils.const import R_earth
from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT_SSO
from leocat.cov import Satellite, Instrument

from leocat.utils.math import unit
from leocat.utils.geodesy import lla_to_ecf, cart_to_RADEC

"""
Work-in-progress (WIP) suite

This example demonstrates the usage of Satellite and
Instrument objects to determine instantaneous coverage,
of a satellite that is rolling +/-10 deg. off-nadir in
the cross-track (CT) direction, with a large rectangular
field-of-view (FOV). Satellite/Instrument classes constrast
the rest of LEOCAT, which is dedicated to estimating
regional coverage over days, months, and years.

The difference in instantaneous coverage is that satellites
or their sensors may be generally agile, looking off-nadir
towards targets-of-interest as needed. Most other utilities
in LEOCAT assume nadir pointing geometry, but this suite
allows for more general attitude control.

The Instrument class uses a novel approach to grid point (GP)
access in that we don't iterate over all grid cells/points
on the Earth's surface to determine which are within the
field-of-view. Instead, we project a boundary mesh of the 
FOV onto Earth's surface, hash which grid cells (GCs) the
boundary points intersect, then "bridge across" the footprint.
The consequence is that only GCs at the footprint boundary
are queried, so while the output is areal, the computation is
nearly linear with respect to boundary perimeter. Conversely,
if the whole globe is queried, computation grows quadratically
with respect to resolution. In this technique, we've reduced
the footprint query complexity from quadratic over the globe to
roughly linear over the footprint boundary.

The Instrument class has been made robust to handle most edge 
cases, including over-the-horizon coverage, polar coverage, 
and variations in footprint dimension due to the WGS84 ellipsoid.

Example process
1. Make the orbit via LEO_RGT_SSO function
	or others: LEO, LEO_SSO, LEO_RGT, etc.
2. Set the time series, t
	Satellite/Instrument access is queried at
	instants of time, not over simulation periods like
	in other LEOCAT routines
3. Set the spatial resolution
4. Set Instrument parameters such as field-of-view (FOV)
	for the cross-track (CT) direction, or FOV_CT
5. Propagate the Satellite forward in time, get the 
	ground track, and generate the roll angle series
6. Plot results for visualization
	Note that the rectangular footprint is highly distorted
	as both the curvature of the Earth and off-nadir pointing
	limit the boundaries of the footprint. Over-the-horizon
	coverage is visible when the footprint dips past a critical
	roll angle, and the familiar rectangular corners turn into
	circular arcs due to the limitations of the horizon visible
	by the Instrument field-of-view.


"""

def lla_to_poly_gcs(lon, lat):
	# Turn DGG lons/lats into grid cell polygons in ECEF
	poly_grid = DGG.get_poly_grid(lon, lat)
	num_gc = len(poly_grid)
	x = poly_grid[:,:,0].flatten()
	y = poly_grid[:,:,1].flatten()
	z = lla_to_ecf(x, y, np.zeros(x.shape))
	poly_grid_ecf = z.reshape((num_gc,5,3))
	return poly_grid_ecf


# 1. Make the orbit via LEO_RGT_SSO function
D = 16 # Number of nodal days in repeat cycle
R = 233 # Number of revs in repeat cycle
JD1 = date_to_jd(2024,1,1) # simulation start date
MLST = 10.0 # equatorial crossing of 10am
orb = LEO_RGT_SSO(D, R, MLST, JD1, direction='descending')


# 2. Set the time series, t
orbit_period = orb.get_period()
t = np.linspace(0,orbit_period,100)


# 3. Set the spatial resolution
res = 200.0


# 4. Set Instrument parameters such as field-of-view (FOV)
FOV_CT = 110.0 # cross-track, deg
FOV_AT = 110.0 # along-track, deg
Inst = Instrument(FOV_CT, FOV_AT) # rectangular FOV


# 5. Propagate the Satellite forward in time, get the 
#	ground track, and generate the roll angle series
Sat = Satellite(orb, res, JD1, Inst)
r_ecf, v_ecf = Sat.get_rv(t)
r_gt = unit(r_ecf) * R_earth
f = 2*np.pi/orbit_period # roll frequency
roll_series = np.sin(2*np.pi*f*t) * 10 # +/-10 deg, oscillatory


# 6. Plot results for visualization
DGG = Sat.space_params['DGG']
lon, lat = DGG.get_lonlat()
poly_grid_ecf = lla_to_poly_gcs(lon, lat)

zoom = 1.5
for k in range(len(t)):
	# Set roll angle, compute covered grid points (GPs)
	R_roll = Sat.get_att(t[k],roll_series[k])
	r_GP = Sat.get_access(r_ecf[k], R_roll) # covered GPs
	r_GP0 = np.mean(r_GP,axis=0)
	RA, DEC = cart_to_RADEC(r_GP0) # for plotting camera
	RA = RA + 20
	DEC = DEC + 10

	# Query internal footprint boundary
	r_mesh = Sat.Inst.FieldOfView.r_mesh
	r_edge = Sat.Inst.FieldOfView.r_edge
	r_boundary = r_mesh
	if not (r_edge is None):
		r_boundary = np.vstack((r_mesh,r_edge))

	# Use LEOCAT's plot_sim to make 3D Earth at low level/matplotlib API
	fig, ax = make_fig('3d')
	lines = []
	lines.append( ax.plot(r_ecf.T[0], r_ecf.T[1], r_ecf.T[2], zorder=99) )
	lines.append( ax.plot(r_gt.T[0], r_gt.T[1], r_gt.T[2], zorder=98) )
	lines.append( ax.plot(r_boundary.T[0], r_boundary.T[1], r_boundary.T[2], '.', zorder=97) )
	fig, ax = plot_sim(lines, RA, DEC, zoom, target=r_GP0, alpha_edge=1.0, figsize=None)
	ax.plot(np.nan, np.nan, c='C0', label='Orbit (ECEF)')
	ax.plot(np.nan, np.nan, c='C1', label='Ground Track')
	ax.plot(np.nan, np.nan, '.', c='C2', label='Footprint boundary')
	ax.plot(r_GP.T[0], r_GP.T[1], r_GP.T[2], '.', c=[0,0.9,0], zorder=0, label='Covered GPs')
	ax.legend(loc='lower left')
	fig.show()

	# break
	pause()
	plt.close('all')
