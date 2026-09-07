
import matplotlib.pyplot as plt
import numpy as np

from leocat.utils.plot import pro_plot
pro_plot()

from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT_SSO
from leocat.cov import get_num_obs, get_revisit, combine_coverage

from leocat.cst import ConstellationShell, WalkerDelta
from leocat.utils.geodesy import DiscreteGlobalGrid


"""
Example demonstrating how to compute coverage for
a Walker Delta constellation of 6 satellites.

In LEOCAT, instead of making 6 orbits then propagating
each, there is a ConstellationShell class that first
computes the coverage for one satellite, then copies and
shifts coverage accordingly based on changes in the
longitude of the ascending node (LAN) and true anomaly
(nu). This way, coverage is only computed once for the
entire constellation, and shifted over in space and time
such that it is approximately equal to the coverage 
obtained by directly computing coverage for every satellite.

The approximation is valid if every satellite orbit is
circular and all are at the same altitude. In practice, 
constellations are only realizable when all satellites 
remain at a specific set of orbital elements such that
precession is constant between each - otherwise phasing 
will diverge and change over time.

Process
1. Make a "template" orbit that specifies altitude,
inclination, equatorial crossing (for SSO), etc.
	or others: LEO, LEO_SSO, LEO_RGT, etc.
2. Find the shifts that other satellites will require
to make a Walker Delta constellation
	Shifts in LAN and nu
3. Set the simulation period and swath size
4. Set the spatial grid via DiscreteGlobalGrid
5. Create the ConstellationShell and compute access
	for each satellite
6. Combine all access events, tally total number of
	observations and compute maximum revisit interval
7. Plot results for visualization


"""


# 1. Make a "template" orbit that specifies altitude,
# inclination, equatorial crossing (for SSO), etc.
D = 16 # Number of nodal days in repeat cycle
R = 233 # Number of revs in repeat cycle
JD1 = date_to_jd(2024,1,1) # simulation start date
MLST = 10.0 # equatorial crossing of 10am
orb = LEO_RGT_SSO(D, R, MLST, JD1, direction='descending')


# 2. Find the shifts that other satellites will require
# to make a Walker Delta constellation
# 	Shifts in LAN and nu
P = 3 # number of planes
F = 2 # number of satellites per plane
LAN_shifts, nu_shifts = WalkerDelta(P, F)


# 3. Set the simulation period and swath size
orbit_period = orb.get_period() # defaults to nodal period
Dn = orb.get_nodal_day()
JD2 = JD1 + 3*orbit_period/86400 # units of solar days
swath = 500 # km


# 4. Set the spatial grid via DiscreteGlobalGrid
res = 100  # 100x100 km^2 grid cells, global
DGG = DiscreteGlobalGrid(A=res**2)
lon, lat = DGG.get_lonlat() # centers of grid cells


# 5. Create the ConstellationShell and compute access
# 	for each satellite
CST_instance = ConstellationShell(orb, swath, JD1, JD2, 
					LAN_shifts=LAN_shifts, nu_shifts=nu_shifts)
#
CST = CST_instance.get_access(lon=lon, lat=lat, res=res, verbose=1, approx=True, fix_noise=True)


# 6. Combine all access events, tally total number of
# 	observations and compute maximum revisit interval
lons, lats = [], []
t_access_list = []
for key in CST:
	lons.append(CST[key]['lon'])
	lats.append(CST[key]['lat'])
	t_access_list.append(CST[key]['t_access'])
lon, lat, t_access_cst = combine_coverage(lons, lats, t_access_list, DGG)

num_obs = get_num_obs(t_access_cst, len(lon)) # number of observations/counts
dt_max = get_revisit(t_access_cst, len(lon), 'max') / 86400 * 24 # max revisit in hours


# 7. Plot results for visualization
b = num_obs > 0
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.scatter(lon[b], lat[b], marker='o', s=2, c=num_obs[b])
fig.colorbar(im, ax=ax, label='N (counts)')
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Latitude (deg)')
ax.set_title('Number of Observations')
fig.show()

# b = num_obs > 0
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.scatter(lon, lat, marker='o', s=2, c=dt_max)
fig.colorbar(im, ax=ax, label='Revisit Max (hrs)')
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Latitude (deg)')
ax.set_title('Revisit Max')
fig.show()

