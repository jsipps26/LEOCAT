
import matplotlib.pyplot as plt
import numpy as np

from leocat.utils.plot import pro_plot
pro_plot() # for nicer figures

from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT
from leocat.cov import LatitudeCoverage


"""
The total number of valid observations over
a given latitude band is an important metric in 
early conceptual mission design. LatitudeCoverage
is a class that computes the approximate avg. 
number of observations, over mission lifetimes (i.e.,
years), with a few features
	1. Any circular low Earth orbit with J2 precession, 
		incl. inclined orbits
	2. Accounts for swath width
	3. Accounts for solar elevation constraints, and 
		day/night flags

In this example, we first compute the avg. number of 
observations N vs. latitude over a single day for a LEO
satellite with an inclination of 70 deg, and a swath
width of 2500 km. The plot shows a number of features
	1. This simulation is on Jan. 1st 2024, so the Earth
		is tilted away from the Sun. With a solar elevation
		constraint of 20 deg, no valid observations can be
		made above ~equator or so. 
	2. The avg. num. obs. = 0 at the south pole (lat=-90 deg)
		because the satellite is at 70 deg. inclination,
		and even with a 2500 km swath width, it cannot cover
		the south pole.
	3. The "spike" ramps up at ground tracks coalesce as they
		move polewards (towards the south pole).

Secondly, we run a series of LatitudeCoverage runs via
	LC.get_series
over an entire year, then plot the avg. number of observations
for each day, over all latitudes - of course, with the solar 
elevation constraint that observations are only valid when
the solar elevation > 20 deg. A few features are observable:
	1. The simulation begins on Jan. 1st 2024, so the Earth
		is tilted away from the Sun. With the solar elevation
		constraint, from the equator and towards the north pole
		is not observed.
	2. As the year progresses forward, two things occur:
		a) The Earth moves around the Sun, causing the north
			pole or south pole to be consistently illuminated
		b) The orbit itself precesses around the Earth, leaving
			"terminator periods" where the swath envelope is 
			focused on dawn/dusk lighting conditions, leading 
			to few observations that meet the solar elevation
			constraints.
	3. During summer, around day ~150, the Earth is now tilted
		towards the Sun, so the satellite routinely observes 
		northern latitudes.
	4. Anywhere N=0, that is a latitude/day combination for
		which the satellite cannot observe anything.

Process
1. Make the orbit via LEO_RGT function
	or others: LEO, LEO_SSO, LEO_RGT_SSO, etc.
2. Create a LatitudeCoverage instance with
	the given orbit, swath, latitude bounds, 
	and start date JD1
3. Get the number of obs. by latitude, with
	a given elevation constraint, elev=20 deg.
		meaning solar elev > 20 deg. is valid
4. Get num. obs. by latitude, for a whole year
5. Plot results for visualization


"""

# 1. Make the orbit via LEO_RGT function
D, R = 16, 233
JD1 = date_to_jd(2024,1,1) # simulation start date
inc = 70.0
orb = LEO_RGT(D, R, inc)
# orb.plot_orbit() # to visualize orbit
orbit_period = orb.get_period()
Dn = orb.get_nodal_day()

# 2. Create a LatitudeCoverage instance with
#	the given orbit, swath, latitude bounds, 
#	and start date JD1
swath = 2500 # km
lat = np.linspace(-90,90,100)
LC = LatitudeCoverage(orb, swath, lat, JD1)

# 3. Get the number of obs. by latitude, with
#	a given elevation constraint, elev=20 deg.
#		meaning solar elev > 20 deg. is valid
elev = 20.0
num_obs = LC.get_num_obs(elev=elev)
num_obs_true = LC.get_num_obs_true(elev=elev)

# 4. Get num. obs. by latitude, for a whole year
JD2 = JD1 + 365
xx, yy, zz = LC.get_series(JD1, JD2, elev=elev)
xx = xx - JD1 # make time relative to JD1



# 5. Plot results for visualization

"""
Plot avg. number of obs. over latitude, for 1 day.

This plot shows a number of features
1. This simulation is on Jan. 1st 2024, so the Earth
	is tilted away from the Sun. With a solar elevation
	constraint of 20 deg, no valid observations can be
	made above ~equator or so. 
2. The avg. num. obs. = 0 at the south pole (lat=-90 deg)
	because the satellite is at 70 deg. inclination,
	and even with a 2500 km swath width, it cannot cover
	the south pole.
3. The "spike" ramps up at ground tracks coalesce as they
	move polewards (towards the south pole).
"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lat, num_obs_true, label='True')
ax.plot(lat, num_obs, '--', label='Estimate')
ax.legend()
ax.grid()
ax.set_title('Single-day Comparison against Truth')
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('Avg. Number of Observations, N (#)')
fig.show()




"""
Plot avg. number of obs. over latitude, for a year.

This next plot shows a few features:
1. The simulation begins on Jan. 1st 2024, so the Earth
	is tilted away from the Sun. With the solar elevation
	constraint, from the equator and towards the north pole
	is not observed.
2. As the year progresses forward, two things occur:
	a) The Earth moves around the Sun, causing the north
		pole or south pole to be consistently illuminated
	b) The orbit itself precesses around the Earth, leaving
		"terminator periods" where the swath envelope is 
		focused on dawn/dusk lighting conditions, leading 
		to few observations that meet the solar elevation
		constraints.
3. During summer, around day ~150, the Earth is now tilted
	towards the Sun, so the satellite routinely observes 
	northern latitudes.
4. Anywhere N=0, that is a latitude/day combination for
	which the satellite cannot observe anything.

"""
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.pcolormesh(xx, yy, zz)
fig.colorbar(im, ax=ax, label='Avg. Number of Observations, N (#)')
ax.set_xlabel('Days from 2024/1/1')
ax.set_ylabel('Latitude (deg)')
ax.set_title('Avg. Number of Obs. with Solar Elevation > %d deg' % elev + '\n' + \
				'over 1 Year by Latitude')
fig.show()
