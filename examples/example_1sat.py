
import matplotlib.pyplot as plt
import numpy as np

from leocat.utils.plot import pro_plot
pro_plot() # for better figures

from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT_SSO
from leocat.cov import get_coverage, get_num_obs, get_revisit


"""
Example demonstrating coverage computation for
1 SSO satellite with a repeat ground track of 16 nodal
days and 233 revolutions (Landsat-8), albeit with
a 1600 km swath width.

Process
1. Make the orbit via LEO_RGT_SSO function
2. Set the simulation period, swath size, and spatial
	grid resolution
3. Compute coverage via get_coverage
4. Compute statistics such as the total number of observations,
	and the maximum revisit interval at each grid point
5. Plot results for visualization


"""

# 1. Make the orbit via LEO_RGT_SSO function
D, R = 16, 233
JD1 = date_to_jd(2024,1,1) # simulation start date
MLST = 10.0
orb = LEO_RGT_SSO(D, R, MLST, JD1, direction='descending')
# orb.plot_orbit() # to visualize orbit


# 2. Set the simulation period, swath size, and spatial
# 	grid resolution
orbit_period = orb.get_period()
swath = 1600 # km
JD2 = JD1 + 2*orbit_period/86400 # 2 orbital revolutions total
res = 100 # 100x100 km^2 grid cell


# 3. Compute coverage via get_coverage
lon, lat, t_access = get_coverage(orb, swath, JD1, JD2, res=res)


# 4. Compute statistics such as the total number of observations,
# 	and the maximum revisit interval at each grid point
num_obs = get_num_obs(t_access, len(lon)) # number of observations/counts
dt_max = get_revisit(t_access, len(lon), 'max') / 86400 * 24 # max revisit in hours


# 5. Plot results for visualization
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