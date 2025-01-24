
import matplotlib.pyplot as plt
import numpy as np

from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT_SSO
from leocat.cov import get_coverage, get_num_obs, get_revisit


D, R = 16, 233
JD1 = date_to_jd(2024,1,1) # simulation start date
MLST = 10.0
orb = LEO_RGT_SSO(D, R, MLST, JD1, direction='descending')

orbit_period = orb.get_period()
swath = 1600 # km

# JD2 = JD1 + 16.0 # units of days
JD2 = JD1 + 2*orbit_period/86400 # units of days

# lon, lat = np.array([-43]), np.array([-66]) # single lon/lat point
res = 100 # km, resolution of grid (if global)

# if you want global
lon, lat, t_access = get_coverage(orb, swath, JD1, JD2, res=res)

# if you want a single point
# lon, lat, t_access = get_coverage(orb, swath, JD1, JD2, lon=lon, lat=lat)

keys = np.array(list(t_access.keys()))

num_obs = get_num_obs(t_access, len(lon)) # number of observations/counts
dt_max = get_revisit(t_access, len(lon), 'max') / 86400 * 24 # max revisit in hours


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