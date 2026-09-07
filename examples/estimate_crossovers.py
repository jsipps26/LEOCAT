
import matplotlib.pyplot as plt
import numpy as np

from leocat.utils.plot import *
pro_plot()

from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT_SSO
from leocat.src.crossovers import CrossoverEstimator

"""
Example demonstrating how to estimate ground track
crossover locations, along with time of first and 
second overpass at a given point.

Crossover estimation is non-linear even for circular,
keplerian orbits, since the Earth rotates. The 
CrossoverEstimator implements a non-linear least-squares
solution to finding the crossover location for every
orbit, with every other orbit. In general, it is O(n^2)
for n orbital revolutions. There are likely improvements
since many latitudes are equivalent.

There is no swath size because CrossoverEstimator is focused
on finding only the locations and times at which the ground 
track crosses itself.

Process
1. Make the orbit via LEO_RGT_SSO function
	or others: LEO, LEO_SSO, LEO_RGT, etc.
2. Initialize and run crossover estimator
3. Plot results for visualization

"""


# 1. Make a "template" orbit that specifies altitude,
# inclination, equatorial crossing (for SSO), etc.
D = 16 # Number of nodal days in repeat cycle
R = 233 # Number of revs in repeat cycle
JD1 = date_to_jd(2024,1,1) # simulation start date
MLST = 10.0 # equatorial crossing of 10am
orb = LEO_RGT_SSO(D, R, MLST, JD1, direction='descending')

# 2. Initialize and run crossover estimator
num_tracks = R # check all tracks over the repeat cycle (R)
CE = CrossoverEstimator(orb, num_tracks, JD1)
crosstracks = CE.find_initial_conditions() # initialize
cross_data = CE.find_crossovers(crosstracks) # determine all crossovers
lon, lat, t1, t2 = CE.get_access(cross_data) # format into pos/time information

dt_vec = (t2-t1) / 86400 # time to crossover, solar days

# 3. Plot results for visualization
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.scatter(lon, lat, marker='o', s=2, c=dt_vec)
fig.colorbar(im, ax=ax, label='Crossover Time (days)')
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Latitude (deg)')
ax.set_title('Time to Cross Ground Track')
fig.show()



