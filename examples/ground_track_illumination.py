
import matplotlib.pyplot as plt
import numpy as np
import os, sys

from leocat.utils.general import rnp, pause
from leocat.utils.plot import pro_plot, make_fig, plot_sim
pro_plot()
rnp()

from leocat.utils.const import R_earth
from leocat.utils.time import date_to_jd
from leocat.orb import LEO_RGT_SSO, LEO_MSSO
from leocat.cov import Satellite, Instrument

from leocat.utils.orbit import MLST_to_LAN, beta_analytic
from leocat.utils.math import unit
from leocat.utils.geodesy import lla_to_ecf, cart_to_RADEC

from leocat.src.illumination import GroundTrackIllumination

"""
This example shows how to compute the total illumination 
of sunlight normal to the swath envelope per day, for
1 year. The novelty is that we fit illumination
functions from Fund. of Astrodynamics (Vallado) with
polynomials, then evaluate the total illumination by 
analytic integration. The result is that we can query
the illumination of the swath envelope over several
years nearly instantly.

While the beta angle does correlate with whether the
swath envelope is illuminated by sunlight, it is not
as comprehensive as is required to know which periods
of the mission lifetime contain effectively unusable
illumination performance. By directly summing the total
illumination in lux on the swath envelope, we can find
when the satellite will be able to observe much of the
Earth during the day.

Of course, sun-synchronous orbits (SSOs) have consistent
daily illumination, but with GroundTrackIllumination, we
can query non-SSO illumination profiles for small or 
large swath envelopes.


"""


JD1 = date_to_jd(2024,1,1) # simulation start date
alt = 705
num_cycles = -3
MLST = 10.0
LAN0 = np.degrees(MLST_to_LAN(MLST,JD1))
orb = LEO_MSSO(alt, num_cycles, LAN=LAN0)

I_GT = GroundTrackIllumination(body='sun')

num_years = 1.0
JD2_des = JD1 + num_years*365
dJD_des = 1.0
swath = 2000
I_GT.set_ephemeris(orb, JD1, JD2_des, dJD_des)
I_est = I_GT.predict(w=swath)
# I_true = I_GT.predict_true(w=swath)
days = I_GT.t_bar/86400
LAN_bar = I_GT.LAN_bar
JD_bar = I_GT.JD_bar
beta = beta_analytic(LAN_bar, orb.inc, JD_bar, body='sun')


z = I_est / np.max(I_est)

fig, ax = make_fig()
ax.plot(days, z*100, c='C0', label='Illumination')
ax.plot(np.nan, np.nan, c='C1', label='Beta Angle')
ax.set_ylabel('Total Illumination rel. to Maximum (%)')
ax.set_xlabel('Days since Epoch (1/1/24)')
ax.legend()
ax2 = ax.twinx()
ax2.plot(days, np.abs(beta), c='C1')
ax2.set_ylabel('Abs(Beta Angle) (deg)')
ax2.set_ylim([-5,95])
title = 'Daily Swath Illumination over 1 Year'
ax.set_title(title)
fig.show()


