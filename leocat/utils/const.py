
# import numpy as np
from math import pi
import os, sys

TWO_PI = 2*np.pi
LON_ORIGIN, LAT_ORIGIN = -180, -90

# EPSG_LLA = 4326
OBLIQUITY = 23.43648 # deg
FLATTENING = 1/298.257223563

R_earth = 6378137/1e3 # 6378.1363, km, equatorial radius
R_earth_pole = R_earth * (1-FLATTENING) # km, polar radius

MU = 3.986004418e14 / (1000**3) # 398600.4415, km^3/s^2
W_EARTH = 7.292115146706979e-05 # rad/s
J2 = 0.00108248 # 0.0010826267
LAN_dot_SSO = W_EARTH - 2*pi/86400

degree_str = r'$\degree$'

PROJ4_LLA = '+proj=longlat +datum=WGS84 +no_defs +type=crs' # +units=km'
PROJ4_ECF = '+proj=geocent +datum=WGS84 +units=km +no_defs'
PROJ4_SIN_MD = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=km +no_defs' # modis sinusoidal
PROJ4_SIN = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs'
PROJ4_CEA = '+proj=cea +lon_0=0 +x_0=0 +y_0=0 +lat_ts=45 +ellps=WGS84 +datum=WGS84 +units=km +no_defs'
PROJ4_LAEA_N = '+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs'
PROJ4_LAEA_S = '+proj=laea +lon_0=0 +lat_0=-90 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=km +no_defs'
LAEA_LAT = 60.0 # defines bounds for laea

MATH_SYMBOLS = ['+','-','*','/','**']

LEOCAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MAIN_DIR = os.path.join(LEOCAT_DIR, 'leocat')
UTILS_DIR = os.path.join(MAIN_DIR,'utils')


