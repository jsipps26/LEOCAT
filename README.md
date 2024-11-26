# Low Earth Orbit Coverage Analysis Tool (LEOCAT) v0.1

WORK-IN-PROGRESS RESEARCH PROJECT<br>

**Install**<br> 
Only tested on python 3.7+<br>
Make a new environment with python version 3.7, either through conda or your virtual environment manager of choice<br> 
$ conda create --name leocat_test python=3.7<br>
$ conda activate leocat_test<br>

Navigate to install location<br> 
Installing through a Jupyter notebook/hub has encountered issues with dependencies (see below).<br>
$ git clone https://github.com/jsipps26/LEOCAT.git<br>
$ cd TEMPO-main<br>
$ python install.py<br> OR $ pip install .

**Demo test**<br> 
Once in the TEMPO directory, try the following: <br> 
$ python tempo/examples/LS8_example.py<br>
$ python tempo/examples/IS2_example.py<br>


**Dependencies**<br> 
dill>=0.3.5.1<br>
matplotlib>=3.5.2<br>
numba>=0.53.1<br>
numpy>=1.21.6<br>
pandas>=1.3.5<br>
pyproj>=3.2.1<br>
numpy-quaternion>=2021.6.9.13.34.11<br>
scipy>=1.7.3<br>
pyshp>=2.3.0<br>
tqdm>=4.62.3<br>
requests>=2.28.1<br>

If it cannot find numpy-quaternion or numba versions, please install manually via<br>
$ pip install numba<br>
$ pip install numpy-quaternion<br>

**External Packages/Data Included**<br>
fqs (https://github.com/NKrvavica/fqs) for quartic computation<br> 
jd_to_date and date_to_jd from https://gist.github.com/jiffyclub/1294443<br>
ne_10m_coastline dataset (https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/) for plotting<br> 
ECI to ECEF rotation for years 2018-2028<br>

Please email any errors encountered to john.sipps@utexas.edu<br>

Developed by Jonathan Sipps<br> 
Graduate Research Fellow<br> 
The University of Texas at Austin<br> 
john.sipps@utexas.edu<br> 

This work is supported by a NASA Space Technology Graduate Research Opportunity (NSTGRO) fellowship.<br> 