# Low Earth Orbit Coverage Analysis Tool (LEOCAT) v0.1

This python package is made to efficiently compute satellite regional access (the portion of Earth observed) for conical and rectangular fields of view (FOVs), utilizing efficient algorithms, as well as numpy vectorization and jit compilation via numba. 

Access calculations are split into two major categories: time-series and grid-point series (or GP series). In time-series approaches, the satellite is propagated forward one step at a time, and access is computed. In GP series approaches, the time at which the satellite observes a point on Earth (a GP) is input, and the output is a time-series. The former is for accurate FOV area coverage over short spans (e.g., a day of simulation) while the latter is for coverage over months or years, which reveals satellite observation capability in the long term.
<br>

Example: Determine average time between observations over Ukraine:<br>
Simulation details: <br>
Two satellites on 16-day repeat ground track, sun-synchronous equatorial crossing at 10am (descending) <br>
Sat1 and Sat2 are phased 180 deg. apart, with a swath size of 345 km (FOV of 27 deg.)<br>
![ukraine_clouds](https://github.com/user-attachments/assets/b29f9773-a7de-431f-b528-b7c44e3874c9)

**Install**<br> 
Tested on python 3.7+<br>
Make a new environment with python version 3.7, either through conda or your virtual environment manager of choice<br> 
$ conda create --name leocat_test python=3.7<br>
$ conda activate leocat_test<br>

Navigate to install location<br> 
Installing through a Jupyter notebook/hub has encountered issues with dependencies (see below).<br>
$ git clone https://github.com/jsipps26/LEOCAT.git<br>
$ cd LEOCAT-main<br>
$ pip install .

**Dependencies**<br> 
dill>=0.3.5.1<br>
matplotlib>=3.5.2<br>
numba>=0.53.1<br>
numpy>=1.21.6<br>
pandas>=1.3.5<br>
pyproj>=3.2.1<br>
scipy>=1.7.3<br>
tqdm>=4.62.3<br>

**External Packages/Data Included**<br>
fqs (https://github.com/NKrvavica/fqs) for quartic computation<br> 
jd_to_date and date_to_jd from https://gist.github.com/jiffyclub/1294443<br>

Please email any errors encountered to john.sipps@utexas.edu<br>

Developed by Jonathan Sipps<br> 
Graduate Research Fellow<br> 
The University of Texas at Austin<br> 
john.sipps@utexas.edu<br> 

This work is supported by a NASA Space Technology Graduate Research Opportunity (NSTGRO) fellowship.<br> 
