import xarray as xr 
import numpy as np 
import pickle

import points_of_interest as poi 
from paths import *

# open file
city='toronto'
fpath = winds_pkl + city + '/20200501'
infile = open(fpath, 'rb')
ds = pickle.load(infile)
infile.close()

# anchor point and average bearing
anchor = poi.cities[city]
avg_bear = 
