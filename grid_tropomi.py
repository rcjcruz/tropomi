#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to grid TROPOMI data into a uniform lat/lon grid spanning from -180 to 180
longitude, -90 to 90 latitude. 

Averaged values are found in val_arr_mean array (shape: (180, 360))
"""


# Preamble
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import datetime as dt
from collections import namedtuple
import sys

import open_tropomi as ot

f = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'
ds = ot.dsread(f)

# lat/lon max/min
latmn, lonmn, latmx, lonmx = (-90, -180, 90, 180)
# create a uniform lat/lon grid
lat_bnds = np.arange(latmn, latmx, 1)
lon_bnds = np.arange(lonmn, lonmx, 1)
# val_arr will accumulate the values within each grid entry
val_arr = np.zeros([lat_bnds.size, lon_bnds.size])
# densarr will count the number of observations that occur within that grid entry
dens_arr = np.zeros([lat_bnds.size, lon_bnds.size], dtype=np.int32)

# Get an array of no2, lat, lon values
no2 = ds['nitrogendioxide_tropospheric_column'].values
lat = ds.latitude.values
lon = ds.longitude.values

# Check if the lat and lon values are found within the lat/lon mn/mx
lat_flt = (lat > latmn) * (lat < latmx)
lon_flt = (lon > lonmn) * (lon < lonmx)

# Create a filter array with only the lat/lon values in
filter_arr = lat_flt * lon_flt

# Keep no2 values that are within the bounded lat/lon
no2 = no2[filter_arr]

# Locate corners of observations
# Longitude
lon0 = ds['longitude_bounds'][0].values
lon1 = ds['longitude_bounds'][1].values
lon2 = ds['longitude_bounds'][2].values
lon3 = ds['longitude_bounds'][3].values
# Latitude
lat0 = ds['latitude_bounds'][0].values
lat1 = ds['latitude_bounds'][1].values
lat2 = ds['latitude_bounds'][2].values
lat3 = ds['latitude_bounds'][3].values

# Filter lat/lon mn/mx values for each grid square
vlonmn = np.minimum(lon0, lon1)[filter_arr]
vlonmx = np.maximum(lon2, lon3)[filter_arr]
vlatmn = np.minimum(lat0, lat1)[filter_arr]
vlatmx = np.maximum(lat2, lat3)[filter_arr]

for k in range(no2.size):
    # Find the indices in the lat/lon_bnds grid at which the
    # max/min lat/lon would fit (i.e. finding the grid squares that the data
    # point has values)
    lat_inds = np.searchsorted(lat_bnds, np.array(
        [vlatmn[k], vlatmx[k]]))  # look at documentation
    lon_inds = np.searchsorted(lon_bnds, np.array([vlonmn[k], vlonmx[k]]))

    # Obtain the lat/lon indices that will be used to slice val_arr
    lat_slice = slice(lat_inds[0], lat_inds[1]+1)
    lon_slice = slice(lon_inds[0], lon_inds[1]+1)

    # Add the NO2 values that fit in those lat/lon grid squares to val_arr and
    # add 1 to dens_arr to increase the count of observations found in that
    # grid square
    val_arr[lat_slice, lon_slice] += no2[k]
    dens_arr[lat_slice, lon_slice] += 1

# Set negative values in val_arr to 0
val_arr = val_arr.clip(min=0)

# Divide val_arr by dens_arr; if dividing by 0, return 0 in that entry
val_arr_mean = np.divide(val_arr, dens_arr, out=(
    np.zeros_like(val_arr)), where=(dens_arr != 0))

#val_arr_mean.shape = (180, 360)

if __name__ == '__main__':
    # print(val_arr[-20])
    # print(dens_arr[-20])
    # print(val_arr_mean[-20])
    print(val_arr_mean.shape)