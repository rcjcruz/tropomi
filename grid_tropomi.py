#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to grid TROPOMI data into a uniform lat/lon grid spanning from -180 to 180
longitude, -90 to 90 latitude.

Averaged values are found in val_arr_mean array (default shape: (180, 360))
"""

# Preamble
import warnings
import numpy as np
import xarray as xr
from glob import glob
import sys
from open_tropomi import *
from dask.diagnostics import ProgressBar

def aggregate_tropomi(ds, bbox=(-90, -180, 90, 180), res=1):
    # lat/lon max/min
    latmn, lonmn, latmx, lonmx = bbox
    # create a uniform lat/lon grid
    lat_bnds = np.arange(latmn, latmx, res)
    lon_bnds = np.arange(lonmn, lonmx, res)
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

    # Filter lat/lon mn/mx values for each grid square
    vlonmn = np.minimum(ds['longitude_bounds'][0].values,
                        ds['longitude_bounds'][1].values)[filter_arr]
    vlonmx = np.maximum(ds['longitude_bounds'][2].values,
                        ds['longitude_bounds'][3].values)[filter_arr]
    vlatmn = np.minimum(ds['latitude_bounds'][0].values,
                        ds['latitude_bounds'][1].values)[filter_arr]
    vlatmx = np.maximum(ds['latitude_bounds'][2].values,
                        ds['latitude_bounds'][3].values)[filter_arr]

    for i in range(no2.size):
        # Find the indices in the lat/lon_bnds grid at which the
        # max/min lat/lon would fit (i.e. finding the grid squares that the data
        # point has values)
        lat_inds = np.searchsorted(lat_bnds, np.array([vlatmn[i], vlatmx[i]]))
        lon_inds = np.searchsorted(lon_bnds, np.array([vlonmn[i], vlonmx[i]]))

        # Obtain the lat/lon indices that will be used to slice val_arr
        lat_slice = slice(lat_inds[0], lat_inds[1]+1)
        lon_slice = slice(lon_inds[0], lon_inds[1]+1)

        # Add the NO2 values that fit in those lat/lon grid squares to val_arr and
        # add 1 to dens_arr to increase the count of observations found in that
        # grid square
        val_arr[lat_slice, lon_slice] += no2[i]
        dens_arr[lat_slice, lon_slice] += 1

    # Set negative values in val_arr to 0
    val_arr = val_arr.clip(min=0)

    # Divide val_arr by dens_arr; if dividing by 0, return 0 in that entry
    val_arr_mean = np.divide(val_arr, dens_arr, out=(
        np.zeros_like(val_arr)), where=(dens_arr != 0))

    # Load date of orbit; will need to update when I work with multiple days
    date = str(ds.time.data)
    date = np.array([date[:date.find('T')]])

    # Create a new DataArray where each averaged value corresponds to
    # lat/lon values that will be plotted on a PlateCarree projection
    new_ds = xr.DataArray(np.array([val_arr_mean]),
                          dims=('time', 'latitude', 'longitude'),
                          coords={'time': date,
                                  'latitude': lat_bnds,
                                  'longitude': lon_bnds})
    return new_ds

#############################

if __name__ == '__main__':
    # f='/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'
    f = '*_20200505*_*.nc'
    ds=dsread(f)
    ds=aggregate_tropomi(ds)
    print(ds)
