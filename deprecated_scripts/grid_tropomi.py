#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: grid_tropomi.py

Script contains functions:
    - aggregate_tropomi(ds, week_num, res=0.05, plot_type='toronto')
    
Script to grid TROPOMI data into a uniform lat/lon grid spanning from -180 to 180
longitude, -90 to 90 latitude.

Averaged values are found in val_arr_mean array
"""

# Preamble
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import sys
import open_tropomi as ot
import points_of_interest as poi

#############################


def aggregate_tropomi(ds, week_num, res=0.05, plot_type='toronto'):
    """
    Return a xr.DataArray with averaged NO2 product aggregated over a uniform
    lat/lon grid with bounds defined by bbox and resolution res.

    Args:
        ds (xr.Dataset): TROPOMI NO2 dataset.
        week_num (int): calendar week; 1 <= week_num <= 52
        res (float): resolution of spacing. Default: 0.05 ~ 6km.
        plot_type (str): specification of plot type. Must be 'world'
            or 'toronto'
    Returns:
        new_ds (xr.Dataset): TROPOMI NO2 dataset aggregated into a 
            uniform lat/lon grid.
    """

    # Define boundaries depending on if plot_type is 'world' or 'toronto'
    if plot_type == 'world':
        bbox = (-180, 180, -90, 90)
    elif plot_type == 'toronto':
        bbox = poi.plot_limits
    else:
        raise ValueError('plot_type must be \'toronto\' or \'world\'')

    # lat/lon max/min
    lonmn, lonmx, latmn, latmx = bbox
    # create a uniform lat/lon grid
    lat_bnds = np.arange(latmn, latmx, res)
    lon_bnds = np.arange(lonmn, lonmx, res)

    # val_arr will accumulate the values within each grid entry
    val_arr = np.zeros([lat_bnds.size, lon_bnds.size])
    # densarr will count the number of observations that occur within that grid entry
    dens_arr = np.zeros([lat_bnds.size, lon_bnds.size], dtype=np.int32)

    # Load an array of no2, lat, lon values
    no2 = ds['nitrogendioxide_tropospheric_column'].values
    lat = ds.latitude.values
    lon = ds.longitude.values

    # Check if the lat and lon values are found within lat/lon bounds
    lat_flt = (lat > latmn) * (lat < latmx)
    lon_flt = (lon > lonmn) * (lon < lonmx)

    # Create array to filter data points found within lat/lon bounds
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
        # point has values for)
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
    date = pd.to_datetime(ds.time.data).floor('D')

    # Create a new DataArray where each averaged value corresponds to
    # lat/lon values that will be plotted on a PlateCarree projection
    new_ds = xr.DataArray(np.array([val_arr_mean]),
                          dims=('time', 'latitude', 'longitude'),
                          coords={'time': np.array([date]),
                                  'latitude': lat_bnds,
                                  'longitude': lon_bnds})

    return new_ds

#############################


def check_valid(ds, week_num_range):
    """
    Check if dataset ds has data for the valid range of week numbers defined by
    week_num_range.

    ds (xr.Dataset): xr.Dataset
    week_num_range (list of int): [week_num_start, week_num_end]
        week_num start and week_num_end must be odd and even, 
        respectively. 

    Returns:
        Boolean if week number of ds is in the week_num_range.
    """

    week_one, week_two = week_num_range
    if (week_one % 2 != 1) or (week_two % 2 != 0) or (week_two - week_one != 1):
        return ValueError('First entry of week_num_range must be an odd int, \
            second entry must be an even int, and the difference between \
                first and second entry must be 1.')

    ds_week = pd.to_datetime(ds.time.data).week

    if (week_one <= ds_week) and (ds_week <= week_two):
        return True
    else:
        return False


if __name__ == '__main__':
    # f='/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'
    f = '*__20200505*_*.nc'
    g = '*__20200504*_*.nc'
    ds1 = ot.dsread(f)
    ds2 = ot.dsread(g)
    # ds1 = aggregate_tropomi(ot.dsread(f))
    # ds2 = aggregate_tropomi(ot.dsread(g))
