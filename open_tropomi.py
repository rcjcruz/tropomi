#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to open multiple netCDF files for TROPOMI NO2 product. 

The resulting dataset has values for nitrogendioxide_tropospheric_column,
qa_value, longitude_bounds, latitude_bounds. The values are sorted according
to time_utc of recorded observation.
"""

import xarray as xr
import pandas as pd
import os
import calendar
import datetime
from datetime import timedelta

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#############################

def dsmod(ds):
    """ 
    Return a modified ds with the time dimension removed, the index 
    changed to time_utc, and scanline and ground pixel values stacked.
    
    ds: DataSet
    """
    
    # remove size-1 dimension time
    ds = ds.squeeze('time')
    # changes the index to time_utc
    ds = ds.set_index(scanline='time_utc')
    
    return ds

#############################

def dsread(f):
    """
    Read netCDF4 files and access PRODUCT and GEOLOCATIONS folders to 
    extract NO2, qa_value, lat/lon bounds into one DataArray.
    
    f: glob string
    """
    fdir = '/export/data/scratch/tropomi/no2/'
    fpath = os.path.join(fdir, f)
    # Load NO2 and qa_value
    ds = xr.open_mfdataset(fpath, group='PRODUCT',
                           concat_dim='scanline', preprocess=dsmod)
    # Load latitude_bounds and longitude_bounds
    bds = xr.open_mfdataset(fpath, group='/PRODUCT/SUPPORT_DATA/GEOLOCATIONS',
                       concat_dim='scanline').squeeze('time')
    
    # Assign lat/lon bounds from bds to data variables in ds
    ds = ds.assign(latitude_bounds = bds['latitude_bounds'])
    ds = ds.assign(longitude_bounds = bds['longitude_bounds'])
    
    # Keep only NO2, qa_value, lat/lon bounds
    ds = ds[['nitrogendioxide_tropospheric_column', 'qa_value', 
             'longitude_bounds', 'latitude_bounds']]
    
    # Make into 1D array
    ds = ds.stack(sounding=['scanline', 'ground_pixel'])
    
    # Remove datapoints with qa_value <= 0.75
    ds = ds.where(ds['qa_value'] > 0.75, drop=True)
    
    return ds

#############################

if __name__ == '__main__':
    f = '*__20200504*_*.nc'
    g = '*__20200505*_*.nc'
    # Work with this specific dataframe so that I get the same value as test_aggregate.py
    # f= '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'
    ds1 = dsread(f)
    ds2 = dsread(g)
    
    d = pd.to_datetime(ds1.time.data).week
    
    
