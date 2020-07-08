#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: open_wind_tropomi.py

Script contains functions:
    - dsmod(ds)
    - dsread(f, city='toronto')

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
import sys
from datetime import timedelta
from paths import *

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
    
    ds (xr.DataSet): TROPOMI NO2 dataset.
    """
    
    # remove size-1 dimension time
    ds = ds.squeeze('time')
    # changes the index to time_utc
    ds = ds.set_index(scanline='time_utc')
    
    return ds

#############################

def dsread(f, city='toronto'):
    """
    Read netCDF4 files and access PRODUCT and GEOLOCATIONS folders to 
    extract NO2, qa_value, lat/lon bounds into one DataArray.
    
    f (glob string): file name.
    city (str): city name.
    
    >>> ds = dsread('20200501', city='toronto')
    """
    
    # Create list of TROPOMI files found in city inventory for a given date.
    f_inv = '{}/{}_inventory.txt'.format(city, city)
    fpath_inv = os.path.join(inventories, f_inv)
    city_inv = open(fpath_inv, 'r+').read().splitlines()
    f_str = '__' + f
    files = [s for s in city_inv if f_str in s]
    
    print('Reading', f)
    # Load NO2 and qa_value
    ds = xr.open_mfdataset(files, group='PRODUCT',
                           concat_dim='scanline',
                           preprocess=dsmod)
    # Load latitude_bounds and longitude_bounds
    bds = xr.open_mfdataset(files, group='/PRODUCT/SUPPORT_DATA/GEOLOCATIONS',
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
    # ds = dsread('*__20200501*.nc')
    ds = dsread('20200501', city='toronto')

    # test_file = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200501T164929_20200501T183100_13213_01_010302_20200503T094618.nc'
    # data2 = xr.open_dataset(test_file, group='/PRODUCT')