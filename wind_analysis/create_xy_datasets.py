#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_xy_datasets.py

Script containing functions:
    - create_xy_dataset(ds, city='toronto')

Functions to create xarray.Dataset containing NO2 TVCD, wind speeds, and bearing
colocated to xy-coordinates with a city of interest as the origin.
"""
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from paths import *
import points_of_interest as poi
import convert as cv

def create_xy_dataset(ds, city='toronto'):
    """
    Return dataset with NO2, wind speed, and bearing with the corresponding
    xy-coordinates as variables. The origin of the Cartesian coordinates is 
    the city of interest.
    
    Args:
        ds (xr.Dataset): dataset containing NO2, wind speed, bearing, latitude,
            and longitude.
        city (str): city of interest.
    
    Returns:
        new_ds (xr.Dataest): dataset containing ds and xy-coordinates as 
            variables.
    """
    
    # load no2, wind speed, bearing, lat/lon
    no2 = ds.no2.where(ds.no2 > 0, np.nan)
    ws = ds.wind_speed
    bear = ds.bearing
    lat = ds.no2.latitude
    lon = ds.no2.longitude
    
    # create empty xy-coordinates arrays
    y = np.zeros_like(lat)
    x = np.zeros_like(lon)
    x, y = np.meshgrid(x, y)
    
    # load origin (city of interest)
    lon0, lat0 = poi.cities[city]
    for i in range(len(lon)):
        for j in range(len(lat)):
            x[i][j], y[i][j] = cv.convert_to_cartesian(lat0, lon0, lat[j], lon[i])
            
    # create dataset with xy coordinates as a data variable
    new_ds = xr.Dataset({'no2': no2,
                         'wind_speed': ws,
                         'bearing': bear,
                         'x_coords': xr.DataArray(data=x,
                                                  dims=['y', 'x'],
                                                  attrs={'units': 'km'}),
                        'y_coords': xr.DataArray(data=y,
                                                  dims=['y', 'x'],
                                                  attrs={'units': 'km'})},
                        attrs={'description': 'dataset for NO2 TVCD, wind speed, \
                        and bearing with corresponding xy-coordinates \
                             calculated using haversine formula.',
                               'origin': city})
    return new_ds

if __name__ == '__main__':
    city = 'toronto'
    fpath = winds_pkl + city + '/20200522_avg'
    infile = open(fpath, 'rb')
    ds = pickle.load(infile)
    infile.close()
    new_ds = create_xy_dataset(ds,'toronto')