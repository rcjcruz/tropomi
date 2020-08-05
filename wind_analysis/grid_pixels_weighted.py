#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_pixels_weighted.py

Functions: 
    - grid_weighted(date, data_type, city='toronto', dist=70., res=1., avg=False)
    - average(tf, data_type, city='toronto', dist=70., res=1., wind_type='all', pickle_bool=True)
"""

import xarray as xr
import numpy as np
import shapely.geometry as geometry
import pickle
import os
import glob
import string
import pandas as pd
import matplotlib.pyplot as plt           


from paths import *
import points_of_interest as poi
import sort_winds as sw

# if data_type is rotate, need to first fit a 2D gaussian to cartesian values 
# then find the major axis from the most representative level

# the loaded cartesian dataset has an average_wind_direction 

def grid_weighted(date, data_type, city='toronto', dist=70., res=1., avg=False):
    """
    Return dataset with NO2 TVCD oversampled onto a uniform grid of resolution
    res with a maximum distance dist from the origin in the x-y direction. 

    Weighted average is calculated as NO2 TVCD * area fraction / (NO2 error)**2. 
    Area fraction is the intersection area  of the TROPOMI pixel and the grid 
    pixel / total area of TROPOMI pixel.
    Error is calculated as 1 / sqrt(sum(weights)) where the weights are 
    area fraction / (NO2 error)**2.

    Args:
        date (str or list): date of observation. Format: YYYYMMDD
        data_type (str): type of data. Accepted values: 'cartesian' or 'rotated'
        city (str): city of interest. Default: 'toronto'
        dist (float): distance (in x-y) from the origin. Default: 70 km.
        res (float): resolution of grid. Default: 1 km.
        avg (bool): determine if NO2 TVCD will be averaged. Default: False.

    Returns:
        ds (xr.Dataset): dataset including the weighted average of NO2 TVCD
            for each grid pixel (no2_avg) including error (no2_sem). Raw 
            data (no2_raw) and weights are included for further data processing. 
    
    # To grid one day 
    >>> ds = grid_weighted('20200401', data_type='cartesian', res=5, dist=70, avg=True)

    """
    if isinstance(date, str):
        f_str = city + '/' + date

        if data_type == 'cartesian':
            fpath = os.path.join(cartesian_pkl, f_str)
        elif data_type == 'rotated':
            fpath = os.path.join(rotated_pkl, f_str)
        else:
            return ValueError('Invalid data type. Must be "cartesian" or "rotated"')

        with open(fpath, 'rb') as infile:
            ds = pickle.load(infile)
            print('Opened', fpath)

    else:
        ds = date

    # load no2, error, coordinates, and bounds
    date = pd.to_datetime(ds.measurement_time.values).date()
    no2 = ds.no2.values
    er = ds.no2_error.values
    x, y = ds.x.values, ds.y.values
    xbd, ybd = ds.x_bounds.values, ds.y_bounds.values

    # create grids with
    x_grid = np.arange(-dist, dist+res, res, dtype=int)
    y_grid = np.arange(-dist, dist+res, res, dtype=int)
    weight_tot = np.zeros([y_grid.size, x_grid.size])
    no2_values = np.zeros([y_grid.size, x_grid.size])
    no2_avg = np.zeros([y_grid.size, x_grid.size])
    sem_grid = np.zeros([y_grid.size, x_grid.size])

    # Check if the lat and lon values are found within lat/lon bounds
    y_flt = (y > min(y_grid)) * (y < max(y_grid))
    x_flt = (x > min(x_grid)) * (x < max(y_grid))

    # Create array to filter data points found within lat/lon bounds
    filter_arr = y_flt * x_flt

    # Keep no2 values that are within the bounded lat/lon
    no2 = no2[filter_arr]
    er = er[filter_arr]
    x = x[filter_arr]
    y = y[filter_arr]
    xbd = [xbd[0][filter_arr], xbd[1][filter_arr],
           xbd[2][filter_arr], xbd[3][filter_arr]]
    ybd = [ybd[0][filter_arr], ybd[1][filter_arr],
           ybd[2][filter_arr], ybd[3][filter_arr]]

    print('... Searching ...')
    print('[{}] TOTAL:'.format(date), len(no2))
    for k in range(len(no2)):
        print('[{}] Reading scanline {} of {}'.format(date, k, len(no2)))
        # define the polygon
        points = [geometry.Point(xbd[0][k], ybd[0][k]),
                  geometry.Point(xbd[1][k], ybd[1][k]),
                  geometry.Point(xbd[2][k], ybd[2][k]),
                  geometry.Point(xbd[3][k], ybd[3][k])]
        poly = geometry.Polygon([[p.x, p.y] for p in points])
        footprint = poly.area  # determine the polygon area

        for i in range(x_grid.size):  # for each x in the grid
            for j in range(y_grid.size):  # for each y in the grid
                # if the polygon contains at least one corner of the grid pixel
                pixel = [geometry.Point(x_grid[i]+res/2, y_grid[j]+res/2),
                         geometry.Point(x_grid[i]-res/2, y_grid[j]+res/2),
                         geometry.Point(x_grid[i]-res/2, y_grid[j]-res/2),
                         geometry.Point(x_grid[i]+res/2, y_grid[j]-res/2)]

                if (poly.contains(pixel[0]) or poly.contains(pixel[1])
                        or poly.contains(pixel[2]) or poly.contains(pixel[3])):
                    pixel_poly = geometry.Polygon(
                        [[p.x, p.y] for p in pixel])  # create polygon of pixel

                    # if the entire pixel is contained in the TROPOMI footprint:
                    if (poly.contains(pixel[0]) and poly.contains(pixel[1])
                            and poly.contains(pixel[2]) and poly.contains(pixel[3])):

                        # area frac is overlapping pixel area / total footprint of TROPOMI
                        area_frac = pixel_poly.area / footprint
                    else:
                        intersect = pixel_poly.intersection(poly)
                        area_frac = intersect.area / footprint

                    # append no2 value * area frac to pixel's list
                    weight = area_frac / (er[k] ** 2)
                    no2_values[j, i] += (no2[k] * weight)
                    weight_tot[j, i] += weight

    if avg:
        print('... Averaging ...')
        for n in range(len(no2_values)):
            for m in range(len(no2_values[n])):
                if weight_tot[n, m] == 0:  # remove divide by zero cases
                    no2_avg[n, m] = None
                    sem_grid[n, m] = None
                else:
                    no2_avg[n, m] = no2_values[n, m] / weight_tot[n, m]
                    sem_grid[n, m] = 1 / (np.sqrt(weight_tot[n, m]))

    else:
        print('... Not averaging ...')

    # create dataset
    ds = xr.Dataset({'x_coords': xr.DataArray(x_grid, dims=['x'], coords=[x_grid],
                                              attrs={'description': 'distance from origin along x-axis', 'units': 'km'}),
                     'y_coords': xr.DataArray(y_grid, dims=['y'],
                                              coords=[y_grid], attrs={'description': 'distance from origin along y-axis', 'units': 'km'}),
                     'no2_raw': xr.DataArray(no2_values,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'raw NO2 tropospheric vertical column ', 'units': 'mol m-2'}),
                     'weights': xr.DataArray(weight_tot,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'weights calculated using overlapping pixel area / total TROPOMI pixel area all multiplied by 1 / the square of the error of the TROPOMI pixel.', 'units': 'mol m-2'}),
                     'no2_avg': xr.DataArray(no2_avg,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 TVCD weighted average', 'units': 'mol m-2'}),
                     'no2_sem': xr.DataArray(sem_grid,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 TVCD standard error in the weighted mean', 'units': 'mol m-2'}),
                     },
                    attrs={'time': date})
    return ds



def average(tf, data_type, city='toronto', dist=70., res=1., wind_type='all', pickle_bool=True):
    """
    Return a dictionary with wind_types as keys and an averaged dataset of 
    NO2 TVCD over a time frane of interest.

    Args:
        tf (str): timeframe of interest. Accepted values:
            'may_1819', 'may_20', 'march_19', 'march_20', 
            'april_19', 'april_20', 'june_19', 'june_20', 
            'pre-vid', 'covid'
        data_type (str): type of data. Accepted values: 'cartesian' or 'rotated'
        city (str): city of interest. Default: 'toronto'
        dist (float): distance (in x and y) from city. Default: 70 km. 
        res (float): resolution of grid. Default: 1 km.
        wind_type (str or list): wind types of interest. Accepted values: 
            'all', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', or a list of 
            a combination of these values.
        pickle_bool (bool): bool to determine if dictionary will be pickled.
            Default: True.
            
    >>> average('covid')
    """

    # create grid with resolution res
    x_grid = np.arange(-dist, dist+res, res, dtype=int)
    y_grid = np.arange(-dist, dist+res, res, dtype=int)

    # load dict of timeframe with datasets sorted by wind speed
    ws_dict, first, last = sw.sort_winds(tf=tf, city=city, return_dates=True)

    # check if looking over all wind types (A-G) or specific ones
    if wind_type == 'all':
        wind_types = list(ws_dict.keys())
    elif isinstance(wind_type, list):
        wind_types = wind_type
    elif isinstance(wind_type, str):
        wind_types = list(wind_type)
    else:
        return ValueError('wind_type must be "all", a string containing a single letter from A-G, or a list of letters.')

    avg_dict = {}
    # iterate over the wind_types
    for wt in wind_types:
        if ws_dict[wt] != []:  # check that the wind_type has datasets
            # create dataset for wind type to store average data values
            ds_list = []
            raw_list = []
            weights_list = []

            for ds in ws_dict[wt]:  # iterate over each ds in the wind_type
                # grids data without averaging
                gridded_ds = grid_weighted(ds, data_type=data_type, city=city,
                                           dist=dist, res=res, avg=False)
                ds_list.append(gridded_ds)
                raw_list.append(gridded_ds.no2_raw.values)
                weights_list.append(gridded_ds.weights.values)

            # sum the raw values (NO2 * weights) and weights
            raw = np.sum(raw_list, axis=0)
            weights = np.sum(weights_list, axis=0)
            no2_avg = np.zeros(raw.shape)
            sem = np.zeros(raw.shape)

            # calculate average NO2 TVCD and calculate error in weighted mean
            for i in np.ndindex(raw.shape):
                if weights[i] == 0:
                    no2_avg[i] = None
                    sem[i] = None
                else:
                    no2_avg[i] = raw[i] / weights[i]
                    sem[i] = 1 / np.sqrt(weights[i])

            # create dataset to store averaged data
            dist2 = dist * 2
            if data_type == 'cartesian':
                desc = 'dataset of Cartesian coordinates for TROPOMI for {}-{} timeframe at a {}km resolution over a {}x{}km box over {}'.format(
                    first, last, res, dist2, dist2, city)
            elif data_type == 'rotated':
                desc = 'dataset of rotated Cartesian coordinates for TROPOMI for {}-{} timeframe at a {}km resolution over a {}x{}km box over {}'.format(
                    first, last, res, dist2, dist2, city)

            avg_ds = xr.Dataset({'x_coords': ds.x,
                                 'y_coords': ds.y,
                                 'no2_avg': xr.DataArray(no2_avg,
                                                         dims=['y', 'x'],
                                                         coords=[
                                                             y_grid, x_grid],
                                                         attrs={'description': 'NO2 TVCD weighted average', 'units': 'mol m-2'}),
                                 'no2_sem': xr.DataArray(sem,
                                                         dims=['y', 'x'],
                                                         coords=[
                                                             y_grid, x_grid],
                                                         attrs={'description': 'NO2 TVCD standard error in the weighted mean', 'units': 'mol m-2'}),
                                 },
                                attrs={'description': desc,
                                       'timeframe': '{}-{}'.format(first, last),
                                       'wind type': wt,
                                       'wind speed': '{}-{} m/s'.format(poi.wind_type[wt][0], poi.wind_type[wt][1])})

            # store averaged dataset into wind_type in the averaged dictionary
            avg_dict[wt] = avg_ds

    # pickle the dictionary
    if pickle_bool and isinstance(wind_type, str):
        nomen_dict = poi.nomen_dict
        nomen = '{time}_{data_type}_{wind_type}_{dist}'.format(time=nomen_dict[tf],
                                                               data_type=nomen_dict[data_type],
                                                               wind_type=wind_type,
                                                               dist=str(dist))
        f_str = '{}/gridded/'.format(city) + nomen
        if data_type == 'cartesian':
            pkl_path = os.path.join(cartesian_pkl, f_str)
        elif data_type == 'rotated':
            pkl_path = os.path.join(rotated_pkl, f_str)

        print('Pickling', pkl_path)
        with open(pkl_path, 'wb') as outfile:
            pickle.dump(avg_dict, outfile)

    return avg_dict


