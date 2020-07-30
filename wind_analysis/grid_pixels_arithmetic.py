#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_pixels_arithmetic.py
"""

import xarray as xr
import numpy as np
import shapely.geometry as geometry
import pickle
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from paths import *
import points_of_interest as poi
import sort_winds as sw

def grid_arithmetic(date, data_type='cartesian', city='toronto', res=1., dist=70, avg=False):
    """

    Args:
        date (str or list): date of observation. Format: YYYYMMDD
        data_type (str): 'cartesian' or 'rotated'
        city (str): city of interest. Default: 'toronto'
        res (float): resolution of grid. Default: 1 km.
        plot (bool): plot gridded TROPOMI data. Default: True.

    Returns:

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

    elif isinstance(date, list):
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
    no2_avg = np.zeros([y_grid.size, x_grid.size])
    no2_median = np.zeros([y_grid.size, x_grid.size])
    error_grid = np.zeros([y_grid.size, x_grid.size])
    sem_grid = np.zeros([y_grid.size, x_grid.size])
    std_grid = np.zeros([y_grid.size, x_grid.size])

    # accumulate values to get mean adnd median
    no2_values = np.empty([y_grid.size, x_grid.size], object)
    for i in np.ndindex(no2_values.shape):
        no2_values[i] = []

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
    print('TOTAL:', len(no2))
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

                    # create a polygon representing the pixel
                    pixel_poly = geometry.Polygon([[p.x, p.y] for p in pixel])

                    # if the entire pixel is contained in the TROPOMI footprint:
                    if (poly.contains(pixel[0]) and poly.contains(pixel[1])
                            and poly.contains(pixel[2]) and poly.contains(pixel[3])):

                        # area frac is overlapping pixel area / total footprint of TROPOMI
                        area_frac = pixel_poly.area / footprint
                    else:
                        intersect = pixel_poly.intersection(poly)
                        area_frac = intersect.area / footprint

                    # append no2 value * area frac to pixel's list
                    no2_values[j, i].append(no2[k] * area_frac)

    if avg:
        print('... Averaging ...')
        for n in range(len(no2_values)):
            for m in range(len(no2_values[n])):
                if len(no2_values[n, m]) == 0:  # remove divide by zero cases
                    no2_avg[n, m] = None
                    no2_median[n, m] = None
                    error_grid[n, m] = None
                    sem_grid[n, m] = None
                    std_grid[n, m] = None
                else:
                    no2_avg[n, m] = np.sum(no2_values[n, m]) / len(no2_values[n, m])
                    no2_median[n, m] = np.median(no2_values[n, m])
                    std_grid[n, m] = np.std(no2_values[n, m])
                    sem_grid[n, m] = stats.sem(no2_values[n, m], ddof=0)
                    error_grid[n, m] = np.maximum(sem_grid[n, m], std_grid[n, m])

    else:
        print('... Not averaging ...')

    # create dataset
    ds = xr.Dataset({'x_coords': xr.DataArray(x_grid, dims=['x'], coords=[x_grid],
                                              attrs={'description': 'distance from origin in along x-axis', 'units': 'km'}),
                     'y_coords': xr.DataArray(y_grid, dims=['y'],
                                              coords=[y_grid], attrs={'description': 'distance from origin in along y-axis', 'units': 'km'}),
                     'no2_raw': xr.DataArray(no2_values,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 tropospheric vertical column weighted average', 'units': 'mol m-2'}),
                     'no2_avg': xr.DataArray(no2_avg,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 tropospheric vertical column arithmetic mean', 'units': 'mol m-2'}),
                     'no2_median': xr.DataArray(no2_median,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 tropospheric vertical column median', 'units': 'mol m-2'}),
                     'no2_sem': xr.DataArray(sem_grid,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 tropospheric vertical column standard error in the mean', 'units': 'mol m-2'}),
                     'no2_sem': xr.DataArray(std_grid,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 tropospheric vertical column standard deviation', 'units': 'mol m-2'}),
                     'no2_error': xr.DataArray(error_grid,
                                             dims=['y', 'x'],
                                             coords=[y_grid, x_grid],
                                             attrs={'description': 'NO2 tropospheric vertical column error', 'units': 'mol m-2'}),
                    },
                    attrs={'time': date})

    return ds

if __name__ == '__main__':
    ds1 = grid_arithmetic('20200520', data_type='cartesian', res=1, dist=30, avg=True)
    ds2 = grid_arithmetic('20200520', data_type='rotated', res=1, dist=30, avg=True)
    print('... Plotting ...')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('May 20, 2020')
    im1 = ax1.pcolormesh(ds1.x_coords.values, ds1.y_coords.values, ds1.no2_avg.values, cmap='Blues')
    im2 = ax2.pcolormesh(ds2.x_coords.values, ds2.y_coords.values, ds2.no2_avg.values, cmap='Blues')
    ax1.scatter(0, 0)
    ax1.annotate('toronto', (0, 0))
    plt.colorbar(im1, orientation='horizontal')
    ax2.scatter(0, 0)
    ax2.annotate('toronto', (0, 0))
    plt.show()
