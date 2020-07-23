#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_wind_and_grid.py

Script containing functions:
    - get_wind_speed_and_dir(ds)
    - add_wind(f, city='toronto')
    - grid_data(ds, city='toronto', res=0.05)
    - add_wind_and_grid(f, city-'toronto')

Functions to add wind data (speed and bearing obtained from 
get_wind_speed_and_dir) to colocated NO2 observations, to aggregate NO2 TVCD 
wind speed, and bearing over uniform lat/lon grid over a given city.
"""

import xarray as xr
import numpy as np
import pandas as pd
import time
import glob
import os
import pickle

from paths import *
import points_of_interest as poi
import open_tropomi as ot


def get_wind_speed_and_dir(ds):
    """
    Return wind speed and bearing given u and v of ds.

    Args:
        ds (xr.Dataset): dataset of wind containing U850 and V850 variables.

    Returns:
        ds (xr.Dataset): datasets of wind with speed and bearing variables appended.
    """
    # Load U850 and V850 variables
    if ('U850' not in ds.variables) or ('V850' not in ds.variables):
        raise KeyError(
            '"U850" and "V850" are required to calculate wind speed.')
    else:
        u = ds.U850
        v = ds.V850
        speed = np.sqrt(u**2 + v**2)
        bearing = np.degrees(np.arctan2(v, u))

    return speed, bearing.where(bearing > 0, bearing + 360)

################################


def add_wind(f, city='toronto'):
    """
    Return a dataset for data f over city with wind data that matches lat/lon/time
    of TROPOMI observation.

    Args:
        f (str): date string of the form YYYYMMDD.
        city (str): city of interest.

    Returns:
        no2 (xr.Dataset): dataset of NO2 TVCD with eastward (u) and northward (v)
            wind components.

    >>> no2 = add_wind('20200501', 'toronto')
    """

    start_time = time.time()

    # Load city limits
w, e, s, n = poi.get_plot_limits(city=city, extent=1, res=0)

    # Load dataset
    no2 = ot.dsread(f, city)
    # Subset NO2 dataset over +-1 deg lat/lon around the city
no2 = no2.where((no2.longitude >= w) & (no2.longitude <= e) & (no2.latitude >= s) & (no2.latitude <= n), drop=True)
    if no2.nitrogendioxide_tropospheric_column.size == 0:
        return None
    no2 = no2.rename({'time': 'measurement_time'})  # rename time
    # create u-component variable
    no2['u'] = (['sounding'], np.zeros([no2.sounding.size]))
    # create v-component variable
    no2['v'] = (['sounding'], np.zeros([no2.sounding.size]))

    # Load wind
    f_str = '*' + f + '*'
    fpath = os.path.join(winds, f_str)
    for file in glob.glob(fpath):
        wind = xr.open_dataset(file)
        interp_wind = wind.interp(
            lat=no2.latitude, lon=no2.longitude, method='linear')
        interp_wind = interp_wind.dropna(dim='sounding')

    # iterate over each observation and append wind speed and bearing to no2
    for i in range(len(no2.scanline)):
        print('Reading scanline', i)
        # Load timestamp of observation
        t_obs = pd.to_datetime(no2.scanline[i].values)
        hour = t_obs.hour
        lat, lon = no2.latitude.values[i], no2.longitude.values[i]
        # load averaged winds from hour
        winds_from_hour = interp_wind.isel(time=hour)

        for j in range(len(winds_from_hour.U850)):
            # add uv- wind components to matching lat/lon/timestamp
            if ((winds_from_hour.lon.values[j] == lon) and
                    (winds_from_hour.lat.values[j] == lat)):
                no2.u[i] += winds_from_hour.U850.values[j]
                no2.v[i] += winds_from_hour.V850.values[j]

    # pickle files
    fdir = winds_pkl + city + '/'
    filename = f + '_raw'
    output_file = os.path.join(fdir, filename)
    with open(output_file, 'wb') as outfile:
        print('Pickling %s' % f)
        pickle.dump(no2, outfile)
    return no2

################################


def grid_data(ds, city='toronto', res=0.05):
    """
    Return a xr.Dataset with NO2, wind speed, and bearing aggregated and averaged
    over a uniform lat/longrid over city with a resolution res. Pickles dataset in 
    '/export/data/scratch/tropomi_rc/wind_pkl/'

    Args:
        ds (xr.Dataset): dataset of raw NO2 TVCD and interpolated wind speed and
            bearing.
        city (str): city name.
        res (float): resolution.

    Returns:
        new_ds (xr.Dataset): dataset of aggregated and averaged NO2 TVCD with
            eastward (u) and northward (v) wind components.

    >>> ds = add_wind('20200501', 'toronto')
    >>> ds = grid_data(ds, 'toronto')
    """

    # Lat/lon max/min
    lonmn, lonmx, latmn, latmx = poi.get_plot_limits(city=city, extent=1, res=res)


    # Create a uniform lat/lon grid
    lat_bnds = np.arange(latmn, latmx, res)
    lon_bnds = np.arange(lonmn, lonmx, res)

    # arr will accumulate the values within each grid entry
    no2_arr = np.zeros([lat_bnds.size, lon_bnds.size])
    u_arr = np.zeros([lat_bnds.size, lon_bnds.size])
    v_arr = np.zeros([lat_bnds.size, lon_bnds.size])
    # dens_arr will count the number of observations that occur within that grid entry
    dens_arr = np.zeros([lat_bnds.size, lon_bnds.size], dtype=np.int32)

    # Load datasets
    no2 = ds.nitrogendioxide_tropospheric_column.values
    u = ds.u.values
    v = ds.v.values
    lat = ds.latitude.values
    lon = ds.longitude.values

    # Check if the lat and lon values are found within lat/lon bounds
    lat_flt = (lat > latmn) * (lat < latmx)
    lon_flt = (lon > lonmn) * (lon < lonmx)

    # Create array to filter data points found within lat/lon bounds
    filter_arr = lat_flt * lon_flt

    # Keep no2 values that are within the bounded lat/lon
    no2 = no2[filter_arr]
    u = u[filter_arr]
    v = v[filter_arr]

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
        lat_inds = np.searchsorted(
            lat_bnds, np.array([vlatmn[i], vlatmx[i]]))
        lon_inds = np.searchsorted(
            lon_bnds, np.array([vlonmn[i], vlonmx[i]]))

        # Obtain the lat/lon indices that will be used to slice val_arr
        lat_slice = slice(lat_inds[0], lat_inds[1]+1)
        lon_slice = slice(lon_inds[0], lon_inds[1]+1)

        # Add the NO2 values that fit in those lat/lon grid squares to val_arr and
        # add 1 to dens_arr to increase the count of observations found in that
        # grid square
        no2_arr[lat_slice, lon_slice] += no2[i]
        u_arr[lat_slice, lon_slice] += u[i]
        v_arr[lat_slice, lon_slice] += v[i]
        dens_arr[lat_slice, lon_slice] += 1

    # Divide val_arr by dens_arr; if dividing by 0, return 0 in that entry
    # no2_arr = no2_arr.clip(min=0)

    no2_arr_mean = np.divide(no2_arr, dens_arr, out=(
        np.zeros_like(no2_arr)), where=(dens_arr != 0))
    u_arr_mean = np.divide(u_arr, dens_arr, out=(
        np.zeros_like(u_arr)), where=(dens_arr != 0))
    v_arr_mean = np.divide(v_arr, dens_arr, out=(
        np.zeros_like(v_arr)), where=(dens_arr != 0))

    # CREATE NEW DATASET WITH NO2, WS, AND BEARING FOR EACH LAT, LON
    new_ds = xr.Dataset({
        'no2': xr.DataArray(
            data=np.array([no2_arr_mean]),   # enter data here
            dims=['time', 'latitude', 'longitude'],
            coords={'latitude': ('latitude', lat_bnds),
                    'longitude': ('longitude', lon_bnds),
                    'time': np.array([ds.measurement_time.values])},
            attrs={'units': 'mol m-2'}),
        'u': xr.DataArray(
            data=np.array([u_arr_mean]),   # enter data here
            dims=['time', 'latitude', 'longitude'],
            coords={'latitude': ('latitude', lat_bnds),
                    'longitude': ('longitude', lon_bnds),
                    'time': np.array([ds.measurement_time.values])},
            attrs={'units': 'm/s'}),
        'v': xr.DataArray(
            data=np.array([v_arr_mean]),   # enter data here
            dims=['time', 'latitude', 'longitude'],
            coords={'latitude': ('latitude', lat_bnds),
                    'longitude': ('longitude', lon_bnds),
                    'time': np.array([ds.measurement_time.values])},
            attrs={'units': 'm/s'})},
        attrs={'description': 'dataset for NO2 TVCD, wind speed, and bearing'})

    return new_ds

################################


def add_wind_and_grid(f, city='toronto'):
    """
    Return dataset for f over city with no2, wind speeds, and bearing 
    averaged. Dataset is pickled to 
    '/export/data/scratch/tropomi_rc/wind_pkl/city/'

    Args: 
        f (str): date of observation in format YYYYMMDD.
        city (name): city name. 

    Returns:
        ds (xr.Dataset): dataset of aggregated and averaged NO2 TVCD with
            eastward (u) and northward (v) wind components.
    """
    start_time = time.time()
    
    # add wind and grid data
    ds = add_wind(f, city)

    if ds != None:
        gridded_ds = grid_data(ds, city)

        # count number of nonzero values in no2
        nonzero = np.count_nonzero(gridded_ds.no2.values)
        total = np.size(gridded_ds.no2.values)

        fdir = winds_pkl + city + '/'
        if nonzero / total < 0.25:
            # ins = insufficient
            print('{} has insufficient data but will be pickled.'.format(f))
            filename = f + '_ins'
        else:
            print('{} has sufficient data.'.format(f))
            filename = f + '_avg'
        # Pickle files
        output_file = os.path.join(fdir, filename)
        with open(output_file, 'wb') as outfile:
            print('Pickling %s' % filename)
            pickle.dump(gridded_ds, outfile)

        # return processing time
        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds))
        print('------------------------------------------------------------')
        return gridded_ds

    else:
        print('{} had insufficient data.'.format(f))
        print('------------------------------------------------------------')


if __name__ == '__main__':
    city = 'toronto'
    f = '20200510'
    ds = add_wind(f)
    # for i in np.arange(20200501, 20200532, 1):
    #     f = str(i)
    #     add_wind_and_grid(f)
        
    # for j in np.arange(20190501, 20190532, 1):
    #     f = str(j)
    #     add_wind_and_grid(f)
        
    # for k in np.arange(20180501, 20180502, 1):
    #     f = str(k)
    #     add_wind_and_grid(f)



    # fpath = winds_pkl + city + '/20200507_raw'
    # infile = open(fpath, 'rb')
    # ds = pickle.load(infile)
    # infile.close()

    # ds = grid_data(ds)
