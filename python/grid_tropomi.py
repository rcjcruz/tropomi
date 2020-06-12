import os
import sys
import glob
import pickle
import time
import pandas as pd
import xarray as xr
import numpy as np
from pprint import pprint

import points_of_interest as poi
from paths import *
import calendar_functions as cf

# Open the inventory for the two-weeks of interest


def create_dict(f):
    """
    Return dictionary containing datasets of all files found in inventory f.

    Args:
        f (str): file name containing datasets for two-week range.
    Returns:
        files_dict (dict): dictionary containing datasets with date of 
            orbit as key.
    """

    fdir = inventories
    try:
        fpath = os.path.join(fdir, f)
        files = []

        with open(fpath, 'r') as file_list:
            for test_file in file_list:
                date = test_file.strip()
                files.append(date)
        # print(files)

    except:
        print(
            'Did not find a text file containing file names (perhaps name does not match)')
        sys.exit()

    # Create a dictionary of all the datasets found in the two week range
    # FUTURE: Make a pkl file for each dataset
    files_dict = {}
    for file in files:
        pkl_path = os.path.join(tropomi_pkl, file)
        with open(pkl_path, 'rb') as infile:
            ds = pickle.load(infile)
            files_dict[file] = ds

    return files_dict


# # create new dataset
# # week1920_ds = xr.concat(list(files_dict.values()), dim='time')


def aggregate_tropomi(f, res=0.05):
    """
    Return a xr.DataArray with averaged NO2 product aggregated over a uniform
    lat/lon grid with bounds defined by poi.plot_limits and resolution res.

    Args:
        f (str): file name containing datasets for two-week range.
        res (float): resolution of spacing. Default: 0.05 ~ 6km.
    Returns:
        new_ds (xr.Dataset): TROPOMI NO2 dataset aggregated into a 
            uniform lat/lon grid.
    """
    start_time = time.time()

    # Lat/lon max/min
    lonmn, lonmx, latmn, latmx = poi.plot_limits

    # Create a uniform lat/lon grid
    lat_bnds = np.arange(latmn, latmx, res)
    lon_bnds = np.arange(lonmn, lonmx, res)

    # val_arr will accumulate the values within each grid entry
    val_arr = np.zeros([lat_bnds.size, lon_bnds.size])
    # densarr will count the number of observations that occur within that grid entry
    dens_arr = np.zeros([lat_bnds.size, lon_bnds.size], dtype=np.int32)

    # Load files into dictionary
    year = int(f[25:29])
    ds_dict = create_dict(f)

    # Iterate over each date in two-week range and add values to val_arr and
    # dens_arr
    j = 1
    for date in list(ds_dict.keys()):
        print('[{}] Aggregating {}'.format(j, date))
        ds = ds_dict[date]
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
            val_arr[lat_slice, lon_slice] += no2[i]
            dens_arr[lat_slice, lon_slice] += 1

        j += 1

    # Set negative values in val_arr to 0
    val_arr = val_arr.clip(min=0)

    # Divide val_arr by dens_arr; if dividing by 0, return 0 in that entry
    val_arr_mean = np.divide(val_arr, dens_arr, out=(
        np.zeros_like(val_arr)), where=(dens_arr != 0))

    # Load week of orbit; will need to update when I work with multiple day
    dict_keys = sorted(list(ds_dict.keys()))
    first, last = dict_keys[0], dict_keys[-1]
    monday = pd.to_datetime(first)
    sunday = pd.to_datetime(last)
    week_num = cf.get_odd_week_number(monday.year, monday.month, monday.day)
    
    date = str(year) + ', weeks ' + \
        str(week_num) + ' to ' + str(week_num + 1)

    # Create a new DataArray where each averaged value corresponds to
    # lat/lon values that will be plotted on a PlateCarree projection
    new_ds = xr.DataArray(np.array([val_arr_mean]),
                          dims=('time', 'latitude', 'longitude'),
                          coords={'time': np.array([date]),
                                  'latitude': lat_bnds,
                                  'longitude': lon_bnds},
                          attrs={'weeks': '{:02d}-{:02d}'.format(week_num, week_num + 1),
                                 'first day': monday,
                                 'last day': sunday,
                                 'year': year})

    # Pickle new ds
    output_file = '{}_W{:02d}_{:02d}'.format(
        year, week_num, week_num + 1)
    pkl_path = os.path.join(tropomi_pkl_week, output_file)
    print(pkl_path)

    with open(pkl_path, 'wb') as outfile:
        pickle.dump(new_ds, outfile)

    # Print processing time
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

    return new_ds

#############################


if __name__ == '__main__':
    # f = 'inventory_2020_W01_02.txt'
    # # files_dict = create_dict(f)
    # ds = aggregate_tropomi(f)


# # to create pkl files for every two week range dict
    for i in range(1, 10, 2):
        fpath = os.path.join(inventories, '*W0{}*'.format(i))
        files = glob.glob(fpath)
        for test_file in files:
            ds=aggregate_tropomi(test_file)