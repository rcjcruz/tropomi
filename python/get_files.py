#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: get_files.py

Script containing functions:
    - get_city_files(f, city='toronto', extent=1, append_new=False)
    - create_city_list(f)
    - create_date_list(f)
    - create_city_orbits_by_date(f)
    - pickle_files(f, city='toronto')
        pickle daily .nc files into xr.DataArray and store in 
        /export/data/scratch/tropomi_rc/day_pkl/city
    - aggregate_files(**kwargs)
        aggregate pickled files into two-week bins and store them in 
        inventory files in ~/tropomi/inventories/city
    
Return a text file which contains all .nc files in 
/export/data/scratch/tropomi/no2 directory which contain an orbit over 
the city of interest.

Creates text file inventory_WXX_XX.txt for all pkl files which fall
within a two-week period. 
"""


import time
import datetime
import calendar
from datetime import timedelta
import glob
import os
import sys
import textwrap
import os
import pickle
import fnmatch

import _pickle as cPickle
import xarray as xr
import pandas as pd
from collections import namedtuple

from paths import *
import open_tropomi as ot
import points_of_interest as poi
import calendar_functions as cf

######################


def get_city_files(f, city='toronto', extent=1, append_new=False):
    """
    TODO: only iterate over new files and append
    Return .txt file containing .nc files with orbits over city given 
    file f. If .txt file exists, load append_new=True to append new files.

    Args:
        f (str): file name of TROPOMI NO2 .nc files.
        city (str): city name. Valid cities: 'toronto', 'vancouver', 'montreal',
            'new_york', 'los_angeles'. Default: 'toronto'
        extent (int): the lon/lat extension from the city of interest. 
            Default: 1
        append_new (bool): if True, only append new files not found in existing
            .txt file. Default: False.
    """
    
    # Check if valid city, and if so, create plot limits around city
    if city not in poi.cities.keys():
        return ValueError('Invalid city. City must be %s' % list(poi.cities.keys()))
    else:
        # Create plot_limits surrounding city
        extent = 5
        city_coords = poi.cities[city]
        plot_limits = (city_coords.lon-extent,
                       city_coords.lon+extent,
                       city_coords.lat-extent,
                       city_coords.lat+extent)
        e, w, s, n = plot_limits

    # Load output file; location is ~/tropomi/inventories/city/
    output_file = '{}/{}_inventory.txt'.format(city, city)
    output_fpath = os.path.join(inventories, output_file)

    if append_new:
        # Open the text file with pointer at beginning
        file_object = open(output_fpath, "r+")
    else:
        file_object = open(output_fpath, "w+")

    # Keep track of start time of proeess
    start_time = time.time()

    # Load path to NO2 files
    fdir = tropomi_no2
    fpath = os.path.join(fdir, f)
    
    # Iterate over all files in no2 directory
    files = sorted(glob.glob(fpath))

    if append_new:
        if 'OFFL' not in f:
            return ValueError('f must include \'OFFL\'')
        # Load text from city inventory .txt file
        text = file_object.readlines()
        print(sorted(text))

        # Obtain last date of observations already catalogued; only use
        # files with 'OFFL' because 'RPRO' stopped in 2019
        offl_files = []
        for file in text:
            if 'OFFL' in file:
                offl_files.append(file)
        last_date = offl_files[-1][53:61]

        j = 1
        # Check at the bottom of the list and append if date of file is
        # greater than the last added file to the inventory
        for i in reversed(range(len(files))):
            date_of_obs = files[i][53:61]
            if date_of_obs > last_date:
                with xr.open_dataset(
                        files[i], group='/PRODUCT')['nitrogendioxide_tropospheric_column'] as ds:
                    # Keep track of start time of iteration
                    start_time_iter = time.time()

                    # Check if ds contains values over Toronto
                    extracted = ds.where(
                        (ds.longitude > e) &
                        (ds.longitude < w) &
                        (ds.latitude > s) &
                        (ds.latitude < n), drop=True)

                    # If extracted data is not empty, write the file name to
                    # the output_file
                    if len(extracted.data) != 0:
                        print('[{}] {} includes an orbit over {}'.format(
                            j, files[i], city))
                        file_object.writelines([files[i], '\n'])

                    else:
                        print('[{}] {} does not include an orbit over {}'.format(
                            j, files[i], city))

                    print("--- %s seconds ---" %
                          (time.time() - start_time_iter))
                    i += 1
                    j += 1
    else:
        j = 1
        for i in range(len(files)):
            with xr.open_dataset(
                    files[i], group='/PRODUCT')['nitrogendioxide_tropospheric_column'] as ds:
                # Keep track of start time of iteration
                start_time_iter = time.time()

                # Check if ds contains values over Toronto
                extracted = ds.where(
                    (ds.longitude > e) &
                    (ds.longitude < w) &
                    (ds.latitude > s) &
                    (ds.latitude < n), drop=True)

                # If extract_toronto data is not empty, write the file name to
                # the output_file
                if len(extracted.data) != 0:
                    print('[{}] {} includes an orbit over {}'.format(
                        j, files[i], city))
                    file_object.writelines([files[i], '\n'])

                else:
                    print('[{}] {} does not include an orbit over {}'.format(
                        j, files[i], city))

                print("--- %s seconds ---" % (time.time() - start_time_iter))
                i += 1
                j += 1

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

#######################


def create_city_list(f):
    """
    Return a list of all the file names extracted from f.

    Args:
        f (str): file name of inventory of city orbits.

    >>> list_of_toronto_files = city_list(toronto_files)
    """

    city_list = []
    with open(f, 'r') as city_inv:
        for test_file in city_inv:
            stripped_line = test_file.strip()
            city_list.append(stripped_line)

    return city_list

######################


def create_date_list(f):
    """
    Return a list of the dates of all orbits over a city given f.

    Args:
        f (str): file name of inventory of city orbits.

    >>> list_of_dates = create_date_list(toronto_files)
    """

    city_list = create_city_list(f)

    list_of_dates = []
    for orbit in city_list:
        date = orbit[53:61]
        if date not in list_of_dates:
            list_of_dates.append(date)

    # Sort list of dates if not already sorted
    list_of_dates = sorted(list_of_dates)

    return(list_of_dates)

######################


def create_city_orbits_by_date(f):
    """
    Return dict of city orbits sorted by date given files in f.

    f (str): file name of inventory of city orbits.

    >>> dict_of_toronto_orbits = create_toronto_orbits_by_date(toronto_files)
    """

    list_of_city_files = create_city_list(f)
    list_of_dates = create_date_list(f)
    files_dict = {}

    for date in list_of_dates:
        date_glob = '*__%s*.nc' % date
        matching_files = fnmatch.filter(list_of_city_files, date_glob)
        files_dict[date] = matching_files

    return(files_dict)

######################


def pickle_files(f, city='toronto'):
    """
    Open all .nc files written in f as xr.Datasets and pickle into 
    /export/data/scratch/tropomi_rc/day_pkl/[city] directory under the date
    of orbit.

    Args:
        f (str): file name of inventory of city orbits.

    >>> toronto_files = os.path.join(inventories, 'toronto/toronto_inventory.txt')
    >>> pickle_files(toronto_files, city='toronto')
    """
    # Create dictory of Toronto orbits sorted by date
    dict_of_city_orbits = create_city_orbits_by_date(f)
    dates = list(dict_of_city_orbits.keys())

    start_time = time.time()
    fdir = tropomi_pkl_day  # directory to store daily pickle files
    i = 1  # counter

    # Get list of pkl files
    fpath = os.path.join(fdir, '{}/*'.format(city))
    pkl_list = sorted(glob.glob(fpath))
    date_list = []

    for file in pkl_list:
        date_list.append(file[-8:])

    for date in dates:
        if date not in date_list:
            start_time_iter = time.time()

            f = '*__%s*.nc' % date

            # Read all .nc files for a date into a xr.DataArray
            ds = ot.dsread(f)

            # Save pickled file to /export/data/scratch/tropomi_rc/day_pkl/city
            pkl_path = fdir + city + '/'
            output_file = os.path.join(pkl_path, date)

            # Pickle files
            with open(output_file, 'wb') as outfile:
                print('Pickling %s' % date)
                pickle.dump(ds, outfile)

            print("[%s] --- %s seconds ---" %
                  (i, (time.time() - start_time_iter)))

            i += 1

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

#######################


def aggregate_files_into_txt(aggregate='weekly', city='toronto', **kwargs):
    """
    Return .txt file containing pickled TROPOMI files found within a two week 
    range given date. Pickled TROPOMI files must be found in 
    /export/data/scratch/tropomi_rc/city.

    Args: 
        aggregate (str): aggregation type. Accepts: 'monthly', 'weekly'. 
            Default: 'weekly'

    Kwargs:
        year (int): year of interest
        month (int): month of interest; 1 <= month <= 12
        day (int): day of interest
        calendar_week (int): calendar week of interest; 1 <= calendar_week <= 52

    Returns:
        .txt file with strings of dates for orbits found in week range given
        date.
    """

    if aggregate == 'weekly':
        # year, month, day format
        if ('year' in kwargs) and ('month' in kwargs) and ('day' in kwargs):
            year, month, day = kwargs['year'], kwargs['month'], kwargs['day']

            # Get week number for date
            calendar_week = cf.get_odd_week_number(year, month, day)

            # Load start and end date of two week range
            start, end = cf.get_start_and_end_date_from_calendar_week(
                year, calendar_week)

        # year, calendar_week format
        elif ('year' in kwargs) and ('calendar_week' in kwargs):
            year, calendar_week = kwargs['year'], kwargs['calendar_week']

            if calendar_week % 2 == 0:
                print('--- Calendar week must be odd. '
                      'Subtracted 1 from calendar week and continued processing. ---')
                calendar_week -= 1

            start, end = cf.get_start_and_end_date_from_calendar_week(
                year, calendar_week)

        # raise error if not a valid kwargs combination
        else:
            return ValueError(textwrap.fill(
                textwrap.dedent("""Invalid kwargs. Valid kwargs combinations:
                                - year, month, day
                                - year, calendar_week""")))

        # Naming the inventory .txt file
        output_file = "{}/week/{}_W{:02d}_{:02d}.txt".format(city, year,
                                                             calendar_week,
                                                             (calendar_week + 1))  # file to save the results

        output_fpath = os.path.join(inventories, output_file)

        with open(output_fpath, "w+") as file_object:
            # Iterate over the files in the tropomi_rc/day_pkl/ directory and
            # append them to the inventory
            fpath = os.path.join(tropomi_pkl_day, '{}/*'.format(city))

            for file in glob.glob(fpath):
                # Find date of .nc file and convert to datetime object
                date = pd.to_datetime(
                    file[-8:], format='%Y%m%d', errors='ignore')

                # Write file to txt file if found between start and end and the
                # date of observation is not already written in the text file
                if start <= date <= end:
                    file_name = file[-8:]
                    file_object.seek(0)  # return to top of the text file
                    if file_name not in file_object.read():
                        file_object.writelines([file_name, '\n'])
                        print('Added ', date)
                    else:
                        pass

        # Print statement upon completion
        print('Created {} containing pkl files '
              'for weeks {} & {}, {} to {}'.format(output_file, calendar_week,
                                                   (calendar_week + 1), start, end))

        return start, end, calendar_week

    elif aggregate == 'monthly':
        # Year and month format
        if ('year' in kwargs) and ('month' in kwargs):
            year, month = kwargs['year'], kwargs['month']

            # Load start and end date of month
            start, end = cf.get_start_and_end_date_from_month(year, month)

        # raise error if not a valid kwargs combination
        else:
            return ValueError(textwrap.fill(
                textwrap.dedent("""Invalid kwargs. Valid kwargs combinations:
                                - year, month""")))

        # Naming the inventory .txt file
        # file to save the results
        output_file = "{}/month/{}_M{:02d}.txt".format(city, year, month)
        output_fpath = os.path.join(inventories, output_file)

        with open(output_fpath, "w+") as file_object:
            # Iterate over the files in the tropomi_rc/day_pkl/ directory and
            # append them to the inventory
            fpath = os.path.join(tropomi_pkl_day, '{}/*'.format(city))

            for file in glob.glob(fpath):
                # Find date of .nc file and convert to datetime object
                date = pd.to_datetime(
                    file[-8:], format='%Y%m%d', errors='ignore')

                # Write file to txt file if found between start and end and the
                # date of observation is not already written in the text file
                if start <= date <= end:
                    file_name = file[-8:]
                    file_object.seek(0)  # return to top of the text file
                    if file_name not in file_object.read():
                        file_object.writelines([file_name, '\n'])
                        print('Added ', date)
                    else:
                        pass

        # Print statement upon completion
        print('Created {} containing pkl files '
              'for month {}, {} to {}'.format(output_file, month, start, end))

        return start, end, month

    else:
        return TypeError('aggregate type must be "weekly" or "monthly".')

######################


if __name__ == '__main__':
    # 0) Choose city
    cities = ['toronto', 'montreal', 'los_angeles', 'vancouver', 'new_york']
    city_of_choice = cities[0]
    
    # 1) Read city files
    # f = '*.nc'
    # get_city_files(f, city=city_of_choice, append_new=False)

    # # 2) Pickle files
    # my_files = os.path.join(inventories, '{}/{}_inventory.txt'.format(city_of_choice, city_of_choice))
    # pickle_files(my_files, city=city_of_choice)

    # OPTIONAL:
    # list_of_files = create_toronto_list(vancouver_files)
    # list_of_dates = create_date_list(vancouver_files)
    # dict_of_montreal_orbits = create_toronto_orbits_by_date(vancouver_files)

    # 3) Create inventory text file for every couple weeks between 2018 and 2020
    # for i in range(2018, 2021):
    #     for j in range(1, 52, 2):
            # start, end, calendar_week = aggregate_files_into_txt(city=city_of_choice,
            #                                                     aggregate='weekly',
            #                                                     year=i, calendar_week=j)
    # OR create inventory text file given a month and year
    for city in cities:
        for i in range(2019, 2021):
            start, end, month = aggregate_files_into_txt(city=city,
                                                        aggregate='monthly',
                                                        year=i,
                                                        month=5)

    # # delete empty files
    # inv_list = glob.glob(os.path.join(inventories, '{}/week/*W*'.format(city_of_choice)))
    # for inv in inv_list:
    #     if os.path.getsize(inv) == 0:
    #         print('{} is empty and was deleted.'.format(inv))
    #         os.remove(inv)
