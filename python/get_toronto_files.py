#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: get_toronto_files.py

Script contains functions:
    - get_toronto_files(f)

Return a text file toronto_inventory.txt which contains all .nc files 
in /export/data/scratch/tropomi/no2 directory which contain an orbit over 
Toronto. An orbit over Toronto is defined as passing over 
79.3832 +-5 W, 43.6532 +-5 N.
"""

import os
import glob
import sys
import xarray as xr
import time
from collections import namedtuple
from paths import *

# fpath = "/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc"
# fpath = "/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T070610_20200505T084741_13264_01_010302_20200507T000819.nc"

##############################


def get_toronto_files(f):


    # Load path to NO2 files
    fdir = tropomi_no2
    fpath = os.path.join(fdir, f)

    # Toronto coordinates
    Point = namedtuple('Point', 'lon lat')
    toronto_coords = Point(-79.3832, 43.6532)

    extent_size = 5
    plot_limits = (toronto_coords.lon-extent_size,
                   toronto_coords.lon+extent_size,
                   toronto_coords.lat-extent_size,
                   toronto_coords.lat+extent_size)

    e, w, s, n = plot_limits

    # Load output file
    output_file = 'toronto_inventory.txt'
    output_fpath = os.path.join(inventories, output_file)

    # Open the text file
    file_object = open(output_fpath, "r+")

    # Keep track of start time of proeess
    start_time = time.time()

    # Iterate over all files in no2 directory
    files = sorted(glob.glob(fpath))
    
    i=1
    for file in files:
        with xr.open_dataset(
                file, group='/PRODUCT')['nitrogendioxide_tropospheric_column'] as ds:
            # Keep track of start time of iteration
            start_time_iter = time.time()

            # Check if ds contains values over Toronto
            extract_toronto = ds.where(
                (ds.longitude > e) &
                (ds.longitude < w) &
                (ds.latitude > s) &
                (ds.latitude < n), drop=True)

            # If extract_toronto data is not empty, write the file name to
            # the output_file
            if len(extract_toronto.data) != 0:
                print('[%s] %s includes an orbit over Toronto' %
                      (i, files[i]))
                file_object.writelines([files[i], '\n'])

            else:
                print('[%s] %s does not include an orbit over Toronto' %
                      (i, files[i]))

            print("--- %s seconds ---" % (time.time() - start_time_iter))
            i+=1

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))


def add_new_files(f):
    """
    Append new files to toronto_inventory.txt.
    
    f: file path to TROPOMI no2 data.
    """
    
    fdir = tropomi_no2
    fpath = os.path.join(fdir, f)

    # Toronto coordinates
    Point = namedtuple('Point', 'lon lat')
    toronto_coords = Point(-79.3832, 43.6532)

    extent_size = 5
    plot_limits = (toronto_coords.lon-extent_size,
                   toronto_coords.lon+extent_size,
                   toronto_coords.lat-extent_size,
                   toronto_coords.lat+extent_size)

    e, w, s, n = plot_limits

    # Load output file
    output_file = 'toronto_inventory.txt'
    output_fpath = os.path.join(inventories, output_file)
    
    # Open the text file
    file_object = open(output_fpath, "r+")

    # Keep track of start time of proeess
    start_time = time.time()

    # Iterate over all files in no2 directory
    files = sorted(glob.glob(fpath))
    
    # Load text from toronto_inventory.txt
    text = file_object.readlines()
    
    offl_files = []
    for file in text:
        if 'OFFL' in file:
            offl_files.append(file)
    last_date = offl_files[-1][53:61]
    
    j=1
    for i in reversed(range(len(files))):
        date_of_obs = files[i][53:61]
        if date_of_obs > last_date:
            with xr.open_dataset(
                files[i], group='/PRODUCT')['nitrogendioxide_tropospheric_column'] as ds:
                # Keep track of start time of iteration
                start_time_iter = time.time()

                # Check if ds contains values over Toronto
                extract_toronto = ds.where(
                    (ds.longitude > e) &
                    (ds.longitude < w) &
                    (ds.latitude > s) &
                    (ds.latitude < n), drop=True)

                # If extract_toronto data is not empty, write the file name to
                # the output_file
                if len(extract_toronto.data) != 0:
                    print('[%s] %s includes an orbit over Toronto' %
                        (j, files[i]))
                    file_object.writelines([files[i], '\n'])

                else:
                    print('[%s] %s does not include an orbit over Toronto' %
                        (j, files[i]))

                print("--- %s seconds ---" % (time.time() - start_time_iter))
                j+=1

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

  
if __name__ == '__main__':
    f = '*.nc'
    # get_toronto_files(f)
    offl_f = '*OFFL*.nc'
    add_new_files(offl_f)

f1 = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20181022T172623_20181022T190753_05311_01_010200_20181028T192639.nc'
f2 = 'S5P_OFFL_L2__NO2____20200506T050543_20200506T064713_13277_01_010302_20200507T215646.nc'