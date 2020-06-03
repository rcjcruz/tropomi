#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import sys
import xarray as xr
import time
from collections import namedtuple

# fpath = "/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc"
# fpath = "/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T070610_20200505T084741_13264_01_010302_20200507T000819.nc"

##############################


def get_toronto_files(f):

    # Load path to NO2 files
    fdir = '/export/data/scratch/tropomi/no2/'
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

    # Create text file of Toronto inventory or empties it if it has contents
    if os.path.getsize(output_file) > 0:
        open(output_file, 'w').close()

    # Open the text file to write
    file_object = open(output_file, "w+")

    # Keep track of start time of proeess
    start_time = time.time()

    # Iterate over all files in no2 directory
    files = sorted(glob.glob(fpath))

    for i in range(len(sorted(glob.glob(fpath)))):
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
                      (i + 1, files[i]))
                file_object.writelines([files[i], '\n'])

            else:
                print('[%s] %s does not include an orbit over Toronto' %
                      (i + 1, files[i]))

            print("--- %s seconds ---" % (time.time() - start_time_iter))

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))


if __name__ == '__main__':
    f = '*.nc'
    get_toronto_files(f)
