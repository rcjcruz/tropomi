
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: pickle_files.py

Script containing functions:
    - create_toronto_list(f)
    - create_date_list(f)
    - create_toronto_orbits_by_date(f)
    - pickle_files(f)
    
Create a list of the Toronto inventory and a dictionary of the orbits sorted
by date. Return a directory /export/data/scratch/tropomi_gta/pkl which includes
pickle files of xr.DataArray objects for the datasets of each day.
"""

import os
import time
import pickle
import _pickle as cPickle
import fnmatch
import glob
# import pprint
# import bz2 # for compressing 

from paths import *
import open_tropomi as ot

# Load list of all files written in toronto_inventory.txt
toronto_files = os.path.join(inventories, 'toronto_inventory.txt')

######################


def create_toronto_list(f):
    """
    Return a list of all the file names extracted from f.

    Args:
        f (str): file name of inventory of Toronto orbits.
    """
    toronto_list = []
    with open(f, 'r') as toronto_inv:
        for test_file in toronto_inv:
            stripped_line = test_file.strip()
            toronto_list.append(stripped_line)

    return toronto_list

######################


def create_date_list(f):
    """
    Return a list of the dates of all orbits over Toronto given f.

    Args:
        f (str): file name of inventory of Toronto orbits.
    """

    toronto_list = create_toronto_list(f)

    list_of_dates = []
    for orbit in toronto_list:
        date = orbit[53:61]
        if date not in list_of_dates:
            list_of_dates.append(date)

    # Sort list of dates if not already sorted
    list_of_dates = sorted(list_of_dates)

    return(list_of_dates)

######################


def create_toronto_orbits_by_date(f):
    """
    Return dict of Toronto orbits sorted by date given files in f.

    f (str): file name of inventory of Toronto orbits.
    """
    list_of_toronto_files = create_toronto_list(f)
    list_of_dates = create_date_list(f)
    files_dict = {}

    for date in list_of_dates:
        date_glob = '*__%s*.nc' % date
        matching_files = fnmatch.filter(list_of_toronto_files, date_glob)
        files_dict[date] = matching_files

    return(files_dict)

######################


def pickle_files(f):
    """
    Pickle all .nc files written in f into ../pkl directory.

    Args:
        f (str): file name of inventory of Toronto orbits.
    """
    # Create dictory of Toronto orbits sorted by date
    dict_of_toronto_orbits = create_toronto_orbits_by_date(toronto_files)
    dates = list(dict_of_toronto_orbits.keys())

    start_time = time.time()
    fdir = tropomi_pkl  # directory to store pickle files
    i = 1  # counter
    
    # Get list of pkl files
    fpath = os.path.join(fdir, '*')
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

            output_file = os.path.join(fdir, date)
            # Pickle files
            # with bz2.BZ2File(output_file + '.pbz2', 'w') as outfile:
            #     print('Pickling %s' % date)
            #     cPickle.dump(ds, outfile)
            with open(output_file, 'wb') as outfile:
                print('Pickling %s' % date)
                pickle.dump(ds, outfile)

            print("[%s] --- %s seconds ---" % (i, (time.time() - start_time_iter)))

            i += 1

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))
    
######################


if __name__ == '__main__':
    # list_of_toronto_files = create_toronto_list(toronto_files)
    # list_of_dates = create_date_list(toronto_files)
    # dict_of_toronto_orbits = create_toronto_orbits_by_date(toronto_files)
    pickle_files(toronto_files)
