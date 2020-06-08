#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: get_files.py

Script containing functions:
    - get_files(**kwargs)
    
Creates text file inventory_WXX_XX.txt for all pkl files which fall
within a two-week period. 
"""


import time
import datetime
from datetime import timedelta
import pandas as pd
from glob import glob
import os
import sys
import textwrap
from paths import *
import calendar_functions as cf
######################


def get_files(**kwargs):
    """
    Return .txt file with TROPOMI no2 files within a two week range given date.

    Kwargs:
        year (int): year of interest
        month (int): month of interest; 1 <= month <= 12
        day (int): day of interest
        calendar_week (int): calendar week of interest; 1 <= calendar_week <= 52

    Returns:
        .txt file with strings of dates for orbits found in week range given
        date.
    """

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
    output_file = "inventory_{}_W{:02d}_{:02d}.txt".format(year, calendar_week,
                                                   (calendar_week + 1))  # file to save the results
    output_fpath = os.path.join(inventories, output_file)

    with open(output_fpath, "w+") as file_object:
        # Iterate over the files in the tropomi_gta/pkl directory and append them
        # to the inventory
        fpath = os.path.join(tropomi_pkl, '*')

        for file in glob(fpath):
            # Find date of .nc file and convert to datetime object
            date = pd.to_datetime(file[-8:], format='%Y%m%d', errors='ignore')

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

######################


if __name__ == '__main__':
    # create inventory text file for every couple weeks between 2018 and 2020
    for i in range(2018, 2021):
        for j in range(1, 52, 2):
            
            start, end, calendar_week = get_files(year=i, calendar_week=j)
            
    # delete empty files
    inv_list = glob(os.path.join(inventories, '*'))
    for inv in inv_list:
        if os.path.getsize(inv) == 0:
            print('{} is empty and was deleted.'.format(inv))
            os.remove(inv)
        

    # get_files(year=2020, month=1)
