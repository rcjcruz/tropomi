#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: get_files.py

Script containing functions:
    - get_odd_week_number (year, month, day)
    - get_start_and_end_date_from_calendar_week(year, calendar_week)
    - get_files(**kwargs)
    
Get files creates text file two_week_inventory.txt for all .nc files which fall
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

######################


def get_odd_week_number(year, month, day):
    """
    Get the nearest odd week number for a given date. If week number is odd,
    return week number. If week number if even, return the floor odd week number.

    Args:
        year (int): year of interest
        month (int): month of interest; 1 <= month <= 12
        day (int): day of interest

    Returns:
        calendar_week (int): week number for nearest odd week number for the date.
    """
    date = str(year) + '-' + str(month) + '-' + str(day)
    date = pd.to_datetime(date, format='%Y-%m-%d', errors='ignore')
    d = date.week
    if d % 2 == 0:
        d -= 1
    return d

######################


def get_start_and_end_date_from_calendar_week(year, calendar_week):
    """
    Return start and end date of two week period given year and calendar_week.

    Args:
        year (int): year of interest
        calendar_week (int): calendar week of interest; must be odd. If unsure
            of calendar_week, use get_odd_week_number to obtain it.
            calendar_week must be odd.

    Returns:
        monday (datetime): date of the monday (start date) of the two week period
        sunday (datetime): date of the second sunday (end date) of the two week
            period

    """

    if calendar_week % 2 == 0:
        return ValueError('Calendar week must be odd.')

    # If Monday of week 1 falls between Jan 5 to Jan 7, subtract 1 from week_number
    # because the Epoch calendar is weird
    week1_start = pd.to_datetime(str(str(year) + '0105'), format='%Y%m%d')
    week1_end = pd.to_datetime(str(str(year) + '0107'), format='%Y%m%d')
    week1_monday = datetime.datetime.strptime(f'{year}-1-1', '%Y-%W-%w').date()

    if week1_start <= week1_monday <= week1_end:
        calendar_week -= 1
        monday = datetime.datetime.strptime(
            f'{year}-{calendar_week}-1', '%Y-%W-%w').date()

    else:
        # Load Monday of week number
        monday = datetime.datetime.strptime(
            f'{str(year)}-{str(calendar_week)}-1', '%Y-%W-%w').date()

    return monday, monday + datetime.timedelta(days=13.9)

##################


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
        calendar_week = get_odd_week_number(year, month, day)

        # Load start and end date of two week range
        start, end = get_start_and_end_date_from_calendar_week(
            year, calendar_week)

    # year, calendar_week format
    elif ('year' in kwargs) and ('calendar_week' in kwargs):
        year, calendar_week = kwargs['year'], kwargs['calendar_week']

        if calendar_week % 2 == 0:
            print('--- Calendar week must be odd. '
                  'Subtracted 1 from calendar week and continued processing. ---')
            calendar_week -= 1

        start, end = get_start_and_end_date_from_calendar_week(
            year, calendar_week)

    # raise error if not a valid kwargs combination
    else:
        return ValueError(textwrap.fill(
            textwrap.dedent("""Invalid kwargs. Valid kwargs combinations:
                            - year, month, day
                            - year, calendar_week""")))

    # Naming the inventory .txt file
    output_file = "inventory_{}_W{}_{}.txt".format(year, calendar_week,
                                                   (calendar_week + 1))  # file to save the results
    output_fpath = os.path.join(inventories, output_file)

    # If file exists, remove it, otherwise, create a new file
    try:
        os.remove(output_fpath)
    except OSError:
        pass

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
    print('Created {} containing .nc files '
          'for weeks {} & {}, {} to {}'.format(output_file, calendar_week,
                                               (calendar_week + 1), start, end))

    return start, end, calendar_week

######################


if __name__ == '__main__':
    start, end, calendar_week = get_files(year=2020, calendar_week=18)

    # get_files(year=2020, month=1)
