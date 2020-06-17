#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: calendar_functions.py 

Script containing functions:
    - get_odd_week_number (year, month, day)
    - get_start_and_end_date_from_calendar_week(year, calendar_week)
    - get_start_and_end_date_from_month(year, month)
"""

import time
import datetime
from datetime import timedelta
import pandas as pd
from glob import glob
import os
import sys
import textwrap
import calendar
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

def get_start_and_end_date_from_month(year, month):
    """
    Return datetime objects for the first day and last day of a given month 
    and year.
    
    Args:
        year (int): calendar year. 
        month (int): calendar month. Pre-condition: 1 <= month <= 12.
    
    Returns:
        first_day (datetime): date of first day of month.
        last_day (datetime): date of last day of month.
    """
    
    # Load the total number of days in month
    _, num_days = calendar.monthrange(year, month)
    
    first_day = datetime.date(year, month, 1)
    last_day = datetime.date(year, month, num_days)
    
    return first_day, last_day