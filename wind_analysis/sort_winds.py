#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sort_winds.py

Functions:
    sort_winds(tf, city='toronto', return_dates=False)

Return a dictionary for a time frame for a city where the dictionary keys are
wind speed ranges (i.e. 'A' = 0-2m/s, 'B' = 2-4m/s, etc). 
"""


import xarray as xr
import numpy as np
import os
import glob
import pickle
import pandas as pd 
import calendar 

import points_of_interest as poi
from paths import *


def sort_winds(tf, city='toronto', return_dates=False):
    """
    Return dictionary with datasets from a given timeframe tf over a city
    where datasets are aggregated by wind speeds.

    Args:
        tf (str): timeframe of interest. Accepted values:
            'may_1819', 'may_20', 'march_19', 'march_20', 
            'april_19', 'april_20', 'june_19', 'june_20', 
            'pre-vid', 'covid'
        city (str): city of interest. Default: 'toronto'
        return_dates (bool): Bool to determine if the start and end dates 
            of observations will be returned. Default: True.

    Returns:
        ws_dict (dict): dictionary of datasets organized by wind speeds.
        first_day (pd.datetime): first day of observations. Returned if
            return_dates is True.
        last_dat (pd.datetime): last day of observations. Returned if 
            return_dates is True.
    """

    # dictionary of wind types and possible date aggregations
    wind_type = poi.wind_type

    dates_dict = {'may_1819': ['201805*', '201905*'], 'may_20': ['202005*'],
                    'march_19': ['201903*'], 'march_20': ['202003*'],
                    'april_19': ['201904*'], 'april_20': ['202004*'],
                    'june_19': ['201906*'], 'june_20': ['202006*'],
                    'pre-vid': ['2019*'], 'covid': ['2020*']}

    # create dict to store wind datasets with same names as wind_type
    ws_dict = {}
    for wt in list(wind_type.keys()):
        ws_dict[wt] = []

    # Successfully opens all the files in the timeframe
    tframe = dates_dict[tf]
    for time in tframe:
        f_str = city + '/' + time
        fpath = os.path.join(rotated_pkl, f_str)
        print('---------------------------------------------------------------')
        print(fpath)

        for file in np.sort(glob.glob(fpath)):
            with open(file, 'rb') as infile:
                ds = pickle.load(infile)
                print('opened', file)
                # load wind speed
                ws = ds.attrs['average wind speed']

                # determine the wind_type for ds
                for wt in list(wind_type.keys()):
                    if ws >= wind_type[wt][0] and ws <= wind_type[wt][1]:
                        ws_dict[wt].append(ds)
                        print('{} belongs in wind type {}'.format(fpath, wt))
                    
    # formatting stand and end dates
    if return_dates:
        if tf == 'pre-vid':
            first_day = pd.to_datetime('20190301', format='%Y%m%d')
            last_day = pd.to_datetime('20190630', format='%Y%m%d')
        elif tf == 'covid':
            first_day = pd.to_datetime('20200301', format='%Y%m%d')
            last_day = pd.to_datetime('20200630', format='%Y%m%d')
        else:
            first_day = pd.to_datetime(tframe[0][:-1], format='%Y%m')
            days_num = calendar.monthrange(int(tframe[-1][:4]), int(tframe[-1][4:6]))[1]
            last_day = pd.to_datetime((tframe[-1][:-1]+str(days_num)), format='%Y%m%d')
        return (ws_dict, first_day, last_day)
    
    else:
        return ws_dict


if __name__ == '__main__':
    ws_dict = sort_winds('covid', 'toronto')
