#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cartesian_rotate.py

Functions:
    create_ds(date, city='toronto', rotate=True)

Script to load TROPOMI NO2 with wind components included from a date and convert
lat/lon of pixel centre and pixel bounds to Cartesian (i.e. distance from city).
If rotate is True, Cartesian coordinates are rotated such that the average
wind direction for the day is along the x-axis.
"""

import pickle
import glob
import numpy as np
import xarray as xr
import time

from paths import *
import points_of_interest as poi
import convert as cv


def create_ds(date, city='toronto', rotate=True):
    """
    Return dataset with lat/lon pixel centre and bounds are converted to
    Cartesian coordinates as the distance from city. If rotate is True,
    all coordinates are rotated so that the average wind direction for the day
    is along the x-axis. Datasets are pickled into
    /export/data/scratch/tropomi_rc/cartesian or rotated

    Args:
        date (str): date of observation. Format: YYYYMMDD
        city (str): city of interest.
        rotate (bool): to determine whether to rotate the pixels. Default: False.

    Returns:
        cartesian_ds (xr.Dataset): dataset with Cartesian-transformed NO2
            pixels.
        rotated_ds (xr.Dataset):
    """
    start_time = time.time()

    print('------------------------------------------------------------------')
    print('... {} ...'.format(date))
    if rotate:
        print('PROCESS: convert and rotate.')
    else:
        print('PROCESS: convert.')
    # load dataset of date with winds
    fpath = city + '/' + date + '_raw'
    file = os.path.join(winds_pkl, fpath)
    infile = open(file, 'rb')
    ds = pickle.load(infile)
    infile.close()

    # calculate average angle and wind speed for date
    avg_u = np.average(ds.u)
    avg_v = np.average(ds.v)
    angle = np.degrees(np.arctan2(avg_v, avg_u))
    if angle < 0:
        angle += 360
    wind_speed = np.sqrt(avg_u ** 2 + avg_v ** 2)

    # load datasets
    lon = ds.longitude.values
    lat = ds.latitude.values
    lon0 = ds.longitude_bounds[0]
    lon1 = ds.longitude_bounds[1]
    lon2 = ds.longitude_bounds[2]
    lon3 = ds.longitude_bounds[3]
    lat0 = ds.latitude_bounds[0]
    lat1 = ds.latitude_bounds[1]
    lat2 = ds.latitude_bounds[2]
    lat3 = ds.latitude_bounds[3]

    # create dataarrays for latitude and longitude
    no2 = ds.nitrogendioxide_tropospheric_column
    no2_error = ds.nitrogendioxide_tropospheric_column_precision
    sounding = ds.sounding
    corner = [0.0, 1.0, 2.0, 3.0]
    x = xr.DataArray(np.zeros_like(lon), dims=['sounding'],
                     coords=[sounding], attrs={'long_name': 'pixel centre x-coordinate',
                                               'units': 'km'})
    y = xr.DataArray(np.zeros_like(lat), dims=['sounding'],
                     coords=[sounding], attrs={'long_name': 'pixel centre y-coordinate',
                                               'units': 'km'})
    rx = xr.DataArray(np.zeros_like(lon), dims=['sounding'],
                      coords=[sounding], attrs={'long_name': 'pixel centre of rotated x-coordinate',
                                                'units': 'km'})
    ry = xr.DataArray(np.zeros_like(lat), dims=['sounding'],
                      coords=[sounding], attrs={'long_name': 'pixel centre of rotated y-coordinate',
                                                'units': 'km'})
    x0 = xr.DataArray(np.zeros_like(lon0), dims=[
                      'sounding'], coords=[sounding])
    x1 = xr.DataArray(np.zeros_like(lon1), dims=[
                      'sounding'], coords=[sounding])
    x2 = xr.DataArray(np.zeros_like(lon2), dims=[
                      'sounding'], coords=[sounding])
    x3 = xr.DataArray(np.zeros_like(lon3), dims=[
                      'sounding'], coords=[sounding])
    y0 = xr.DataArray(np.zeros_like(lat0), dims=[
                      'sounding'], coords=[sounding])
    y1 = xr.DataArray(np.zeros_like(lat1), dims=[
                      'sounding'], coords=[sounding])
    y2 = xr.DataArray(np.zeros_like(lat2), dims=[
                      'sounding'], coords=[sounding])
    y3 = xr.DataArray(np.zeros_like(lat3), dims=[
                      'sounding'], coords=[sounding])

    # # load city as origin
    lon_city, lat_city = poi.cities[city]
    city_coords = (lat_city, lon_city)

    # convert each xy-coordinate pair to cartesian
    print('TOTAL:', len(lat))
    for i in range(len(lat)):
        print('[{}] Converting scanline {} to Cartesian'.format(date, i))
        # convert centre lat/lon to cartesian
        x[i], y[i] = cv.convert_to_cartesian(city_coords, (lat[i], lon[i]))

        # convert corner points to cartesian
        x0[i], y0[i] = cv.convert_to_cartesian(city_coords, (lat0[i], lon0[i]))
        x1[i], y1[i] = cv.convert_to_cartesian(city_coords, (lat1[i], lon1[i]))
        x2[i], y2[i] = cv.convert_to_cartesian(city_coords, (lat2[i], lon2[i]))
        x3[i], y3[i] = cv.convert_to_cartesian(city_coords, (lat3[i], lon3[i]))

    x_corners = xr.DataArray([x0, x1, x2, x3],
                             dims=['corner', 'sounding'],
                             coords=[corner, sounding],
                             attrs={'description': 'x-coordinates of corners',
                                    'units': 'km'})
    y_corners = xr.DataArray([y0, y1, y2, y3],
                             dims=['corner', 'sounding'],
                             coords=[corner, sounding],
                             attrs={'description': 'y-coordinates of corners',
                                    'units': 'km'})

    # create dataset of cartesian coordinates
    cartesian_ds = xr.Dataset({'no2': no2, 'no2_error': no2_error, 'x': x, 'y': y,
                               'x_bounds': x_corners, 'y_bounds': y_corners},
                              attrs={
        'description': 'dataset of Cartesian coordinates around city of interest.',
        'origin': city, 'average direction': angle, 'average wind speed': wind_speed})

    # pickle cartesian dataset
    output_file = '{}/{}'.format(city, date)
    pkl_path = os.path.join(cartesian_pkl, output_file)
    print('Pickling', pkl_path)
    with open(pkl_path, 'wb') as outfile:
        pickle.dump(cartesian_ds, outfile)

    # rotate all cartesian points by the average angle so that the average
    # wind direction is now along the x-axis
    if rotate:
        origin = (0, 0)
        for j in range(len(x)):
            print('[{}] Rotating scanline {}'.format(date, j))
            # convert centre lat/lon to cartesian
            rx[j], ry[j] = cv.rotate(origin, (x[j], y[j]), angle)

            # convert corner points
            x0[j], y0[j] = cv.rotate(origin, (x0[j], y0[j]), angle)
            x1[j], y1[j] = cv.rotate(origin, (x1[j], y1[j]), angle)
            x2[j], y2[j] = cv.rotate(origin, (x2[j], y2[j]), angle)
            x3[j], y3[j] = cv.rotate(origin, (x3[j], y3[j]), angle)

        # create DataArray of rotated corners
        x_corners = xr.DataArray([x0, x1, x2, x3],
                                 dims=['corner', 'sounding'],
                                 coords=[corner, sounding],
                                 attrs={'description': 'x-coordinates of corners',
                                        'units': 'km'})
        y_corners = xr.DataArray([y0, y1, y2, y3],
                                 dims=['corner', 'sounding'],
                                 coords=[corner, sounding],
                                 attrs={'description': 'y-coordinates of corners',
                                        'units': 'km'})

        # create Dataset of rotated Cartesian coordinates
        rotated_ds = xr.Dataset({'no2': no2, 'no2_error': no2_error, 'x': rx, 'y': ry,
                                 'x_bounds': x_corners, 'y_bounds': y_corners},
                                attrs={
                                    'description': 'dataset of rotated Cartesian coordinates around city of interest.',
                                    'origin': city, 'average direction': angle,
                                    'average wind speed': wind_speed})

        # pickle rotated_ds
        output_file = '{}/{}'.format(city, date)
        pkl_path = os.path.join(rotated_pkl, output_file)
        print('Pickling', pkl_path)
        with open(pkl_path, 'wb') as outfile:
            pickle.dump(rotated_ds, outfile)

    # Print processing time
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

    if rotate:
        return (cartesian_ds, rotated_ds)

    else:
        return cartesian_ds


if __name__ == '__main__':
    city = 'toronto'
    ls_202005 = ['20200501', '20200502', '20200503', '20200504', '20200505',
                 '20200506', '20200507', '20200508', '20200509', '20200511',
                 '20200512', '20200513', '20200515', '20200516', '20200517',
                 '20200519', '20200520', '20200521', '20200522', '20200523',
                 '20200524', '20200525', '20200526', '20200527', '20200529',
                 '20200530', '20200531'] # done
    ls_201905 = ['20190502', '20190505', '20190506', '20190507', '20190508',
                 '20190509', '20190510', '20190511', '20190514', '20190515',
                 '20190516', '20190517', '20190518', '20190519', '20190520',
                 '20190521', '20190523', '20190524', '20190525', '20190526',
                 '20190527', '20190529', '20190530', '20190531'] # done
    ls_201805 = ['20180501', '20180502', '20180504', '20180505', '20180506',
                 '20180507', '20180508', '20180509', '20180510', '20180511',
                 '20180512', '20180513', '20180514', '20180515', '20180516',
                 '20180517', '20180518', '20180519', '20180520', '20180521',
                 '20180523', '20180525', '20180526', '20180527', '20180528',
                 '20180529', '20180530', '20180531'] # done
    ls_202006 = ['20200601', '20200602', '20200603', '20200604', '20200605',
                 '20200606', '20200607', '20200608', '20200609', '20200610',
                 '20200611', '20200612', '20200613', '20200614', '20200615',
                 '20200616', '20200617', '20200618', '20200619', '20200620',
                 '20200621', '20200622', '20200623', '20200624', '20200625',
                 '20200626', '20200627', '20200628', '20200629', '20200630'] # done
    ls_201906 = ['20190601', '20190602', '20190603', '20190604', '20190606',
                 '20190607', '20190608', '20190609', '20190611', '20190612',
                 '20190613', '20190614', '20190617', '20190618', '20190619',
                 '20190620', '20190621', '20190622', '20190623', '20190624',
                 '20190625', '20190626', '20190627', '20190628', '20190629'] # done
    ls_201903 = ['20190301', '20190303', '20190304', '20190305', '20190306',
                 '20190307', '20190308', '20190309', '20190311', '20190312',
                 '20190313', '20190316', '20190317', '20190318', '20190319',
                 '20190320', '20190321', '20190322', '20190323', '20190324',
                 '20190325', '20190326', '20190327', '20190329', '20190331'] # done
    ls_202003 = ['20200301', '20200302', '20200304', '20200305', '20200306',
                 '20200307', '20200308', '20200309', '20200310', '20200311',
                 '20200312', '20200313', '20200315', '20200316', '20200317',
                 '20200320', '20200321', '20200322', '20200324', '20200325',
                 '20200327', '20200329', '20200331'] # done
    ls_201904 = ['20190401', '20190403', '20190404', '20190406', '20190407',
                 '20190408', '20190409', '20190410', '20190413', '20190415',
                 '20190416', '20190417', '20190421', '20190422', '20190423',
                 '20190424', '20190425', '20190427', '20190428', '20190430'] # done
    ls_202004 = ['20200401', '20200402', '20200403', '20200404', '20200405',
                 '20200406', '20200407', '20200408', '20200409', '20200410',
                 '20200411', '20200412', '20200414', '20200415', '20200416',
                 '20200417', '20200418', '20200419', '20200420', '20200421',
                 '20200422', '20200423', '20200424', '20200425', '20200427',
                 '20200428', '20200430'] # done

    # # to search
    # f_str = os.path.join(winds_pkl, 'toronto/202004*raw')
    # ls = glob.glob(f_str)
    # new_ls = []

    # for file in ls:
    #     date = file[-12:-4]
    #     new_ls.append(date)
    # 
    # new_ls = np.sort(new_ls)
        
    # to convert from list
    i=26
    for date in ls_202004[i:i+2]:
        cartesian_ds, rotated_ds = create_ds(date=date, city=city, rotate=True)

    # individual
    # date = '20190508'
    # cartesian_ds, rotated_ds = create_ds(date=date, city=city, rotate=True)
