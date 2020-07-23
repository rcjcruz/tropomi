#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import pickle
import glob
import numpy as np
import xarray as xr
import time

from paths import *
import points_of_interest as poi
import convert as cv


def create_ds(date, city='toronto', rotate=False):
    """
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
    wind_speed = np.sqrt(avg_u ** 2 + avg_v ** 2)

    # load datasets
    lon = ds.longitude
    lat = ds.latitude
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
    sounding = ds.sounding
    corner = [0.0, 1.0, 2.0, 3.0]
    x = xr.DataArray(np.zeros_like(lon),
                     dims=['sounding'],
                     coords=[sounding], attrs={'long_name': 'pixel centre x-coordinate',
                                               'units': 'km'})
    y = xr.DataArray(np.zeros_like(lat),
                     dims=['sounding'],
                     coords=[sounding], attrs={'long_name': 'pixel centre y-coordinate',
                                               'units': 'km'})
    rx = xr.DataArray(np.zeros_like(lon),
                     dims=['sounding'],
                     coords=[sounding], attrs={'long_name': 'pixel centre of rotated x-coordinate',
                                               'units': 'km'})
    ry = xr.DataArray(np.zeros_like(lat),
                     dims=['sounding'],
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
    print('Total:', len(lat))
    for i in range(len(lat)):
        print('Converting scanline {} to Cartesian'.format(i))
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
    cartesian_ds = xr.Dataset({'no2': no2, 'x': x, 'y': y,
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
            print('Rotating scanline', j)
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
        rotated_ds = xr.Dataset({'no2': no2, 'x': rx, 'y': ry,
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
    f_str = os.path.join(winds_pkl, 'toronto/2019*raw')
    ls = glob.glob(f_str)
    for file in ls[21:]:
        date = file[-12:-4]
        cartesian_ds, rotated_ds = create_ds(date=date, city=city, rotate=True) 
