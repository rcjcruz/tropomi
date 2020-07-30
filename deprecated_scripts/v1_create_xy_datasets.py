#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_xy_datasets.py

Script containing functions:
    - rotate(pivot, point, angle)
    - create_xy_dataset(ds, city='toronto')

Functions to create xarray.Dataset containing NO2 TVCD and eastward (u) and 
northward (v) wind components colocated to xy-coordinates with a city of 
interest as the origin.
"""
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from paths import *
import points_of_interest as poi
import convert as cv

import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

def create_xy_dataset(ds, city='toronto', rotate_coords=False):
    """
    Return dataset with NO2 and eastward (u) and northward (v) wind components
    with the corresponding xy-coordinates as variables. The origin of the 
    Cartesian coordinates is the city of interest.

    If rotate_coords is True, xy-coordinates are rotated such that the
    prevailing wind direction lies along the x-axis. 

    Args:
        ds (xr.Dataset): dataset containing NO2 and u-v wind components.
        city (str): city of interest.
        rotate (bool): rotate pixels so the wind direction is along the x-axis.
            Default: False.

    Returns:
        cartesian_ds (xr.Dataset): dataset containing no2, uv-wind components,
            and xy-coordinates as variables.
        rotated_ds (xr.Dataset): dataset containing no2 and rotated xy-
            coordinates.
    """

    # load no2, wind speed, bearing, lat/lon
    no2 = ds.no2.where(ds.no2 > 0, np.nan)
    u, v = ds.u, ds.v
    lat, lon = ds.no2.latitude, ds.no2.longitude

    # create empty xy-coordinates arrays
    y = np.zeros_like(lat)
    x = np.zeros_like(lon)
    x, y = np.meshgrid(x, y)

    # load origin (city of interest)
    lon0, lat0 = poi.cities[city]
    for i in range(len(lon)):
        for j in range(len(lat)):
            x[i][j], y[i][j] = cv.convert_to_cartesian(
                lat0, lon0, lat[j], lon[i])

    # create dataset with xy coordinates as a data variable
    cartesian_ds = xr.Dataset({'no2': no2,
                               'u': u,
                               'v': v,
                               'x_coords': xr.DataArray(data=x,
                                                        dims=['y', 'x'],
                                                        attrs={'units': 'km'}),
                               'y_coords': xr.DataArray(data=y,
                                                        dims=['y', 'x'],
                                                        attrs={'units': 'km'})},
                              attrs={'description': 'dataset for NO2 TVCD with \
                                  wind components and corresponding xy-coordinates \
                                          calculated using haversine formula.',
                                     'origin': city})

    if rotate_coords:
        # find average wind speed and direction over region
        # direction is calculated with x-axis as 0 degrees
        # (Cartesian angles, not bearing)
        avg_u, avg_v = np.average(u), np.average(v)
        direction = np.degrees(np.arctan2(avg_v, avg_u))
        if direction < 0:
            direction += 360

        rx = np.zeros_like(x, dtype=float)
        ry = np.zeros_like(y, dtype=float)

        for i in range(len(x)):
            for j in range(len(y)):
                rx[i][j], ry[i][j] = cv.rotate(pivot=(0, 0),
                                               point=(x[i][j], y[i][j]),
                                               angle=direction)

        cv.rotated_ds = xr.Dataset({'no2': no2,
                                    'x_coords': xr.DataArray(data=rx,
                                                             dims=['y', 'x'],
                                                             attrs={'units': 'km',
                                                                    'description': 'rotated x-coords'}),
                                    'y_coords': xr.DataArray(data=ry,
                                                             dims=['y', 'x'],
                                                             attrs={'units': 'km',
                                                                    'description': 'rotated y-coords.'})},
                                   attrs={'description': 'dataset for NO2 TVCD and uv-components of wind \
                                     rotated about origin (city).',
                                          'origin': city})
        return (cartesian_ds, rotated_ds)

    else:
        return cartesian_ds


if __name__ == '__main__':
    city = 'toronto'
    f = '202005'
    f_str = '/' + f + '*_avg'
    fpath = winds_pkl + city + f_str
    for file in glob.glob(fpath):
        infile = open(file, 'rb')
        ds = pickle.load(infile)
        infile.close()
        cartesian_ds, rotated_ds = create_xy_dataset(
            ds, 'toronto', rotate_coords=True)

        no2 = ds.no2.where(ds.no2 > 0, np.nan)
        u = ds.u
        v = ds.v
        lat = ds.no2.latitude
        lon = ds.no2.longitude

        avg_u, avg_v = np.average(u), np.average(v)
        avg_bear = np.degrees(np.arctan2(avg_u, avg_v))
        if avg_bear < 0:
            avg_bear += 360
        avg_ws = np.sqrt(avg_u ** 2 + avg_v ** 2)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        date_str = str(pd.to_datetime(ds.time.values)[0].date())
        ax1.text(0, 1.07,
                 r"NO$_2$ troposheric vertical column",
                 fontsize=18,
                 transform=ax1.transAxes)
        ax1.text(0, 1.02,
                 r"{}, {}".format(city, date_str),
                 fontsize=14,
                 transform=ax1.transAxes)
        im1 = ax1.pcolormesh(cartesian_ds.x_coords.values,
                             cartesian_ds.y_coords.values,
                             cartesian_ds.no2.values[0],
                             cmap='Blues')
        im2 = ax1.scatter(0, 0, color='r', marker='*')
        im3 = ax1.annotate('toronto', (0, 0))

        ax2 = fig.add_subplot(122)
        im4 = ax2.pcolormesh(rotated_ds.x_coords.values,
                             rotated_ds.y_coords.values,
                             rotated_ds.no2.values[0],
                             cmap='Blues')
        im5 = ax2.scatter(0, 0, color='r', marker='*')
        im6 = ax2.annotate('toronto', (0, 0))

        qv1 = ax1.quiver(0, 0, avg_u, avg_v, scale=100, color='k')
        qv2 = ax2.quiver(0, 0, avg_ws, 0, scale=100, color='k')

        plt.show()

    # fpath = winds_pkl + city + '/20200521_avg'
    # infile = open(fpath, 'rb')
    # ds = pickle.load(infile)
    # infile.close()
    # cartesian_ds, rotated_ds = create_xy_dataset(ds, 'toronto', rotate_coords=True)


# ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
# im1 = ds.no2.isel(time=0).plot.pcolormesh(ax=ax1,
#                                           transform=ccrs.PlateCarree(),
#                                           infer_intervals=True,
#                                           cmap='Blues',
#                                           robust=True,
#                                           x='longitude',
#                                           y='latitude',
#                                           add_colorbar=False)

# extent = 0.8
# city_coords = poi.cities[city]
# plot_limits = (city_coords.lon-extent,
#                 city_coords.lon+extent+0.05,
#                 city_coords.lat-extent,
#                 city_coords.lat+extent+0.05)
# ax1.set_extent(plot_limits)

# qv = plt.quiver(lon, lat, u[0, :, :], v[0, :, :], scale=400, color='k')
# gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                    linewidth=1, color='gray', alpha=0.5, linestyle=':')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# plt.show()

###################

# no2 = ds.no2.where(ds.no2 > 0, np.nan)
# u = ds.u
# v = ds.v
# lat = ds.no2.latitude
# lon = ds.no2.longitude

# avg_u, avg_v = np.average(u), np.average(v)
# avg_bear = np.degrees(np.arctan2(avg_u, avg_v))
# if avg_bear < 0:
#     avg_bear += 360
# avg_ws = np.sqrt(avg_u ** 2 + avg_v ** 2)

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# im1 = ax1.pcolormesh(cartesian_ds.x_coords.values,
#                      cartesian_ds.y_coords.values,
#                      cartesian_ds.no2.values[0],
#                      cmap='Blues')
# im2 = ax1.scatter(0, 0, color='r', marker='*')
# im3 = ax1.annotate('toronto', (0, 0))

# ax2 = fig.add_subplot(122)
# im4 = ax2.pcolormesh(rotated_ds.x_coords.values,
#                      rotated_ds.y_coords.values,
#                      rotated_ds.no2.values[0],
#                      cmap='Blues')
# im5 = ax2.scatter(0, 0, color='r', marker='*')
# im6 = ax2.annotate('toronto', (0, 0))


# qv1 = ax1.quiver(0, 0, avg_u, avg_v, scale=100, color='k')
# qv2 = ax2.quiver(0, 0, avg_ws, 0, scale=100, color='k')


# plt.show()
