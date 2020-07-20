#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_xy_datasets.py

Script containing functions:
    - create_xy_dataset(ds, city='toronto')

Functions to create xarray.Dataset containing NO2 TVCD, wind speeds, and bearing
colocated to xy-coordinates with a city of interest as the origin.
"""
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from paths import *
import points_of_interest as poi
import convert as cv

import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


def rotate(pivot, point, angle):
    """
    Return xy-coordinate for an initial point (x, y) rotated by angle (in
    radians) around a pivot (x, y).

    Args:
        pivot (tuple of floats): pivot point of rotation.
        point (tuple of floats): xy-coordinates of point to be rotated.
        angle (float): angle to rotate the point around the pivot.
            Must be in radians.
    Returns:
        (xnew, ynew): xy-coordinates of rotated point.
    """
    # to rotate cw, need negative bearing
    s = np.sin(np.radians(-angle))
    c = np.cos(np.radians(-angle))

    # translate point back to origin
    x = point[0] - pivot[0]
    y = point[1] - pivot[1]

    # rotate clockwise to the x-axis
    xnew = x * c - y * s
    ynew = x * s + y * c

    # translate point back
    xnew += pivot[0]
    ynew += pivot[1]

    return (xnew, ynew)


def create_xy_dataset(ds, city='toronto', rotate_coords=False):
    """
    Return dataset with NO2, wind speed, and bearing with the corresponding
    xy-coordinates as variables. The origin of the Cartesian coordinates is
    the city of interest.

    Args:
        ds (xr.Dataset): dataset containing NO2, wind speed, bearing, latitude,
            and longitude.
        city (str): city of interest.
        rotate (bool): rotate pixels so the wind direction is along the x-axis.
            Default: False.

    Returns:
        new_ds (xr.Dataest): dataset containing ds and xy-coordinates as
            variables.
    """

    # load no2, wind speed, bearing, lat/lon
    no2 = ds.no2.where(ds.no2 > 0, np.nan)
    u = ds.u
    v = ds.v
    lat = ds.no2.latitude
    lon = ds.no2.longitude

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
        # find average wind speed and bearing over region
        avg_u, avg_v = np.average(u), np.average(v)
        avg_bear = np.degrees(np.arctan2(avg_v, avg_u))
        if avg_bear < 0:
            avg_bear += 360

        rx = np.zeros_like(x, dtype=float)
        ry = np.zeros_like(y, dtype=float)

        for i in range(len(x)):
            for j in range(len(y)):
                rx[i][j], ry[i][j] = rotate(pivot=(0, 0),
                                            point=(x[i][j], y[i][j]),
                                            angle=avg_bear)

        rotated_ds = xr.Dataset({'no2': no2,
                                 'x_coords': xr.DataArray(data=rx,
                                                          dims=['y', 'x'],
                                                          attrs={'units': 'km',
                                                                 'description': 'rotated x-coords'}),
                                 'y_coords': xr.DataArray(data=y,
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
    fpath = winds_pkl + city + '/20200503_avg'
    infile = open(fpath, 'rb')
    ds = pickle.load(infile)
    infile.close()
    cartesian_ds = create_xy_dataset(ds, 'toronto', rotate_coords=False)

fig = plt.figure()


no2 = ds.no2.where(ds.no2 > 0, np.nan)
u = ds.u
v = ds.v
lat = ds.no2.latitude
lon = ds.no2.longitude


ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
im1 = ds.no2.isel(time=0).plot.pcolormesh(ax=ax1,
                                          transform=ccrs.PlateCarree(),
                                          infer_intervals=True,
                                          cmap='Blues',
                                          robust=True,
                                          x='longitude',
                                          y='latitude',
                                          add_colorbar=False)

extent = 0.8
city_coords = poi.cities[city]
plot_limits = (city_coords.lon-extent,
                city_coords.lon+extent+0.05,
                city_coords.lat-extent,
                city_coords.lat+extent+0.05)
ax1.set_extent(plot_limits)

qv = plt.quiver(lon, lat, u[0, :, :], v[0, :, :], scale=400, color='k')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   linewidth=1, color='gray', alpha=0.5, linestyle=':')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()

###################

# ws = ds.wind_speed
# bear = ds.bearing
# u, v = -ws*np.sin(np.radians(bear)), -ws*np.cos(np.radians(bear))
# avg_u, avg_v = np.average(u), np.average(v)
# avg_bear = np.degrees(np.arctan2(avg_v, avg_u))
# if avg_bear < 0:
#     avg_bear += 360
# avg_ws = np.sqrt(avg_u ** 2 + avg_v ** 2)

# ax1=fig.add_subplot(121)
# im1=ax1.pcolormesh(cartesian_ds.x_coords.values,
#                      cartesian_ds.y_coords.values,
#                      cartesian_ds.no2.values[0],
#                      cmap='Blues')
# im2=ax1.scatter(0, 0, color='r', marker='*')
# im3=ax1.annotate('toronto', (0, 0))

# qv = plt.quiver(0, 0, avg_u, avg_v, scale=10, color='k')
# qv = plt.quiver(lon, lat, u[0, :, :],
#                 v[0, :, :], scale=400, color='k')


# ax2=fig.add_subplot(122)
# im4=ax2.pcolormesh(rotated_ds.x_coords.values,
#                      rotated_ds.y_coords.values,
#                      rotated_ds.no2.values[0],
#                      cmap='Blues')
# im5=ax2.scatter(0, 0, color='r', marker='*')
# im6=ax2.annotate('toronto', (0, 0))
# plt.show()
