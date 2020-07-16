import os
import glob
import pickle
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import time
import itertools
import datetime
from datetime import timedelta
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.ticker import LogFormatter
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.colorbar import colorbar

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

from paths import *
import points_of_interest as poi 
import add_wind as aw


def plot_tropomi(ds, city='toronto', **kwargs):
    """
    Return a Cartopy plot of averaged TROPOMI data ds over a given city.
    Aggregated data type is supplied to plot_type.

    Args:
        ds (xr.DataArray): TROPOMI tropospheric NO2 dataset.
        plot_type (str): accepted values: 'weekly', 'monthly'. Default: 'weekly'
        city (str): city name. Valid cities: 'toronto', 'vancouver', 'montreal',
            'new_york', 'los_angeles'. Default: 'toronto'

    >>> test_file = os.path.join(tropomi_pkl_month, 'toronto/2019_M03')
    >>> infile = open(test_file, 'rb')
    >>> ds = pickle.load(infile)
    >>> infile.close()
    >>> plot_tropomi(ds)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.tight_layout
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # load no2 and wind components
    no2 = ds.no2.where(ds.no2 > 0, np.nan)
    ws = ds.wind_speed
    bear = ds.bearing
    lat = ds.no2.latitude
    lon = ds.no2.longitude
    u = -ws*np.sin(np.radians(bear))
    v = -ws*np.cos(np.radians(bear))

    # Load date and location
    date_str = str(pd.to_datetime(ds.time.values)[0].date())
    ax.text(0, 1.07,
            r"NO$_2$ troposheric vertical column",
            fontsize=18,
            transform=ax.transAxes)
    ax.text(0, 1.02,
            r"{}, {}".format(city, date_str),
            fontsize=14,
            transform=ax.transAxes)

    # set map to plot within plot_limits
    extent = 0.8
    city_coords = poi.cities[city]
    plot_limits = (city_coords.lon-extent,
                   city_coords.lon+extent+0.05,
                   city_coords.lat-extent,
                   city_coords.lat+extent+0.05)
    ax.set_extent(plot_limits)

    # Plot cities of interest
    for city_entry in poi.cities_coords.keys():
        city_name = city_entry[:-7]
        ax.plot(poi.cities_coords[city_entry].lon,
                poi.cities_coords[city_entry].lat,
                marker='*',
                markeredgewidth=2,
                markeredgecolor='black',
                markerfacecolor='black',
                markersize=5)
        ax.text(poi.cities_coords[city_entry].lon + 0.05,
                poi.cities_coords[city_entry].lat + 0.05,
                city_name,
                bbox=dict(facecolor='wheat',
                          edgecolor='black', boxstyle='round'),
                transform=ccrs.Geodetic())


    vmin = np.amin(no2)
    vmax = np.amax(no2)

    # plot averaged values
    im = no2.isel(time=0).plot.pcolormesh(ax=ax,
                                         transform=ccrs.PlateCarree(),
                                         infer_intervals=True,
                                         cmap='Blues',
                                         vmin=vmin,
                                         vmax=vmax,
                                         robust=True,
                                         x='longitude',
                                         y='latitude',
                                         add_colorbar=False)

    # Plot winds
    qv = plt.quiver(lon, lat, u[0, :, :],
                    v[0, :, :], scale=400, color='k')
    qk = plt.quiverkey(qv, 0.8, 0.9, 10, r'10 $\frac{m}{s}$',
                       labelpos='E', coordinates='figure')

    # remove default title
    ax.set_title('')

    # set colorbar properties
    cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label(r"NO$_2$ TVCD (mol/m$^2$)", labelpad=30, fontsize=14)
    cbar.outline.set_visible(False)

    # Define Natural Earth features
    countries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='10m',
        facecolor='none')

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    
    roads = cfeature.NaturalEarthFeature(
        category='cultural',
        name='roads',
        scale='10m',
        facecolor='none')

    ax.add_feature(countries, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(states_provinces, edgecolor='gray')
    ax.add_feature(roads, edgecolor='gray')

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Show plot
    plt.show()
    
################################

city = 'toronto'
f = '20200520'
# add_wind(f, city)

fpath = winds_pkl + city + '/20200522_avg'
infile = open(fpath, 'rb')
ds = pickle.load(infile)
infile.close()

plot_tropomi(ds)