import points_of_interest as poi
import add_wind_and_grid as aw
from paths import *
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


def plot_tropomi(ds, city='toronto', wind=False):
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
    lat = ds.no2.latitude
    lon = ds.no2.longitude
    u = ds.u
    v = ds.v

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

    # Plot winds if wind is True
    if wind:
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
# add_wind(f, city)

good_dates = ['20180501_avg', '20180502_avg', '20180504_avg', '20180505_avg',
              '20180507_avg', '20180508_avg', '20180509_avg', '20180513_avg',
              '20180516_avg', '20180517_avg', '20180518_avg', '20180520_avg',
              '20180521_avg', '20180523_avg', '20180525_avg', '20180527_avg',
              '20180528_avg', '20180529_avg', '20180530_avg', '20180531_avg',
              '20190312_avg', '20190317_avg', '20190318_avg', '20190319_avg',
              '20190320_avg', '20190323_avg', '20190325_avg', '20190326_avg',
              '20190327_avg', '20190329_avg', '20190401_avg', '20190403_avg',
              '20190404_avg', '20190406_avg', '20190410_avg', '20190413_avg',
              '20190415_avg', '20190417_avg', '20190422_avg', '20190424_avg',
              '20190425_avg', '20190428_avg', '20190505_avg', '20190506_avg',
              '20190507_avg', '20190508_avg', '20190511_avg', '20190517_avg',
              '20190519_avg', '20190521_avg', '20190523_avg', '20190524_avg',
              '20190525_avg', '20190526_avg', '20190527_avg', '20190531_avg',
              '20190602_avg', '20190603_avg', '20190606_avg', '20190607_avg',
              '20190608_avg', '20190609_avg', '20190611_avg', '20190612_avg',
              '20190614_avg', '20190617_avg', '20190618_avg', '20190621_avg',
              '20190622_avg', '20190623_avg', '20190624_avg', '20190625_avg',
              '20190626_avg', '20190627_avg', '20190628_avg', '20190629_avg',
              '20200301_avg', '20200305_avg', '20200307_avg', '20200308_avg',
              '20200309_avg', '20200315_avg', '20200316_avg', '20200321_avg',
              '20200322_avg', '20200325_avg', '20200327_avg', '20200329_avg',
              '20200401_avg', '20200402_avg', '20200405_avg', '20200406_avg',
              '20200408_avg', '20200409_avg', '20200410_avg', '20200411_avg',
              '20200412_avg', '20200418_avg', '20200420_avg', '20200422_avg',
              '20200423_avg', '20200425_avg', '20200427_avg', '20200428_avg',
              '20200501_avg', '20200503_avg', '20200504_avg', '20200505_avg',
              '20200506_avg', '20200507_avg', '20200513_avg', '20200516_avg',
              '20200519_avg', '20200520_avg', '20200521_avg', '20200522_avg',
              '20200523_avg', '20200524_avg', '20200525_avg', '20200526_avg',
              '20200531_avg', '20200601_avg', '20200602_avg', '20200603_avg',
              '20200604_avg', '20200605_avg', '20200606_avg', '20200607_avg',
              '20200608_avg', '20200609_avg', '20200610_avg', '20200611_avg',
              '20200612_avg', '20200613_avg', '20200614_avg', '20200615_avg',
              '20200616_avg', '20200617_avg', '20200618_avg', '20200619_avg',
              '20200620_avg', '20200621_avg', '20200622_avg', '20200624_avg',
              '20200625_avg', '20200626_avg', '20200627_avg', '20200628_avg',
              '20200629_avg', '20200630_avg']

for date in good_dates[84:]:
#     f_str = '/' + str(date) + '_avg'
    fpath = winds_pkl + city + '/' + date
    infile = open(fpath, 'rb')
    ds = pickle.load(infile)
    infile.close()
    is_plot = input('[Y/N] Do you want to plot {}?'.format(date))
    if is_plot == 'Y' or 'y':
        plot_tropomi(ds, wind=True)

# f_str = '/' + str(20200629) + '_avg'
# fpath = winds_pkl + city + f_str
# infile = open(fpath, 'rb')
# ds = pickle.load(infile)
# infile.close()

# plot_tropomi(ds, wind=True)
