#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: plot_tropomi.py

Script contains functions:
    - plot_tropomi(ds, res, plot_type='toronto'):

Script to grid TROPOMI data into a uniform lat/lon grid spanning from -180 to 180
longitude, -90 to 90 latitude.

Averaged values are found in val_arr_mean array
"""

# Preamble
import open_tropomi as ot
import grid_tropomi as gt
import points_of_interest as poi
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
import string
import calendar
from datetime import timedelta
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
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

# City locations
cities = {'toronto': 'Toronto, Canada',
          'montreal': 'Montreal, Canada',
          'vancouver': 'Vancouver, Canada',
          'new_york': 'New York, USA',
          'los_angeles': 'Los Angeles, USA'}


def load_ds(city, start_month, start_year,
            end_month, end_year):
    """
    Return a list of datasets for city from start to end.

    Args:
        city (str): City of interest. Valid cities include: 'toronto', 
            'montreal', 'vancouver', 'los_angeles', 'new_york'
        start_month (int): start month, 1 <= start_month <= 12
        start_year (int): start year
        end_month (int): end month, 1 <= start_month <= 12
        end_year (int): end year

    Returns:
        ds_list (list of xr.DataArray)
    """
    
    ds_list = []

    if city not in list(cities.keys()):
        return ValueError('Invalid city.')
    else:
        for year in range(start_year, end_year+1):
            for month in range(start_month, end_month+1):
                test_file = os.path.join(
                    tropomi_pkl_month, '{}/{}_M{:02d}'.format(city, year, month))
                infile = open(test_file, 'rb')
                ds = pickle.load(infile)
                ds_list.append(ds)
                infile.close()

    return sorted(ds_list)

############################

def get_resid(ds):
    """
    Args:
        ds (list of xr.DataArray): a list containing the data arrays used to 
            calculate the relative difference.

    Returns:
        avg_ds(xr.DataArray)
        res_ds(xr.DataArray)
    """

    start_year = ds[0].year
    end_year = ds[-1].year
    start_month_name = calendar.month_name[ds[0].month]
    end_month_name = calendar.month_name[ds[-1].month]

    # take sum of years leading up to most recent year then calculate mean
    sum_ds = []
    for d in ds:
        if d.year != end_year:
            sum_ds.append(d)
    
    yearsum = [sum(x) for x in zip(*sum_ds)]
    avg = yearsum[0] / (len(sum_ds))

    # load data array for most recent year in ds
    year_of_interest = ds[-1][0]

    # check if only looking at one month of range of months
    if start_month_name == end_month_name:
        month_name = start_month_name
    else:
        month_name = ''
        # create month name as first letter of each month (i.e. 'MAM')
        for i in range(ds[0].month, ds[-1].month+1):
            month_name += calendar.month_name[i][0]

    # If only looking at one month, make title of plot "Month Year",
    # else, make it "Avg Month Start_Year - End Year"
    if len(ds[:-1]) == 1:
        time_str = '{} {}'.format(month_name, start_year)
    else:
        if start_year == end_year-1:
            time_str = 'Avg. {} {}'.format(month_name, start_year)
        else:
            time_str = 'Avg. {} {}-{}'.format(month_name, start_year, end_year-1)

    # calculate relative difference and store in new_ds
    res_ds = xr.DataArray(np.array([(year_of_interest-avg) / avg]),
                          dims=('time', 'latitude', 'longitude'),
                          coords={'time': np.array(['Diff. {} {}-{}'.format(month_name, start_year, end_year)]),
                                  'latitude': ds[0].latitude,
                                  'longitude': ds[0].longitude},
                          attrs={'month': ds[0].month,
                                 'year': '{}-{}'.format(start_year, end_year),
                                 'title': 'Rel. Diff. {} {}-{}'.format(month_name, start_year, end_year)})

    avg_ds = xr.DataArray(np.array([avg]),
                          dims=('time', 'latitude', 'longitude'),
                          coords={'time': np.array([time_str]),
                                  'latitude': ds[0].latitude,
                                  'longitude': ds[0].longitude},
                          attrs={'month': ds[0].month,
                                 'year': '{}-{}'.format(start_year, end_year-1),
                                 'title': time_str})

    return avg_ds, res_ds

############################

def create_colorbar(im, ax, label, orientation='horizontal',
                    fmt_num=-5, mathbool=True, resid=False):
    """
    Return a colorbar to plot for TROPOMI NO2 TVCD.

    Args:
        im: the mathplotlib.cm.ScalarMappable described by this colorbar.
        ax (Axes): list of axes. 
        label (str): label for colorbar.
        orientation (str): vertical or horizontal.
        fmt_num (num): exponent for scientific notation.
        mathbool (bool): toggle mathText.
        resid (bool): Toggle residual difference. 

    Returns:
        cbar (colorbar)
    """

    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(
                self, useOffset=offset, useMathText=mathText)

        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = r'$\mathdefault{%s}$' % self.format

    if resid:
        asp = 20
    else:
        asp = 45

    cbar = plt.colorbar(mappable=im, ax=ax, orientation=orientation,
                        format=OOMFormatter(fmt_num, mathText=mathbool),
                        aspect=asp)
    cbar.ax.xaxis.get_offset_text().set_visible(False)
    cbar.set_label(
        r"%s ($10^{%s}$ mol/m$^2$)" % (label, fmt_num), labelpad=10, fontsize=14)

    return cbar

############################

def plot_tropomi(ds, plot_type='weekly', city='toronto'):
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

    # Load date and location
    start = str(ds.attrs['first day'].date())
    end = str(ds.attrs['last day'].date())
    year = ds.attrs['year']

    if plot_type == 'weekly':
        weeks = ds.attrs['weeks']
        date_str = 'weeks {}, {} to {}'.format(weeks, start, end)
    elif plot_type == 'monthly':
        month = ds.attrs['month']
        date_str = 'month {:02d}, {} to {}'.format(month, start, end)
    else:
        return ValueError('plot_type must be "monthly" or "weekly"')

    ax.text(0, 1.07,
            r"NO$_2$ troposheric vertical column",
            fontsize=18,
            transform=ax.transAxes)
    ax.text(0, 1.02,
            r"{}, {}".format(cities[city], date_str),
            fontsize=14,
            transform=ax.transAxes)

    # set map to plot within plot_limits
    extent = 1
    city_coords = poi.cities[city]
    plot_limits = (city_coords.lon-extent,
                   city_coords.lon+extent-0.05,
                   city_coords.lat-extent,
                   city_coords.lat+extent-0.05)
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

    # set 0 values to np.nan
    ds = ds.where(ds > 0, np.nan)

    # plot averaged values
    im = ds.isel(time=0).plot.pcolormesh(ax=ax,
                                         transform=ccrs.PlateCarree(),
                                         infer_intervals=True,
                                         cmap='Blues',
                                         #  vmin=10e-6,
                                         #  vmax=6e-5,
                                         robust=True,
                                         x='longitude',
                                         y='latitude',
                                         add_colorbar=False)
    colors = im.cmap(im.norm(im.get_array()))

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

    lakes_50m = cfeature.NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale='10m',
        facecolor='none')

    # pop = shpr.natural_earth(
    #     category='cultural',
    #     name='populated_places',
    #     resolution='10m')

    # xy = [pt.coords[0] for pt in shpr.Reader(pop).geometries()]
    # x, y = zip(*xy)
    # ax.scatter(x, y, transform=ccrs.Geodetic())

    ax.add_feature(countries, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(states_provinces, edgecolor='gray')
    # ax.add_feature(lakes_50m, edgecolor='blue')

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Show plot
    plt.show()

    # Save data to world_figures to toronto_figures with the time
    # of processing appended to the file name
    # ONLY runs if plot_tropomi.py is run directly
    if __name__ == '__main__':
        is_save = str(
            input('Do you want to save a png and KML of this plot? \n (Y/N)'))
        if is_save == 'Y' or is_save == 'y':
            if plot_type == 'weekly':
                print('Saving png for {}, weeks {}'.format(
                    ds.attrs['year'], ds.attrs['weeks']))
                pngfile = '{0}.png'.format(
                    '../figures/{}/{}_W{}'.format(city, year, weeks))

                fig.savefig(pngfile, dpi=300)
            elif plot_type == 'monthly':
                print('Saving png for {}, month {}'.format(
                    ds.attrs['year'], ds.attrs['month']))
                pngfile = '{0}.png'.format(
                    '../figures/{}/{}_M{:02d}'.format(city, year, month))

                fig.savefig(pngfile, dpi=300)

#############################

def plot_residual(ds, plot_type='weekly', city='toronto', diff=False):
    """
    Return a figure with three Cartopy plots of averaged TROPOMI data ds.
    First plot is 2019 data, second plot is 2020 data, third plot is residual.

    Args:
        ds (list of xr.DataArray): first dataset of TROPOMI NO2.
        plot_type (str): accepted values: 'weekly', 'monthly'. Default: 'weekly'
        city (str): city name. Valid cities: 'toronto', 'vancouver', 'montreal',
            'new_york', 'los_angeles'. Default: 'toronto'

    """

    start = ds[0].year
    end = ds[-1].year
    month = ds[0].month

    # Add title to attributes of each dataset in ds
    for d in ds:
        d.attrs['title'] = '{} {}'.format(calendar.month_name[month], d.year)

    # Get residual
    if diff:
        avg, new_ds = get_resid(ds)
        ds = [avg, ds[-1], new_ds]

    # Create figures
    col_nums = len(ds)

    fig, axes = plt.subplots(ncols=col_nums,
                             figsize=(col_nums*5, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    fig.tight_layout()

    # Accumulate gridlines and mappables. Load labels for eahc plot
    gls = []
    ims = []
    labels = list(string.ascii_uppercase)[:col_nums]

    for i, label in enumerate(labels):
        # Load date
        axes[i].text(0.05, 0.95, label, transform=axes[i].transAxes,
                     fontsize=16, fontweight='bold', va='top')

        # set map to plot within plot_limits
        plot_limits = poi.get_plot_limits(city=city, extent=1, res=-0.05)
        axes[i].set_extent(plot_limits, crs=ccrs.PlateCarree())

        # # Plot cities of interest
        marker = itertools.cycle(('black', 'blue', 'red', 'orange', 'green',
                                  'purple', 'gray'))
        city_cntr = 1
        for city_entry in poi.cities_coords.keys():
            city_name = city_entry[:-7]

            # Check if city lat/lon fall within boundary box
            w, e, s, n = plot_limits
            if ((w < poi.cities_coords[city_entry].lon) and
                (e > poi.cities_coords[city_entry].lon) and
                (s < poi.cities_coords[city_entry].lat) and
                    (n > poi.cities_coords[city_entry].lat)):

                color = next(marker)

                # Plot the city
                axes[i].plot(poi.cities_coords[city_entry].lon,
                             poi.cities_coords[city_entry].lat,
                             linestyle='None',
                             marker='*',
                             markeredgewidth=1,
                             markeredgecolor=color,
                             markerfacecolor=color,
                             markersize=5,
                             label=city_name)
                city_cntr += 1

        # Load settings for each plot type, depending if residual diff is toggled
        if diff and i == (len(labels) - 1):
            vmin = -1.0
            vmax = 1.0
            cmap = 'RdBu'
        else:
            vmin = np.amin(ds[:-1])
            vmax = np.amax(ds[:-1])
            cmap = 'Blues'

        # plot NO2 TVCD
        im = ds[i].isel(time=0).plot.pcolormesh(ax=axes[i],
                                                transform=ccrs.PlateCarree(),
                                                infer_intervals=True,
                                                cmap=cmap,
                                                robust=True,
                                                vmin=vmin,
                                                vmax=vmax,
                                                x='longitude',
                                                y='latitude',
                                                add_colorbar=False)
        ims.append(im)

        # # remove default title
        axes[i].set_title(ds[i].title)

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

        axes[i].add_feature(countries, edgecolor='black')
        axes[i].add_feature(cfeature.COASTLINE)
        axes[i].add_feature(states_provinces, edgecolor='gray')
        axes[i].add_feature(roads, edgecolor='gray')

        # Initialize gridlines
        gl = axes[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=1, color='gray', alpha=0.5,
                               linestyle=':')
        gls.append(gl)

        gls[i].xlabels_top = False
        gls[i].ylabels_right = False
        gl.xlocator = ticker.FixedLocator(np.arange(int(plot_limits[0])-1,
                                                    int(plot_limits[1])+1, 0.5))
        gl.ylocator = ticker.FixedLocator(np.arange(int(plot_limits[2])-1,
                                                    int(plot_limits[3])+1, 0.5))
        gls[i].xformatter = LONGITUDE_FORMATTER
        gls[i].yformatter = LATITUDE_FORMATTER

    # set colorbar properties
    if diff:
        cba = create_colorbar(
            im=ims[0], ax=axes[:-1], label='Mean NO$_2$ TVCD')
        cbb = plt.colorbar(
            mappable=ims[-1], ax=axes[-1], orientation='horizontal')
        cbb.set_label(r'Difference', labelpad=10, fontsize=14)

    else:
        cba = create_colorbar(im=ims[0], ax=axes[:], label='Mean NO$_2$ TVCD')

    # set legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=city_cntr)

    # Show plot
    plt.show()

    # Ask to save file
    if __name__ == '__main__':
        is_save = str(
            input('Do you want to save a png of this figure? \n (Y/N)'))
        if is_save == 'Y' or is_save == 'y':
            city_abbrs = {'toronto': 'TOR',
                          'montreal': 'MTL',
                          'vancouver': 'VAN',
                          'los_angeles': 'LAL',
                          'new_york': 'NYC'}

            if diff:
                prefix = 'diff'
            else:
                prefix = 'fig'
            print('Saving png for {} for {}-{}, month {}'.format(city, start, end, month))
            pngfile = '{0}.png'.format(
                '../figures/{}/{}_{}_{}_{}_M{:02d}'.format(city, city_abbrs[city], prefix, start, end,
                                                           month, dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))

            fig.savefig(pngfile, dpi=300)


if __name__ == '__main__':
    test_file = os.path.join(
        tropomi_pkl_week, 'toronto/2019_W11_12')
    infile = open(test_file, 'rb')
    ds = pickle.load(infile)
    infile.close()
    
    plot_tropomi(ds, 'weekly', 'toronto')
    
    # city = 'toronto'
    # ds_list = load_ds(city, start_month=5,
    #                   start_year=2019,
    #                   end_month=5,
    #                   end_year=2020)
    # avg_ds, res_ds = get_resid(ds_list)
    # plot_residual(ds_list, plot_type='monthly', city=city, diff=True)
