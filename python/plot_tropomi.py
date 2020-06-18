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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

# City locations
cities = {'toronto': 'Toronto, Canada',
          'montreal': 'Montreal, Canada',
          'vancouver': 'Vancouver, Canada',
          'new_york': 'New York, USA',
          'los_angeles': 'Los Angeles, USA'}


def get_resid(ds1, ds2):

    ds1 = ds1[0]
    ds2 = ds2[0]

    new_ds = xr.DataArray(np.array([(ds2-ds1) / ds1]),
                          dims=('time', 'latitude', 'longitude'),
                          coords={'time': np.array([ds1.month]),
                                  'latitude': ds1.latitude,
                                  'longitude': ds1.longitude},
                          attrs={'month': 5,
                                 'year': '2019-2020'})

    return new_ds


def create_colorbar(im, ax, label, orientation='horizontal',
                    fmt_num=-5, mathbool=True, resid=False):
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


def load_ds(city, start, end):
    ds_list = []

    if city not in list(cities.keys()):
        return ValueError('Invalid city.')
    else:
        for year in range(start, end+1):
            test_file = os.path.join(
                tropomi_pkl_month, '{}/{}_M05'.format(city, year))
            infile = open(test_file, 'rb')
            ds = pickle.load(infile)
            ds_list.append(ds)
            infile.close()

    return ds_list

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
    for city in poi.cities_coords.keys():
        city_name = city[:-7]
        ax.plot(poi.cities_coords[city].lon,
                poi.cities_coords[city].lat,
                marker='*',
                markeredgewidth=2,
                markeredgecolor='black',
                markerfacecolor='black',
                markersize=5)
        ax.text(poi.cities_coords[city].lon + 0.05,
                poi.cities_coords[city].lat + 0.05,
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
                                         cmap='viridis',
                                         #  vmin=10e-6,
                                         #  vmax=6e-5,
                                         norm=LogNorm(),
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
    cbar.set_label(r"$\log$(NO$_2)$ (mol/m$^2$)", labelpad=30, fontsize=14)
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
                    '../{}_figures/{}_W{:02d}_{:02d}'.format(city, year, weeks, dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))

                fig.savefig(pngfile, dpi=300)
            elif plot_type == 'monthly':
                print('Saving png for {}, month {}'.format(
                    ds.attrs['year'], ds.attrs['month']))
                pngfile = '{0}.png'.format(
                    '../{}_figures/{}_M{:02d}'.format(city, year, month, dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))

                fig.savefig(pngfile, dpi=300)


#############################

# don't use
def plot_single_residual(ds, plot_type='weekly', city='toronto'):
    """
    Return a Cartopy plot of averaged TROPOMI data ds over a given city.
    Aggregated data type is supplied to plot_type.

    Args:
        ds (xr.DataArray): TROPOMI tropospheric NO2 dataset.
        plot_type (str): accepted values: 'weekly', 'monthly'. Default: 'weekly'
        city (str): city name. Valid cities: 'toronto', 'vancouver', 'montreal',
            'new_york', 'los_angeles'. Default: 'toronto'

    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.tight_layout
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Load date
    # start = str(ds.attrs['first day'].date())
    # end = str(ds.attrs['last day'].date())
    year = ds.attrs['year']

    if plot_type == 'weekly':
        weeks = ds.attrs['weeks']
        date_str = 'weeks {}, {} to {}'.format(weeks, start, end)
    elif plot_type == 'monthly':
        month = ds.attrs['month']
        # date_str = 'month {:02d}, {} to {}'.format(month, start, end)
    else:
        return ValueError('plot_type must be "monthly" or "weekly"')

    ax.text(0, 1.07,
            r"NO$_2$ troposheric vertical column",
            fontsize=18,
            transform=ax.transAxes)
    ax.text(0, 1.02,
            r"{}, {}".format(cities[city], 'May 2019-2020'),
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
    for city in poi.cities_coords.keys():
        city_name = city[:-7]
        ax.plot(poi.cities_coords[city].lon,
                poi.cities_coords[city].lat,
                marker='*',
                markeredgewidth=2,
                markeredgecolor='black',
                markerfacecolor='black',
                markersize=5)
        ax.text(poi.cities_coords[city].lon + 0.05,
                poi.cities_coords[city].lat + 0.05,
                city_name,
                bbox=dict(facecolor='wheat',
                          edgecolor='black', boxstyle='round'),
                transform=ccrs.Geodetic())

    # set 0 values to np.nan
    # ds = ds.where(ds > 0, np.nan)

    # plot averaged values
    im = ds.isel(time=0).plot.pcolormesh(ax=ax,
                                         transform=ccrs.PlateCarree(),
                                         infer_intervals=True,
                                         cmap='viridis',
                                         #  vmin=10e-6,
                                         #  vmax=6e-5,
                                         #  norm=LogNorm(),
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
    cbar.set_label(r"$\log$(NO$_2)$ (mol/m$^2$)", labelpad=30, fontsize=14)
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

    lakes_50m = cfeature.NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale='10m',
        facecolor='none')

    ax.add_feature(countries, edgecolor='black')
    ax.add_feature(roads, edgecolor='gray')
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
    # if __name__ == '__main__':
    #     is_save = str(
    #         input('Do you want to save a png and KML of this plot? \n (Y/N)'))
    #     if is_save == 'Y' or is_save == 'y':
    #         if plot_type == 'weekly':
    #             print('Saving png for {}, weeks {}'.format(
    #                 ds.attrs['year'], ds.attrs['weeks']))
    #             pngfile = '{0}.png'.format(
    #                 '../{}_figures/{}_W{:02d}_{:02d}'.format(city, year, weeks, dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))

    #             fig.savefig(pngfile, dpi=300)
    #         elif plot_type == 'monthly':
    #             print('Saving png for {}, month {}'.format(
    #                 ds.attrs['year'], ds.attrs['month']))
    #             pngfile = '{0}.png'.format(
    #                 '../{}_figures/{}_M{:02d}'.format(city, year, month, dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))

    #             fig.savefig(pngfile, dpi=300)


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

    # Get residual
    if diff:
        new_ds = get_resid(ds[0], ds[1])
        ds.append(new_ds)

    col_nums = len(ds)

    fig, axes = plt.subplots(ncols=col_nums,
                             figsize=(col_nums*5, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    fig.tight_layout()

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

            w, e, s, n = plot_limits
            if ((w < poi.cities_coords[city_entry].lon) and
                (e > poi.cities_coords[city_entry].lon) and
                (s < poi.cities_coords[city_entry].lat) and
                    (n > poi.cities_coords[city_entry].lat)):

                color = next(marker)

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

        # plot NO2 TVCD
        im = ds[i].isel(time=0).plot.pcolormesh(ax=axes[i],
                                                transform=ccrs.PlateCarree(),
                                                infer_intervals=True,
                                                cmap='viridis',
                                                robust=True,
                                                x='longitude',
                                                y='latitude',
                                                add_colorbar=False)
        ims.append(im)

        # # remove default title
        axes[i].set_title('')

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
                               linewidth=1, color='gray', alpha=0.5, linestyle=':')
        gls.append(gl)

        gls[i].xlabels_top = False
        gls[i].ylabels_right = False
        gl.xlocator = ticker.FixedLocator(np.arange(int(plot_limits[0]),
                                                    int(plot_limits[1])+1, 0.5))
        gl.ylocator = ticker.FixedLocator(np.arange(int(plot_limits[2]),
                                                    int(plot_limits[3])+1, 0.5))
        gls[i].xformatter = LONGITUDE_FORMATTER
        gls[i].yformatter = LATITUDE_FORMATTER

    # set colorbar properties
    if diff:
        cba = create_colorbar(
            im=ims[0], ax=axes[:-1], label='Mean NO$_2$ TVCD')
        # cbb = create_colorbar(im=ims[-1], ax=axes[-1], label='Residual (NO$_2)$',
        #                       resid=True)

        cbar = plt.colorbar(mappable=ims[-1], ax=axes[-1], orientation='horizontal')
        # cbar.ax.xaxis.get_offset_text().set_visible(False)
        cbar.set_label(
            r'Difference', 
            labelpad=10, fontsize=14)

    else:
        cba = create_colorbar(im=ims[0], ax=axes[:], label='Mean NO$_2$ TVCD')

    # set legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=city_cntr)

    # Show plot
    plt.show()


if __name__ == '__main__':
    city = 'montreal'
    ds_list = load_ds(city, 2019, 2020)
    plot_residual(ds_list, plot_type='monthly', city=city, diff=True)
