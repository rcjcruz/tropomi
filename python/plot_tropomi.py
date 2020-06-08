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
import datetime
from datetime import timedelta
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.colorbar import colorbar

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

#############################

def plot_tropomi(ds):
    """
    Return a Cartopy plot of averaged TROPOMI data ds.  

    plot_type: 'toronto' or 'world'
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.tight_layout
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Load date
    start = str(ds.attrs['first day'].date())
    end = str(ds.attrs['last day'].date())
    weeks = ds.attrs['weeks']
    year = ds.attrs['year']

    date_str = 'weeks {}, {} to {}'.format(weeks, start, end)
    ax.text(0, 1.07,
            r"NO$_2$ troposheric vertical column",
            fontsize=18,
            transform=ax.transAxes)
    ax.text(0, 1.02,
            r"Toronto, Canada, {}".format(date_str),
            fontsize=14,
            transform=ax.transAxes)

    # set map to plot within plot_limits
    ax.set_extent(poi.plot_limits)

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
        if city_name == 'Montreal':
            ax.text(poi.cities_coords[city].lon + 0.3,
                    poi.cities_coords[city].lat + 0.5,
                    city_name,
                    bbox=dict(facecolor='wheat',
                              edgecolor='black', boxstyle='round'),
                    transform=ccrs.Geodetic())
        else:
            ax.text(poi.cities_coords[city].lon + 0.1,
                    poi.cities_coords[city].lat + 0.2,
                    city_name,
                    bbox=dict(facecolor='wheat',
                              edgecolor='black', boxstyle='round'),
                    transform=ccrs.Geodetic())

    # set 0 values to np.nan
    ds = ds.where(ds > 0, np.nan)

    # plot averaged values
    im = ds.isel(time=0).plot.imshow(ax=ax,
                                     transform=ccrs.PlateCarree(),
                                     infer_intervals=True,
                                     cmap='viridis',
                                     vmin=10e-6,
                                     vmax=6e-5,
                                     norm=LogNorm(),
                                     robust=True,
                                     x='longitude',
                                     y='latitude',
                                     add_colorbar=False)

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
    # plt.show()

    # Save data to world_figures to toronto_figures with the time
    # of processing appended to the file name
    # ONLY runs if plot_tropomi.py is run directly
    if __name__ == '__main__':
        # is_save = str(
        #     input('Do you want to save a png of this plot? \n (Y/N)'))
        # if is_save == 'Y' or is_save == 'y':
        print('Saving png for {}, weeks {}'.format(ds.attrs['year'], ds.attrs['weeks']))
        pngfile = '{0}.png'.format(
            '../toronto_figures/TOR_{}_W{}_{}'.format(year, weeks, dt.datetime.now().strftime('_%Y%m%dT%H%M%S')))

        fig.savefig(pngfile, dpi=300)

#############################


# def get_start_and_end_date_from_calendar_week(year, calendar_week):
#     """Calendar week SHOULD be odd
#     Shows two weeks worth of dates.
#     """
#     monday = datetime.datetime.strptime(
#         f'{year}-{calendar_week}-1', '%Y-%W-%w').date()
#     return monday, monday + datetime.timedelta(days=13.9)


# if __name__ == '__main__':
#     f =
#     plot_tropomi(f)

if __name__ == '__main__':
    input_files = os.path.join(tropomi_pkl_week, '*')
    for test_file in sorted(glob.glob(input_files)):   
        infile = open(test_file, 'rb')
        ds = pickle.load(infile)
        infile.close()
        plot_tropomi(ds)
