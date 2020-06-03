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


def plot_tropomi(ds, res, plot_type='toronto'):
    """
    Return a Cartopy plot of averaged TROPOMI data ds.  

    plot_type: 'toronto' or 'world'
    """

    # # Aggregate data
    # ds = gt.aggregate_tropomi(ds, res, plot_type)

    # Create figure and axes
    if plot_type == 'world':
        fig, ax = plt.subplots(figsize=(12, 8))
    elif plot_type == 'toronto':
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        return ValueError('plot_type must be \'toronto\' or \'world\'')
    fig.tight_layout
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Load date
    date = str(ds.time.data[0]).split('T')[0]

    if plot_type == 'toronto':
        ax.text(0, 1.07,
                r"NO$_2$ troposheric vertical column",
                fontsize=18,
                transform=ax.transAxes)
        ax.text(0, 1.02,
                r"Toronto, Canada, {}".format(date),
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
                ax.text(poi.cities_coords[city].lon - 1,
                        poi.cities_coords[city].lat + 0.5,
                        city_name,
                        bbox=dict(facecolor='wheat',
                                  edgecolor='black', boxstyle='round'),
                        transform=ccrs.Geodetic())

    elif plot_type == 'world':
        ax.text(0, 1.10,
                r"NO$_2$ tropospheric vertical column",
                fontsize=18,
                transform=ax.transAxes)
        ax.text(0, 1.04,
                r"{}".format(date),
                fontsize=14,
                transform=ax.transAxes)

        # set map to zoom out as much as possible
        ax.set_global()

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
    if plot_type == 'world':
        cbar_ax = fig.add_axes([0.38, 0.05, 0.25, 0.01])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r"$\log$(NO$_2)$ (mol/m$^2$)", labelpad=30, fontsize=14)
        cbar.outline.set_visible(False)
    elif plot_type == 'toronto':
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

    ax.add_feature(countries, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)

    if plot_type == 'toronto':
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
            input('Do you want to save a png of this plot? \n (Y/N)'))
        if is_save == 'Y' or is_save == 'y':
            if (plot_type == 'world'):
                pngfile = '{0}.png'.format(
                    '../world_figures/WOR_' + date + dt.datetime.now().strftime('_%Y%m%dT%H%M%S'))
            elif (plot_type == 'toronto'):
                pngfile = '{0}.png'.format(
                    '../toronto_figures/TOR_' + date + dt.datetime.now().strftime('_%Y%m%dT%H%M%S'))

            fig.savefig(pngfile, dpi=300)

#############################


# def get_start_and_end_date_from_calendar_week(year, calendar_week):
#     """Calendar week SHOULD be odd
#     Shows two weeks worth of dates.
#     """
#     monday = datetime.datetime.strptime(
#         f'{year}-{calendar_week}-1', '%Y-%W-%w').date()
#     return monday, monday + datetime.timedelta(days=13.9)


if __name__ == '__main__':
    # # # f = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'
    # s = ['*__20200501*_*.nc',
    #      '*__20200502*_*.nc',
    #      '*__20200503*_*.nc',
    #      '*__20200504*_*.nc',
    #      '*__20200505*_*.nc']
    # # s = ['*__20200505*_*.nc']
    # for f in s:
    f = '*__20200505*_*.nc'
    g = '*__20200504*_*.nc'
    ds1 = gt.aggregate_tropomi(ot.dsread(f))
    ds2 = gt.aggregate_tropomi(ot.dsread(g))

    # join data
    ds = xr.concat([ds1, ds2], dim='time')
    # ds = gt.aggregate_tropomi(ds=ds, res=0.5)
    plot_tropomi(ds=ds, plot_type='toronto', res=0.05)
