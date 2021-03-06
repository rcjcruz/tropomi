#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to open multiple netCDF4 files to plot onto a map of the world using a
for loop. 
"""

# Preamble
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
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

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Functions
def subset(no2tc: xr.DataArray,
           plot_extent=(-179, 179, -89, 89)):
    """Return a subset of no2tc data over the plot extent.
    """
    e, w, s, n = plot_extent

    # crop dataset around point of interest and ensure qa_value >= 0.75
    no2tc = no2tc.where(
        (no2tc.longitude > e) &
        (no2tc.longitude < w) &
        (no2tc.latitude > s) &
        (no2tc.latitude < n) &
        (no2tc.isel(time=1) >= 0.75), drop=True)
    return no2tc

# Variables
xno2_path = '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/'
no2_sds_name = 'nitrogendioxide_tropospheric_column'
xno2_sds_name = 'nitrogendioxide_total_column'
qa_sds_name = 'qa_value'
paths = '/export/data/scratch/tropomi/no2/*__20200502*.nc'
# test_file = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc'


# Location of Toronto and other cities of interest
Point = namedtuple('Point', 'lon lat')
cities_coords = {'Toronto_coords': Point(-79.3832, 43.6532),
                 'Ottawa_coords': Point(-75.6972, 45.5215),
                 'Montreal_coords': Point(-73.5673, 45.5017),
                 'New York_coords': Point(-74.0060, 40.7128),
                 'Chicago_coords': Point(-87.6298, 41.8781)
                 }

extent_size = 15
plot_limits = (cities_coords['Toronto_coords'].lon-extent_size,
               cities_coords['Toronto_coords'].lon+extent_size,
               cities_coords['Toronto_coords'].lat-extent_size,
               cities_coords['Toronto_coords'].lat+extent_size)

# PLOTTING
fig, ax = plt.subplots(figsize=(15, 10))
fig.tight_layout

# Set map projection to Plate Carree
ax = plt.axes(projection=ccrs.PlateCarree())

# set plot frame color
ax.outline_patch.set_edgecolor('lightgray')
ax.set_extent(plot_limits)
# ax.set_global()

# Open NO2 datasets
for test_file in glob(paths):
    no2 = xr.open_dataset(test_file, group='/PRODUCT')[no2_sds_name]
    xno2 = xr.open_dataset(test_file, group=xno2_path)[xno2_sds_name]
    qa_value = xr.open_dataset(test_file, group='/PRODUCT')[qa_sds_name]

    # Add longitude and latitude to xno2 dataset
    xno2['latitude'] = no2['latitude']

    # Join xno2 and qa_value into a single dataset
    no2_qa = xr.concat((xno2, qa_value), dim='time')

    # Subset the no2_qa dataset
    no2_qa_filtered = subset(no2_qa)

    # set all negative value to 0
    no2_qa_filtered = no2_qa_filtered.where(no2_qa_filtered > 0, 0)
        
    # plot data
    im = no2_qa_filtered.isel(time=0).plot.pcolormesh(ax=ax,
                                                  transform=ccrs.PlateCarree(),
                                                  infer_intervals=True,
                                                  cmap='RdBu',
                                                  norm=LogNorm(vmin=2e-5,
                                                               vmax=8e-5),
                                                  x='longitude',
                                                  y='latitude',
                                                  zorder=0,
                                                  add_colorbar=False)

countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='10m',
    facecolor='none')

ax.add_feature(countries, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(lakes_50m, edgecolor='blue')
# ax.coastlines(resolution='50m', color='black', linewidth=1)

# set colorbar properties
cbar_ax = fig.add_axes([0.38, 0.05, 0.25, 0.01])
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"NO$_2$ (mol/m$^2$)", labelpad=-45, fontsize=14)
cbar.outline.set_visible(False)

# set gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle=':')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()
