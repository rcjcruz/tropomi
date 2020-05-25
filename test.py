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

# Variables
file_name = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200504T173409_20200504T191539_13256_01_010302_20200506T103112.nc'
sds_name = 'nitrogendioxide_tropospheric_column'
paths = '/export/data/scratch/tropomi/no2/*__20200504*.nc'

# Open NO2 tropospheric column
no2 = xr.open_dataset(file_name, group='/PRODUCT')[sds_name]
qa_value = xr.open_dataset(file_name, group='PRODUCT')['qa_value']

no2_qa = xr.concat((no2, qa_value), dim='time')

# Plot NO2 data


def subset(no2tc: xr.DataArray,
           plot_extent=(-180, 180, -90, 90)):
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


no2_qa_filtered = subset(no2_qa)

# PLOTTING
fig, ax = plt.subplots(figsize=(15, 10))
fig.tight_layout

# Set map projection to Plate Carree
ax = plt.axes(projection=ccrs.PlateCarree())

# set all negative value to 0
no2_qa_filtered = no2_qa_filtered.where(no2_qa_filtered > 0, 0)

# set plot frame color
ax.outline_patch.set_edgecolor('lightgray')
# ax.set_extent(plot_limits)
ax.set_global()

# plot data
im = no2_qa_filtered.isel(time=0).plot.pcolormesh(ax=ax,
                                                  transform=ccrs.PlateCarree(),
                                                  infer_intervals=True,
                                                  cmap='jet',
                                                  norm=LogNorm(vmin=10e-11),
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

# set gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle=':')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()
