# Preamble
import warnings
import numpy as np
import numpy.ma as ma
import types
import xarray as xr
import netCDF4 as nc
import pandas as pd
from glob import glob
from os.path import join
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

# Script
paths = glob("/export/data/scratch/tropomi/no2/*__20200505*.nc")
# no2 = xr.open_mfdataset(paths, group='PRODUCT')['nitrogendioxide_tropospheric_column']
# qa_value = xr.open_mfdataset(paths, group='PRODUCT')['qa_value']
# no2_qa = xr.concat((no2, qa_value), dim='time')

ds_list = []
for p in paths:
    ds = xr.open_dataset(p, group='PRODUCT')[
        'nitrogendioxide_tropospheric_column']
    ds_list.append(ds)

no2_qa = xr.concat(ds_list, dim='time')

# print(no2_qa)


# Get date
# date = no2['time'].values[0]
# date = pd.to_datetime(date)
# date = str(date.date())
# print('Processing plot for', date)

# ---------------------------------------------------
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

# # Functions


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
        (no2tc.latitude < n), drop=True)
    return no2tc


no2_qa = subset(no2_qa, plot_limits)
# print(no2_qa.longitude)

# PLOTTING
fig, ax = plt.subplots(figsize=(15, 10))
fig.tight_layout

# Set map projection to Plate Carree
ax = plt.axes(projection=ccrs.PlateCarree())

# set all negative value to 0
no2_qa = no2_qa.where(no2_qa > 0, drop=True)

# print(no2_qa['longitude'])


# set plot frame color
ax.outline_patch.set_edgecolor('lightgray')
ax.set_global()

# plot data
i = 0
while i < no2_qa.sizes['time']:
    im = no2_qa.isel(time=i).plot.contourf(ax=ax,
                                  transform=ccrs.PlateCarree(),
                                  infer_intervals=True,
                                  cmap='jet',
                                  norm=LogNorm(vmin=10e-6),
                                  x='longitude',
                                  y='latitude',
                                  zorder=0,
                                  add_colorbar=False)
    i += 1


countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='10m',
    facecolor='none')

ax.add_feature(countries, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(lakes_50m, edgecolor='blue')
# ax.coastlines(resolution='50m', color='black', linewidth=1)

# Plot cities of interest
for city in cities_coords.keys():
    city_name = city[:-7]
    ax.plot(cities_coords[city].lon,
            cities_coords[city].lat,
            marker='*',
            markeredgewidth=1,
            markeredgecolor='black',
            markerfacecolor='black',
            markersize=5)
    ax.text(cities_coords[city].lon - 1,
            cities_coords[city].lat + 0.3,
            city_name,
            transform=ccrs.Geodetic())

# remove default title
ax.set_title('')

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




