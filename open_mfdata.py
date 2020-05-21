# Preamble
import numpy as np
import xarray as xr
import netCDF4 as nc
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

paths = glob("/export/data/scratch/tropomi/no2/*__20200505*.nc")
ds2 = xr.open_mfdataset(paths, group='PRODUCT')[
    'nitrogendioxide_tropospheric_column']


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

# Functions


def subset(no2tc: xr.DataArray,
           plot_extent=(-180, 180, -90, 90)):
    """Return a subset of no2tc data over the plot extent.
    """
    e, w, s, n = plot_extent

    # crop dataset around point of interest
    no2tc = no2tc.where(
        (no2tc.longitude > e) &
        (no2tc.longitude < w) &
        (no2tc.latitude > s) &
        (no2tc.latitude < n), drop=True)
    return no2tc


fig, ax = plt.subplots(figsize=(15, 10))
fig.tight_layout

# Set map projection to Plate Carree
ax = plt.axes(projection=ccrs.PlateCarree())

ds2 = subset(ds2)

# set all negative value to 0
ds2 = ds2.where(ds2 > 0, 0)

# set plot frame color
ax.outline_patch.set_edgecolor('lightgray')

# plot data
im = ds2.isel(time=0).plot.pcolormesh(ax=ax,
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
