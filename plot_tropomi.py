# Preamble
from grid_tropomi import *
from points_of_interest import *
import numpy as np
import xarray as xr
import pandas as pd
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
# f = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'

f = '*_20200505*_*.nc'
ds = dsread(f)
ds = aggregate_tropomi(ds=ds, res=0.5)

fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout
ax = plt.axes(projection=ccrs.PlateCarree())

date = ds.time.data[0]

ax.text(0, 1.10,
        r"NO$_2$ tropospheric vertical column",
        fontsize=18,
        transform=ax.transAxes)
ax.text(0, 1.04,
        r"{}".format(date),
        fontsize=14,
        transform=ax.transAxes)

# set map to zoom out as much as possible
# ax.set_global()
ax.set_extent(plot_limits)

# set 0 values to np.nan
ds = ds.where(ds > 0, np.nan)

# plot averaged values
im = ds.isel(time=0).plot.imshow(ax=ax,
                                 transform=ccrs.PlateCarree(),
                                 infer_intervals=True,
                                 cmap='viridis',
                                 norm=LogNorm(vmin=10e-6),
                                 robust=True,
                                 x='longitude',
                                 y='latitude',
                                 add_colorbar=False)

# remove default title
ax.set_title('')

# set colorbar properties
cbar_ax = fig.add_axes([0.38, 0.05, 0.25, 0.01])
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"NO$_2$ (mol/m$^2$)", labelpad=-45, fontsize=14)
cbar.outline.set_visible(False)

# Plot points of interest
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
# define Natural Earth features
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='10m',
    facecolor='none')

ax.add_feature(countries, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle=':')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

if __name__ == '__main__':
    # print(data_mean)
    plt.show()
        # Ask user if they would like to save the plot
    is_save = str(input('Do you want to save a png of this plot? \n (Y/N)'))
    if is_save=='Y' or is_save=='y':
        pngfile = '{0}.png'.format('toronto_figures/TOR_' + date)
        fig.savefig(pngfile, dpi = 300)
