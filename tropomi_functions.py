# Preamble
import numpy as np
import xarray as xr
import netCDF4 as nc
from glob import iglob
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


def read_data(file_name: str):
    """Return the fields defined within file_name and a shortened file_name for
    naming purposes.
    """

    fields = file_name.split('_')
    short_file_name = file_name[33:]
    return (fields, short_file_name)


def plot_no2(no2tc: xr.DataArray,
             plot_type: str,
             fields: list,
             short_file_name: str,
             plot_extent=(-180, 180, -90, 90)):
    """Return a Cartopy plot of tropospheric NO2 vertical column over Toronto
    using no2tc data.
    """
    # Create the figures and axes
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.tight_layout

    # Set map projection to Plate Carree
    ax = plt.axes(projection=ccrs.PlateCarree())

    # subset dataset for points over Toronto
    no2tc = subset(no2tc, plot_extent)

    # set all negative value to 0
    no2tc = no2tc.where(no2tc > 0, 0)

    if plot_type == 'Toronto' or plot_type == 'toronto':
        ax.text(0, 1.05,
            r"NO$_2$ tropospheric vertical column",
            fontsize=18,
            transform=ax.transAxes)
        ax.text(0, 1.02, 
                r"Toronto, Canada, sensing start: {}, sensing end {}".format(
                    # str(no2tc.time.data[0]).split('T')[0],
                    fields[8],
                    fields[9]),
                fontsize=14, 
                transform=ax.transAxes)

        # set map to plot within plot_extent
        ax.set_extent(plot_extent)
        
    elif plot_type == 'globe':
        ax.text(0, 1.10,
                r"NO$_2$ tropospheric vertical column",
                fontsize=18,
                transform=ax.transAxes)
        ax.text(0, 1.04,
                r"Orbit {}, sensing start: {}, sensing end {}".format(fields[10],
                                                                        fields[8],
                                                                        fields[9]),
                fontsize=14,
                transform=ax.transAxes)

        # set map to zoom out as much as possible
        ax.set_global()
        
    else:
        raise ValueError('Invalid plot_type. Expected one of: %s' % 'toronto, globe')
    
    # set plot frame color
    ax.outline_patch.set_edgecolor('lightgray')

    # plot data
    im = no2tc.isel(time=0).plot.pcolormesh(ax=ax,
                                            transform=ccrs.PlateCarree(),
                                            infer_intervals=True,
                                            cmap='jet',
                                            norm=LogNorm(vmin=10e-11,
                                                         vmax=10e-3),
                                            x='longitude',
                                            y='latitude',
                                            zorder=0,
                                            add_colorbar=False)
    
    # plot Toronto
    ax.plot(toronto_coords.lon, toronto_coords.lat, marker='o',
            markeredgewidth=1, markeredgecolor='black',
            markerfacecolor='black', markersize=10)
    ax.text(-79, 44, 'Toronto', transform=ccrs.Geodetic())

    # remove default title
    ax.set_title('')

    # set colorbar properties
    cbar_ax = fig.add_axes([0.38, 0.05, 0.25, 0.01])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"NO$_2$ (mol/m$^2$)", labelpad=-45, fontsize=14)
    cbar.outline.set_visible(False)

    # define Natural Earth features
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='10m',
        facecolor='none')

    lakes_50m = cfeature.NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale='10m',
        facecolor='none')
    
    # set map background and features
    ax.add_feature(states_provinces, edgecolor='black')
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
    
    # Ask user if they would like to save the plot
    is_save = str(input('Do you want to save a png of this plot? \n (Y/N)'))
    if is_save=='Y' or is_save=='y':
        if plot_type == 'toronto' or 'Toronto':
            pngfile = '{0}.png'.format('toronto_figures/' + short_file_name[:-3])
            fig.savefig(pngfile, dpi = 300)
        else:
            pngfile = '{0}.png'.format('world_figures/' + short_file_name[:-3])
            fig.savefig(pngfile, dpi = 300)
