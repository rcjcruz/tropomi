# Preamble
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import datetime as dt
from netCDF4 import Dataset
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


np.set_printoptions(edgeitems=3, infstr='inf',
                    linewidth=75, nanstr='nan', precision=8,
                    suppress=False, threshold=1000, formatter=None)

# Variables
xno2_path = '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/'
no2_sds_name = 'nitrogendioxide_tropospheric_column'
xno2_sds_name = 'nitrogendioxide_total_column'
qa_sds_name = 'qa_value'

file_name = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200502T080302_20200502T094432_13222_01_010302_20200504T005011.nc'
paths = '/export/data/scratch/tropomi/no2/*__20200505*.nc'

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

latmn, lonmn, latmx, lonmx = (-90, -180, 90, 180)
lat_bnds = np.arange(latmn, latmx, 1)
lon_bnds = np.arange(lonmn, lonmx, 1)
pv_arr = np.zeros([lat_bnds.size, lon_bnds.size])
dens_arr = np.zeros([lat_bnds.size, lon_bnds.size], dtype=np.int32)

ds_old = Dataset(file_name, 'r')
tcfmt = '%Y-%m-%dT%H:%M:%SZ'
print(ds_old.id)
grp = ds_old.groups['PRODUCT']
grp_geoloc = grp.groups['SUPPORT_DATA'].groups['GEOLOCATIONS']

# Plot_var function
no2 = grp.variables['nitrogendioxide_tropospheric_column'][0]
scf = 1.  # why do we multiply no2 by scf? unit conversion
# unsure what happens here and how to translate this to xr
plot_var = scf*no2.flatten()


lat = grp.variables['latitude'][0].flatten()
lon = grp.variables['longitude'][0].flatten()
qual = grp.variables['qa_value'][0].flatten()

# Creates a masked array if a value is found within the bounds
lat_flt = (lat > latmn)*(lat < latmx)
lon_flt = (lon > lonmn)*(lon < lonmx)
qual_flt = qual > 0.5
flt = lat_flt*lon_flt*qual_flt
plot_var = plot_var[flt]

vlat = grp_geoloc.variables['latitude_bounds'][:]
vlon = grp_geoloc.variables['longitude_bounds'][:]
# read the documentation for this
vlat = np.array([a.flatten() for a in np.rollaxis(vlat, -1)])
vlon = np.array([a.flatten() for a in np.rollaxis(vlon, -1)])  # read PUM

# trying to get the min/max lat/lon values for the box
vlatmn = np.amin(vlat, axis=0)[flt]
vlatmx = np.amax(vlat, axis=0)[flt]
vlonmn = np.amin(vlon, axis=0)[flt]
vlonmx = np.amax(vlon, axis=0)[flt]


###### ended here
for k in range(plot_var.size):
    # Find the indices in the lat/lon_bnds arrays at which the 
    # max/min lat/lon would fit (i.e. finding the grid square that the data 
    # point fits in)
    lat_inds = np.searchsorted(lat_bnds, np.array([vlatmn[k], vlatmx[k]]))  # look at documentation
    lon_inds = np.searchsorted(lon_bnds, np.array([vlonmn[k], vlonmx[k]]))
    lat_slice = slice(lat_inds[0], lat_inds[1]+1)
    lon_slice = slice(lon_inds[0], lon_inds[1]+1)

    pv_arr[lat_slice, lon_slice] += plot_var[k]
    dens_arr[lat_slice, lon_slice] += 1

no_sound = dens_arr == 0  # this means there were no values found in this area
dens_arr_nz = np.copy(dens_arr)  # not sure what's happening here
dens_arr_nz[no_sound] = 1
plot_var_mean = pv_arr / dens_arr_nz  # this calculates the average

if clim:
    cmin, cmax = clim
else:
    plot_var_nz = plot_var_mean[plot_var_mean != 0]
    cmin = np.floor(cbscf*(plot_var_nz.mean() - 2*plot_var_nz.std()))/cbscf
    cmax = np.ceil(cbscf*(plot_var_nz.mean() + 2*plot_var_nz.std()))/cbscf

k = kml.KML(name='TROPOMI')
foots = kml.Folder('TROPOMI Footprints')
pcount = 0
for i in range(plot_var_mean.shape[0]):
    for j in range(plot_var_mean.shape[1]):
        if dens_arr[i, j] != 0:
            lllat = lat_bnds[i]
            lllon = lon_bnds[j]
            vlat = np.array([lllat, lllat, lllat+res, lllat+res])
            vlon = np.array([lllon, lllon+res, lllon+res, lllon])

            col = colours.Colour(plot_var_mean[i, j], vmin=cmin, vmax=cmax).cformat(
                colours.Colour.GE)
            plgn = kml.Polygon(vlat, vlon, str(pcount), col)
            foots.add_object(plgn)

            pcount += 1

cbar = colours.colour_bar('', cmin=cmin, cmax=cmax, label=cbl)
colour_scale = kml.Overlay.colour_scale(cbar)
k.add_object(colour_scale)
k.add_folder(foots)

kml_floc = os.path.join(kml_dir, kml_fname)
k.write(kml_floc)

np.set_printoptions(threshold=np.inf)

# def aggregate_grid(summary_floc, kml_fname, bbox, res, dmin, dmax, var='no2', clim=None):
#     """
#     summary_floc
#     kml_fname
#     bbox: boundary box (s, w, n, e)
#     """
# latmn, lonmn, latmx, lonmx = bbox
# lat_bnds = np.arange(latmn, latmx, res)
# lon_bnds = np.arange(lonmn, lonmx, res)
# pv_arr = np.zeros([lat_bnds.size, lon_bnds.size])
# dens_arr = np.zeros([lat_bnds.size, lon_bnds.size], dtype=np.int32)

# if var == 'no2':
#     cbl = 'Tropospheric NO$_2$ Column'
#     kml_dir = paths.kml_no2
#     cbscf=1.
# else:
#     raise ValueError("var must be 'no2' given %s" % var)

# tropomi_ds = read_summary_file(summary_floc)
# tcfmt = '%Y-%m-%dT%H:%M:%SZ'
# for ds in tropomi_ds:
#     d0 = dt.datetime.strptime(ds.time_coverage_start, tcfmt)
#     d1 = dt.datetime.strptime(ds.time_coverage_end, tcfmt)
#     if d1 > dmin and d0 < dmax:
#         print(ds.id)
#         grp = ds.groups['PRODUCT']
#         grp_geoloc = grp.groups['SUPPORT_DATA'].groups['GEOLOCATIONS']

#         plot_var = var_sel(ds, var)
#         lat = grp.variables['latitude'][0].flatten()
#         lon = grp.variables['longitude'][0].flatten()
#         qual = grp.variables['qa_value'][0].flatten()

#         #Creates a masked array if a value is found within the bounds
#         lat_flt = (lat>latmn)*(lat<latmx)
#         lon_flt = (lon>lonmn)*(lon<lonmx)
#         qual_flt = qual > 0.5
#         flt = lat_flt*lon_flt*qual_flt
#         plot_var = plot_var[flt]

#         vlat = grp_geoloc.variables['latitude_bounds'][:]
#         vlon = grp_geoloc.variables['longitude_bounds'][:]
#         vlat = np.array([a.flatten() for a in np.rollaxis(vlat, -1)])
#         vlon = np.array([a.flatten() for a in np.rollaxis(vlon, -1)])

#         vlatmn = np.amin(vlat, axis=0)[flt]
#         vlatmx = np.amax(vlat, axis=0)[flt]
#         vlonmn = np.amin(vlon, axis=0)[flt]
#         vlonmx = np.amax(vlon, axis=0)[flt]

#         for k in range(plot_var.size):
#             lat_inds = np.searchsorted(lat_bnds, np.array([vlatmn[k], vlatmx[k]]))
#             lon_inds = np.searchsorted(lon_bnds, np.array([vlonmn[k], vlonmx[k]]))
#             lat_slice = slice(lat_inds[0], lat_inds[1]+1)
#             lon_slice = slice(lon_inds[0], lon_inds[1]+1)

#             pv_arr[lat_slice,lon_slice] += plot_var[k]
#             dens_arr[lat_slice,lon_slice] += 1

# no_sound = dens_arr == 0
# dens_arr_nz = np.copy(dens_arr)
# dens_arr_nz[no_sound] = 1
# plot_var_mean = pv_arr / dens_arr_nz

# if clim:
#     cmin, cmax = clim
# else:
#     plot_var_nz = plot_var_mean[plot_var_mean != 0]
#     cmin = np.floor(cbscf*(plot_var_nz.mean() - 2*plot_var_nz.std()))/cbscf
#     cmax = np.ceil(cbscf*(plot_var_nz.mean() + 2*plot_var_nz.std()))/cbscf

# k = kml.KML(name='TROPOMI')
# foots = kml.Folder('TROPOMI Footprints')
# pcount = 0
# for i in range(plot_var_mean.shape[0]):
#     for j in range(plot_var_mean.shape[1]):
#         if dens_arr[i,j] != 0:
#             lllat = lat_bnds[i]
#             lllon = lon_bnds[j]
#             vlat = np.array([lllat, lllat, lllat+res, lllat+res])
#             vlon = np.array([lllon, lllon+res, lllon+res, lllon])

#             col = colours.Colour(plot_var_mean[i,j], vmin=cmin, vmax=cmax).cformat(colours.Colour.GE)
#             plgn = kml.Polygon(vlat, vlon, str(pcount), col)
#             foots.add_object(plgn)

#             pcount += 1

# cbar = colours.colour_bar('', cmin=cmin, cmax=cmax, label=cbl)
# colour_scale = kml.Overlay.colour_scale(cbar)
# k.add_object(colour_scale)
# k.add_folder(foots)

# kml_floc = os.path.join(kml_dir, kml_fname)
# k.write(kml_floc)


# def var_sel(ds, var, flat=True):

#     grp = ds.groups['PRODUCT']
#     if var == 'no2':
#         no2 = grp.variables['nitrogendioxide_tropospheric_column'][0]
#         scf = 1.
#         out = scf*no2.flatten() if flat else scf*no2
#     else:
#         raise ValueError("var must be one of 'co' or 'no2' given %s" % var)

#     return out
