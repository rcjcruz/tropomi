import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import warnings
from collections import namedtuple
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

import points_of_interest as poi

# Variables
# Define lat/lon for cities of interest
Point = namedtuple('Point', 'lon lat')
def get_plot_limits(city, extent=1, res=0.05):
    cities = {'toronto': Point(-79.3832, 43.6532),
              'montreal': Point(-73.5673, 45.5017),
              'new_york': Point(-74.0060, 40.7128),
              'vancouver': Point(-123.1207, 49.2827),
              'los_angeles': Point(-118.2437, 34.0522)}
    try:
        city_coords = cities[city]
    except KeyError:
        print('Not a valid city. Valid cities include %s' %
              list(cities.keys()))
    else:
        plot_limits = (city_coords.lon-extent, #llcrnlon
                       city_coords.lon+extent+res, #urcrnlon
                       city_coords.lat-extent, #llcrnlat
                       city_coords.lat+extent+res) #urcrnlat
        return plot_limits

def get_wind_speed_and_dir(ds):
    # Load U850 and V850 variables 
    if ('U850' not in ds.variables) or ('V850' not in ds.variables):
        raise KeyError('"U850" and "V850" are required to calculate wind speed.')
    else:
        u = ds.U850
        v = ds.V850 
        speed = np.sqrt(u**2 + v**2)
        bearing = np.degrees(np.arctan2(v, u))
        
    return speed, bearing.where(bearing > 0, bearing + 360)


# Load wind data
data = xr.open_dataset("/export/data/scratch/merra2/tavg1_wind/MERRA2_400.tavg1_2d_slv_Nx.20200501.nc4.nc4")
data['speed'], data['bearing'] = get_wind_speed_and_dir(data)

lats = data['lat']
lons = data['lon']
lon, lat = np.meshgrid(lons, lats)
U850 = data['U850']
V850= data['V850']
U850_nans = U850[:]
V850_nans = V850[:]
fmissing_U850 = U850.fmissing_value
fmissing_V850 = V850.fmissing_value


U850_nans.where(U850_nans == fmissing_U850, np.nan)
V850_nans.where(V850_nans == fmissing_V850, np.nan)

ws = np.sqrt(U850_nans ** 2 + V850_nans ** 2)
ws_dir = np.arctan2(V850_nans, U850_nans)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ws_daily_avg = np.nanmean(ws, 0)

# Plotting wind speeds
# fig = plt.figure(figsize=(8,4))
# ax = plt.axes(projection=ccrs.Robinson())
# ax.set_global()

# plot_limits = get_plot_limits(city='toronto', extent=2, res=-0.05)
# ax.set_extent(plot_limits, crs=ccrs.PlateCarree())
# ax.coastlines(resolution="110m",linewidth=1)
# ax.gridlines(linestyle='--',color='black')

# clevs = np.arange(0,19,1)
# plt.contourf(lon, lat, ws_daily_avg, clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)
# plt.title('MERRA-2 Daily Average 2-meter Wind Speed at 850hPa, 31 May 2020', size=14)
# cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
# cb.set_label('m/s',size=12,rotation=0,labelpad=15)
# cb.ax.tick_params(labelsize=10)

# states_provinces = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='admin_1_states_provinces_lines',
#     scale='10m',
#     facecolor='none')

# roads = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='roads',
#     scale='10m',
#     facecolor='none')

# ax.add_feature(states_provinces, edgecolor='gray')
# ax.add_feature(roads, edgecolor='gray')

# plt.show()
# # Save figure as PNG:
# # fig.savefig('MERRA2_850hpa_ws.png', format='png', dpi=120)


# Plot wind speed and direction

# Set the figure size, projection, and extent
fig = plt.figure(figsize=(9,5))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-62,-38,35,54])
plot_limits = poi.get_plot_limits(city='toronto', extent=2, res=-0.05)
ax.set_extent(plot_limits, crs=ccrs.PlateCarree())
ax.coastlines(resolution="50m",linewidth=1)
# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-65,-60,-50,-40,-30])
gl.ylocator = mticker.FixedLocator([30,40,50,60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':10, 'color':'black'}
gl.ylabel_style = {'size':10, 'color':'black'}

# Plot windspeed
clevs = np.arange(0,14.5,1)
plt.contourf(lon, lat, ws[0,:,:], clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)
plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 May 2020', size=16)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('m/s',size=14,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=10)
# Overlay wind vectors
qv = plt.quiver(lon, lat, U850_nans[0,:,:], V850_nans[0,:,:], scale=420, color='k')

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

ax.add_feature(states_provinces, edgecolor='gray')
ax.add_feature(roads, edgecolor='gray')


plt.show()
# Save figure as PNG:
# fig.savefig('MERRA2_2m_wsVECTORS.png', format='png', dpi=120)