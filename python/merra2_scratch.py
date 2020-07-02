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
import merra2_functions as m2f
import itertools

# Functions


def get_wind_speed_and_dir(ds):
    # Load U850 and V850 variables
    if ('U850' not in ds.variables) or ('V850' not in ds.variables):
        raise KeyError(
            '"U850" and "V850" are required to calculate wind speed.')
    else:
        u = ds.U850
        v = ds.V850
        speed = np.sqrt(u**2 + v**2)
        bearing = np.degrees(np.arctan2(v, u))

    return speed, bearing.where(bearing > 0, bearing + 360)


# Load wind data
data = xr.open_dataset(
    "/export/data/scratch/merra2/tavg1_wind/MERRA2_400.tavg1_2d_slv_Nx.20200501.nc4.nc4")
data['speed'], data['bearing'] = get_wind_speed_and_dir(data)

lats = data['lat']
lons = data['lon']

lonmn, lonmx, latmn, latmx = poi.get_plot_limits('toronto', extent=2)
# lats = np.arange(latmn, latmx, 0.25)
# lons = np.arange(lonmn, lonmx, 0.25)
lon, lat = np.meshgrid(lons, lats)

if __name__ == '__main__':
    # Plotting MERRA2
    # Create a copy of u and v
    u = data['U850']
    v = data['V850']
    u_nans = u[:]
    v_nans = v[:]
    fmissing_u = u.fmissing_value
    fmissing_v = v.fmissing_value

    u_nans.where(u_nans == fmissing_u, np.nan)
    v_nans.where(v_nans == fmissing_v, np.nan)

    ws = np.sqrt(u_nans ** 2 + v_nans ** 2)

    # Plot wind speed and direction
    # Set the figure size, projection, and extent
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent([-62,-38,35,54])

    plot_limits = poi.get_plot_limits(city='toronto', extent=1, res=0.05)
    ax.set_extent(plot_limits, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=1)

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='black', linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(np.arange(int(plot_limits[0])-1,
                                                 int(plot_limits[1])+1, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(int(plot_limits[2])-1,
                                                 int(plot_limits[3])+1, 2))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # # Plot cities of interest
    marker = itertools.cycle(('black', 'blue', 'red', 'orange', 'green',
                              'purple', 'gray'))
    city_cntr = 1
    for city_entry in poi.cities_coords.keys():
        city_name = city_entry[:-7]

        # Check if city lat/lon fall within boundary box
        w, e, s, n = plot_limits
        if ((w < poi.cities_coords[city_entry].lon) and
            (e > poi.cities_coords[city_entry].lon) and
            (s < poi.cities_coords[city_entry].lat) and
                (n > poi.cities_coords[city_entry].lat)):

            color = next(marker)

            # Plot the city
            ax.plot(poi.cities_coords[city_entry].lon,
                    poi.cities_coords[city_entry].lat,
                    linestyle='None',
                    marker='*',
                    markeredgewidth=1,
                    markeredgecolor=color,
                    markerfacecolor=color,
                    markersize=5,
                    label=city_name)
            city_cntr += 1

    # Plot windspeed
    # clevs = np.arange(0, 14.5, 1)
    # plt.contourf(lon, lat, ws[0, :, :], clevs,
    #              transform=ccrs.PlateCarree(), cmap=plt.cm.jet)
    # plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 May 2020', size=16)
    # cb = plt.colorbar(ax=ax, orientation="vertical",
    #                   pad=0.02, aspect=16, shrink=0.8)
    # cb.set_label('m/s', size=14, rotation=0, labelpad=15)
    # cb.ax.tick_params(labelsize=10)
    # Overlay wind vectors

    winds = m2f.load_wind_data(city='toronto',
                           month=5, year=2020, time=15)

    # qv = plt.quiver(lon, lat, u_nans[0, :, :],
    #                 v_nans[0, :, :], scale=100, color='k')
    
    qv = plt.quiver(winds.lon, winds.lat, winds.u[0][0], winds.v[0][0],
                    scale=50, color='k')
    qk = plt.quiverkey(qv, 0.8, 0.7, 1, r'1 $\frac{m}{s}$',
                       labelpos='E', coordinates='figure')
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

    # set legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=city_cntr)

    plt.show()
    # Save figure as PNG:
    # fig.savefig('MERRA2_2m_wsVECTORS.png', format='png', dpi=120)
