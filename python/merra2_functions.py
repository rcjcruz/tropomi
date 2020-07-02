import xarray as xr
import numpy as np
import os
import glob
import points_of_interest as poi
from paths import *


def load_wind_data(city, month, year, time):
    """
    Return a xr.DataArray with wind data over a city averaged 
    at time given month and year.

    Args:
        city (str): city name. Valid cities: 'toronto', 'vancouver', 'montreal',
            'new_york', 'los_angeles'. Default: 'toronto'
        month (int): month of the year. 1 <= month <= 12
        year (int): year.
        time (int): time of day in 24h format. 0 <= time <= 23

    Returns:
        avg_wind (xr.DataArray): wind data.

    >>> load_wind_data('toronto', 5, 2020, 15)
    """

    # load plot limits of city
    lonmn, lonmx, latmn, latmx = poi.get_plot_limits(
        city=city, extent=1, res=0)
    city_lat = poi.cities[city].lat
    city_lon = poi.cities[city].lon
    # Create a uniform lat/lon grid
    lat_bnds = np.array([latmn, latmx])
    lon_bnds = np.array([lonmn, lonmx])

    # Create 1d array to hold average values
    u_arr = np.zeros([1, 1])
    v_arr = np.zeros([1, 1])
    dens_arr = np.zeros([1, 1], dtype=np.int32)

    # Open wind data
    fpath = '*{}{:02d}*'.format(year, month)
    fdir = os.path.join(winds, fpath)

    for f in glob.glob(fdir):
        data = xr.open_dataset(f)
        hour_data = data.isel(time=time)

        # Load lat, lon, U850, V850 data
        lat, lon = hour_data.lat, hour_data.lon
        u, v = hour_data.U850.values, hour_data.V850.values

        # Filter lat/lon values within lat/lon mn/mx
        lat_flt = (lat > latmn) * (lat < latmx)
        lon_flt = (lon > lonmn) * (lon < lonmx)

        # Create a filtering array
        filter_arr = lat_flt * lon_flt

        # Filter data
        u = u[filter_arr]
        v = v[filter_arr]

        # Increment dens_arr while adding winds to u_arr and v_arr
        for i in range(len(u)):
            u_arr += u[i]
            v_arr += v[i]
            dens_arr += 1

    # Divide wind speeds by dens_arr
    u_arr_mean = np.divide(u_arr, dens_arr)
    v_arr_mean = np.divide(v_arr, dens_arr)
    ws = np.sqrt(u_arr_mean ** 2 + v_arr_mean ** 2)

    # Create dataset for monthly averaged wind data
    ds = xr.Dataset({'u': (['lat', 'lon'], u_arr_mean),
                     'v': (['lat', 'lon'], v_arr_mean),
                     'ws': (['lat', 'lon'], ws)},
                    coords={'lat': np.atleast_1d(city_lat),
                            'lon': np.atleast_1d(city_lon)},
                    attrs={'time': '{}-{:02d} {}:00-{}:00 daily'.format(year,
                                                                        month,
                                                                        time,
                                                                        time + 1)})

    return ds


ds = load_wind_data(city='toronto',
                    month=5, year=2020, time=15)
