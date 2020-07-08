import xarray as xr
import numpy as np
import pandas as pd
import time
import glob
import os
import pickle

from paths import *
import points_of_interest as poi
import open_tropomi as ot


def get_wind_speed_and_dir(ds):
    """
    Return wind speed and bearing given u and v of ds.
    
    Args:
        ds (xr.Dataset): dataset of wind containing U850 and V850 variables.
        
    Returns:
        ds (xr.Dataset): datasets of wind with speed and bearing variables appended.
    """
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

################################

def add_wind(f, city='toronto'):
    """
    Return a dataset for data f over city with wind data that matches lat/lon/time
    of TROPOMI observation.
    
    Args:
        f (str): date string of the form YYYYMMDD.
        city (str): city of interest. 
    >>> add_wind('20200501', 'toronto')
    """
    
    start_time = time.time()

    # Load city limits
    plot_limits = poi.get_plot_limits(city=city, extent=1, res=0)
    e, w, s, n = plot_limits
    
    # Load dataset
    no2 = ot.dsread(f, city)
    # Subset NO2 dataset over +-1 deg lat/lon around the city
    no2 = no2.where((no2.longitude > e) & 
                    (no2.longitude < w) & 
                    (no2.latitude > s) & 
                    (no2.latitude < n), drop=True)
    no2 = no2.rename({'time': 'measurement_time'}) # rename time
    no2['wind_speed'] = (['sounding'], np.zeros([no2.sounding.size])) # create ws variable
    no2['bearing'] = (['sounding'], np.zeros([no2.sounding.size])) # create  bearing variable

    # Load wind
    f_str = '*' + f + '*'
    fpath = os.path.join(winds, f_str)
    for file in glob.glob(fpath):
        wind = xr.open_dataset(file)
        interp_wind = wind.interp(
            lat=no2.latitude, lon=no2.longitude, method='linear')
        interp_wind['wind_speed'], interp_wind['bearing'] = get_wind_speed_and_dir(
            interp_wind)
        interp_wind = interp_wind.dropna(dim='sounding')

    # iterate over each observation and append wind speed and bearing to no2
    for i in range(len(no2.scanline)):
        print('Reading scanline', i)
        # Load timestamp of observation
        t_obs = pd.to_datetime(no2.scanline[i].values)
        hour = t_obs.hour
        lat, lon = no2.latitude.values[i], no2.longitude.values[i]
        # load averaged winds from hour
        winds_from_hour = interp_wind.isel(time=hour)
        
        for j in range(len(winds_from_hour.wind_speed)):
            # add wind speed and bearing to matching lat/lon/timestamp
            if ((winds_from_hour.lon.values[j] == lon) and
                    (winds_from_hour.lat.values[j] == lat)):
                no2.wind_speed[i] += winds_from_hour.wind_speed.values[j]
                no2.bearing[i] += winds_from_hour.bearing.values[j]
                
    # pickle files
    fdir = winds_pkl + city + '/'
    output_file = os.path.join(fdir, f)
    with open(output_file, 'wb') as outfile:
        print('Pickling %s' % f)
        pickle.dump(no2, outfile)
        
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

if __name__ == '__main__':
    city = 'toronto'
    f = '20200501'
    add_wind(f, city)