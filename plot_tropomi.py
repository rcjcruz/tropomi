# Preamble
import numpy as np
import xarray as xr
import netCDF4 as nc
from glob import iglob
from os.path import join
from collections import namedtuple
import tropomi_functions as tf

# Variables
file_name = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc'
sds_name = 'nitrogendioxide_tropospheric_column'

# Define plot extent centred around Toronto
Point = namedtuple('Point', 'lon lat')
toronto_coords = Point(-79.3832, 43.6532)

extent_size = 5
plot_limits = (toronto_coords.lon-extent_size,
               toronto_coords.lon+extent_size,
               toronto_coords.lat-extent_size,
               toronto_coords.lat+extent_size)
# print("Plot extent: lon({}, {}), lat({}, {})".format(*plot_limits))


with xr.open_dataset(file_name, group='/PRODUCT')[sds_name] as no2tc:
    # Reading the data
    fields, short_file_name = tf.read_data(file_name)
    
    # Print information about orbit 
    print("Orbit: {}, Sensing Start: {}, Sensing Stop: {}".format(fields[10],
                                                              fields[8],
                                                              fields[9]))

    # Plot orbit on the globe or on Toronto
    # If Toronto, append plot_limits to plot_no2 args
    tf.plot_no2(no2tc, 'globe', fields, short_file_name)
