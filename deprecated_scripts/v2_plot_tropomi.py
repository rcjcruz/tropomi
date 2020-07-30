# Preamble
import numpy as np
import xarray as xr
import netCDF4 as nc
import glob
from os.path import join
from collections import namedtuple
import tropomi_functions as tf

# Variables
file_name = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200503T175306_20200503T193436_13242_01_010302_20200505T104402.nc'

xno2_path = '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/'
sds_name = 'nitrogendioxide_tropospheric_column'
total_sds_name = 'nitrogendioxide_total_column'

# print("Plot extent: lon({}, {}), lat({}, {})".format(*plot_limits))


# with xr.open_dataset(file_name, group=xno2_path)[total_sds_name] as xno2:
# Reading the data

# NO FILTER
# no2 = xr.open_dataset(file_name, group='/PRODUCT')[sds_name]

# WITH FILTER
ds = xr.open_dataset(file_name, group='/PRODUCT')
subset_ds = ds.where(ds['qa_value'] > 0.75, drop=True)
no2 = subset_ds[sds_name]

fields, short_file_name = tf.read_data(file_name)

# Print information about orbit
print("Orbit: {}, Sensing Start: {}, Sensing Stop: {}".format(fields[10],
                                                              fields[8],
                                                              fields[9]))

# Plot orbit on the globe or on Toronto
# If Toronto, append plot_limits to plot_no2 args
tf.plot_no2(no2, 'toronto', fields, short_file_name)
