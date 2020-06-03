import numpy as np
import xarray as xr
import netCDF4 as nc
from glob import iglob
from os.path import join
from collections import namedtuple

file_name = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc'
sds_name = 'nitrogendioxide_tropospheric_column'
total_sds_name = 'nitrogendioxide_total_column'

xno2 = xr.open_dataset(file_name, group='/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/')['nitrogendioxide_total_column']
no2 = xr.open_dataset(file_name, group='/PRODUCT')[sds_name]

xno2['latitude'] = no2['latitude']