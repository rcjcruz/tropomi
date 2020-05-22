#Preamble
import tropomi_functions as tf
import xarray as xr
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.axes import Axes

from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

# Variables
file_name = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc'
sds_name = 'nitrogendioxide_tropospheric_column'

# Script
no2 = xr.open_dataset(file_name, group='/PRODUCT')[sds_name]

# Plotting
ax = plt.axes(projection=ccrs.PlateCarree())

# set all negative value to 0
no2 = no2.where(no2 > 0, 0)
no2 = tf.subset(no2)

no2.isel(time=0).plot.pcolormesh(ax=ax,
                                 x='longitude',
                                 y='latitude',
                                 transform=ccrs.PlateCarree(),
                                 infer_intervals=True,
                                 cmap='jet',
                                 norm=LogNorm(vmin=10e-6))
ax.set_global()
ax.coastlines()
plt.show()
