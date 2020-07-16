import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import convert as cv
import points_of_interest as poi

city = 'toronto'
res=0.05

# Lat/lon max/min
plot_limits = poi.get_plot_limits(city=city, extent=1, res=res)
lonmn, lonmx, latmn, latmx = plot_limits

# Create a uniform lat/lon grid
lat_bnds = np.arange(latmn, latmx, res)
lon_bnds = np.arange(lonmn, lonmx, res)

lons, lats = np.meshgrid(lon_bnds, lat_bnds)

# arr will accumulate the values within each grid entry
no2_arr = np.random.random([lat_bnds.size, lon_bnds.size])
no2 = no2_arr[:-1, :-1]


fig, ax = plt.subplots(figsize=(12, 10))
fig.tight_layout
# ax = plt.axes(projection=ccrs.PlateCarree())

# im = ax.pcolormesh(lons, lats, no2_arr, transform=ccrs.PlateCarree())

# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                     linewidth=1, color='gray', alpha=0.5, linestyle=':')
# gl.xlabels_top = False
# gl.ylabels_right = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# plt.show()

x = np.zeros_like(lons)
y = np.zeros_like(lats)

lon0, lat0 = poi.cities['toronto']
if lons.shape == lats.shape:
    for i in range(len(lons)):
        for j in range(len(lons[i])):
            x[i][j], y[i][j] = cv.convert_to_cartesian(lat0, lon0, lats[i][j], lons[i][j])

im = ax.pcolormesh(x, y, no2_arr)
plt.show()
