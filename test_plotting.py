import new_code as nc
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


fig, ax = plt.subplots(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

#this is 1D so it plots a line
im = nc.ds['nitrogendioxide_tropospheric_column'].scanline.plot.pcolormesh(ax=ax,
                           transform=ccrs.PlateCarree(),
                           infer_intervals=True,
                           cmap='jet',
                           norm=LogNorm(vmin=10e-6),
                           x='longitude',
                           y='latitude',
                           zorder=0,
                           add_colorbar=False)

plt.show()
