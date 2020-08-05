import example as ex
from ellipse import LsqEllipse 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import xarray as xr 
import grid_pixels_weighted as gpw 

ds = gpw.grid_weighted('20200401', data_type='cartesian', res=5, dist=70, avg=True)
x = ds.x.values
y = ds.y.values
# x, y = np.meshgrid(x,y)
no2 = ds.no2_avg

# im = ax.pcolormesh(ds.x.values, ds.y.values, ds.no2_avg, cmap='Blues')
new_no2 = no2.where(no2>= 1/2 * np.nanmax(no2), drop=True)

# X1, X2 = ex.make_test_ellipse()

# X = np.array(list(zip(X1, X2)))
# reg = LsqEllipse().fit(X)
# center, width, height, phi = reg.as_parameters()

# plt.close('all')
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# ax.axis('equal')
# ax.plot(X1, X2, 'ro', label='test data', zorder=1)

# ellipse = Ellipse(xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
#                edgecolor='b', fc='None', lw=2, label='Fit', zorder = 2)
# ax.add_patch(ellipse)

# plt.legend()
# plt.show()