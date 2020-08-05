import math
import numpy as np
import xarray as xr 
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl

import points_of_interest as poi 
from paths import *


def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k == 9:
        return 3, 3
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3


def generate_subplots(k, row_wise=False):
    nrow, ncol = choose_subplot_dimensions(k)
    
    figure, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False,
                                figsize=(15,13))
    
    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for ax in axes[k:]:
            figure.delaxes(ax)
            
        axes = axes[:k]
        
        return figure, axes
    

time = 'pre-vid'
data_type='rotated'
wind_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


figure, axes = generate_subplots(len(wind_types), row_wise=True)
figure.suptitle('2019-03-13 to 2019-06-30')

for i, (wt, ax) in enumerate(zip(wind_types, axes)):
    f = 'toronto/gridded/{}_{}_{}_70'.format(poi.nomen_dict[time], poi.nomen_dict[data_type], wt)
    if data_type == 'rotated':
        f = rotated_pkl + f
    elif data_type == 'cartesian':
        f = cartesian_pkl + f

    with open(f, 'rb') as infile:
        ds = pickle.load(infile)
        ds = ds[wt]
        
    # get wind speed
    ws1 = poi.wind_type[wt][0]
    ws2 = poi.wind_type[wt][1]
    
    im = ax.contourf(ds.x.values, ds.y.values, ds.no2_avg, cmap='jet')
    ax.set_aspect('equal')
    ax.scatter(0,0, marker='*', color='black')
    ax.annotate('{}-{}m/s'.format(ws1, ws2),
                xy=(0.05,0.9), xycoords='axes fraction')
    
    if i == 0 or i == 3 or i == 6:
        ax.set_ylabel('y (km)')
    if i == 6 or i == 7 or i == 8:
        ax.set_xlabel(r'$\leftarrow$ upwind        x (km)     downwind $\rightarrow$')

for ax in axes:
    ax.label_outer()
    
figure.subplots_adjust(wspace=0, hspace=0)
cbar_ax = figure.add_axes([0.9, 0.15, 0.01, 0.7])
figure.colorbar(im, cax=cbar_ax)
plt.tight_layout()
plt.show()