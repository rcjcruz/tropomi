import xarray as xr 
import pickle
import os
import matplotlib.pyplot as plt 

from paths import *
import points_of_interest as poi 

# unpickle files 

def plot(time, data_type, wind_type, city='toronto'):
    
    
    f = '{}/gridded/{}_{}_{}_70'.format(city, poi.nomen_dict[time], poi.nomen_dict[data_type], wind_type )
    if data_type=='rotated':
        f = rotated_pkl + f
    elif data_type=='cartesian':
        f = cartesian_pkl + f
        
    with open(f, 'rb') as infile:
        ds = pickle.load(infile)
        
    ds = ds[wind_type]

    fig, ax = plt.subplots(1, 1)
    im = ax.contourf(ds.x.values, ds.y.values, ds.no2_avg, cmap='jet')
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.scatter(0, 0)
    ax.annotate('toronto', (0, 0))
    plt.show()
    
    return ds

if __name__ == '__main__':
    ds = plot(time='pre-vid', data_type='rotated', wind_type='D')
    print(ds)