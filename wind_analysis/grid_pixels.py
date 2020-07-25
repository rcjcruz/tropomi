import xarray as xr
import numpy as np
import shapely.geometry as geometry
import pickle
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from paths import *
import points_of_interest as poi

wind_type = {'A': [0, 2], 'B': [2, 3], 'C': [3, 4], 'D': [4, 5], 'E': [5, 6],
             'F': [6, 7], 'G': [7, 9]}
dates_dict = {'may': ['201805*', '201905*', '202005*'],
              'march': ['201903*', '202003*'],
              'april': ['201904*', '202004*'],
              'june': ['201906*', '202006*'],
              'pre-vid': ['2019*'],
              'covid': ['2020*']}

# plan:
# - load a dict key (e.g. May)
# - store all the datasets for those months in separate lists so they're all 
#   together
# - sort by wind speeds using wind_type and further separate into wind speeds 
#   using the categorization of month or season

def grid_tropomi(date, data_type='cartesian', city='toronto', res=1., plot=True):
    """

    Args:
        date (str): date of observation. Format: YYYYMMDD
        data_type (str): 'cartesian' or 'rotated'
        city (str): city of interest. Default: 'toronto'
        res (float): resolution of grid. Default: 1 km.
        plot (bool): plot gridded TROPOMI data. Default: True.

    Returns:

    """
    ### ONLY FOR ONE DAY, NEED TO CHANGE TO AVERAGE OVER MULTIPLE DAYS
    # open file cartesian_pkl for cartesian and rotated_pkl for rotated
    f_str = city + '/' + date

    if data_type == 'cartesian':
        fpath = os.path.join(cartesian_pkl, f_str)
    elif data_type == 'rotated':
        fpath = os.path.join(rotated_pkl, f_str)
    else:
        return ValueError('Invalid data type. Must be "cartesian" or "rotated"')

    infile = open(fpath, 'rb')
    ds = pickle.load(infile)
    infile.close()
    print('Opened', fpath)

    # load no2, error, coordinates, and bounds
    date=pd.to_datetime(ds.measurement_time.values).date()
    no2 = ds.no2.values
    er = ds.no2_error.values
    x, y = ds.x.values, ds.y.values
    xbd, ybd = ds.x_bounds.values, ds.y_bounds.values

    # create grids with
    x_grid = np.arange(-70, 70+res, res, dtype=int)
    y_grid = np.arange(-70, 70+res, res, dtype=int)
    no2_grid = np.zeros([y_grid.size, x_grid.size])
    error_grid = np.zeros([y_grid.size, x_grid.size])
    weight_tot = np.zeros([y_grid.size, x_grid.size])

    # Check if the lat and lon values are found within lat/lon bounds
    y_flt = (y > min(y_grid)) * (y < max(y_grid))
    x_flt = (x > min(x_grid)) * (x < max(y_grid))

    # Create array to filter data points found within lat/lon bounds
    filter_arr = y_flt * x_flt

    # Keep no2 values that are within the bounded lat/lon
    no2 = no2[filter_arr]
    er = er[filter_arr]
    x = x[filter_arr]
    y = y[filter_arr]
    xbd = [xbd[0][filter_arr], xbd[1][filter_arr], xbd[2][filter_arr], xbd[3][filter_arr]]
    ybd = [ybd[0][filter_arr], ybd[1][filter_arr], ybd[2][filter_arr], ybd[3][filter_arr]]

    print('... Searching ...')
    print('TOTAL:', len(no2))
    for k in range(len(no2)):
        print('Reading scanline', k)
        # define the polygon
        points = [geometry.Point(xbd[0][k], ybd[0][k]),
                  geometry.Point(xbd[1][k], ybd[1][k]),
                  geometry.Point(xbd[2][k], ybd[2][k]),
                  geometry.Point(xbd[3][k], ybd[3][k])]
        poly = geometry.Polygon([[p.x, p.y] for p in points])
        footprint = poly.area  # determine the polygon area
        print('Created polygon')
        for i in range(x_grid.size):  # for each x in the grid
            for j in range(y_grid.size):  # for each y in the grid
                # if the polygon contains at least one corner of the grid pixel
                if poly.contains(geometry.Point(x_grid[i]+res/2, y_grid[j]+res/2)) \
                        or poly.contains(geometry.Point(x_grid[i]-res/2, y_grid[j]+res/2)) \
                        or poly.contains(geometry.Point(x_grid[i]+res/2, y_grid[j]-res/2)) \
                        or poly.contains(geometry.Point(x_grid[i]-res/2, y_grid[j]-res/2)):

                    # create a polygon representing the pixel
                    pixel = [geometry.Point(x_grid[i]+res/2, y_grid[j]+res/2),
                             geometry.Point(x_grid[i]-res/2, y_grid[j]+res/2),
                             geometry.Point(x_grid[i]-res/2, y_grid[j]-res/2),
                             geometry.Point(x_grid[i]+res/2, y_grid[j]-res/2)]
                    pixel_poly = geometry.Polygon([[p.x, p.y] for p in pixel])

                    # if the entire pixel is contained in the TROPOMI footprint:
                    if (poly.contains(geometry.Point(x_grid[i]+res/2, y_grid[j]+res/2))
                        and poly.contains(geometry.Point(x_grid[i]-res/2, y_grid[j]+res/2))
                        and poly.contains(geometry.Point(x_grid[i]+res/2, y_grid[j]-res/2))
                        and poly.contains(geometry.Point(x_grid[i]-res/2, y_grid[j]-res/2))):

                        # # weight that pixel by the area of the pixel /
                        # # area of the polygon  * NO2 measurement error ** 2
                        weight = pixel_poly.area / ((er[k] ** 2) * footprint)

                    else:
                        intersect = pixel_poly.intersection(poly)
                        weight = intersect.area / ((er[k] ** 2) * footprint)
                        # weight is proportional to the overlapping area
                        # between the footprint and pixel

                    # add all the overlapping NO2 values to that pixel's value
                    # and multiply by its weight
                    no2_val = no2[k] * weight
                    no2_grid[j, i] += no2_val
                    error_grid[j, i] += weight
                    weight_tot[j, i] += weight
                    # print('ADDED NO2:{} and WEIGHT: {} to [{},{}]'.format(no2_val, weight, j, i))

    print('... Averaging ...')
    for n in range(len(weight_tot)):
        for m in range(len(weight_tot[n])):
            if weight_tot[n, m] == 0:  # remove divide by zero cases
                no2_grid[n, m] = None
                weight_tot[n, m] = None
                error_grid[n, m] = None
            else:
                no2_grid[n, m] = no2_grid[n, m] / weight_tot[n, m]
                error_grid[n, m] = 1 / np.sqrt(error_grid[n, m])
                
    # create dataset of values

    if plot:
        print('... Plotting ...')
        fig, ax = plt.subplots()
        ax.pcolormesh(x_grid, y_grid, no2_grid, cmap='Blues')
        plt.scatter(0,0)
        plt.annotate('toronto', (0,0))
        plt.show()
    
    return x_grid, y_grid, no2_grid, error_grid, date

if __name__ == '__main__':
    x_grid, y_grid, no2, err, date = grid_tropomi('20200502', data_type='rotated')

ds = xr.Dataset({'x_coords': xr.DataArray(x_grid, dims=['x'], coords=[x_grid],attrs={'description': 'distance from origin in along x-axis', 'units': 'km'}), 
                 'y_coords': xr.DataArray(y_grid, dims=['y'], coords=[y_grid],attrs={'description': 'distance from origin in along y-axis', 'units': 'km'}),
                 'no2_avg': xr.DataArray(np.array([no2]), dims=['time', 'y', 'x'], coords=[np.array([date]), y_grid, x_grid], attrs={'description': 'NO2 tropospheric vertical column weighted average', 'units': 'mol m-2'}),
                 'no2_avg_error': xr.DataArray(np.array([err]), dims=['time', 'y', 'x'], coords=[np.array([date]), y_grid, x_grid], attrs={'description': 'NO2 tropospheric vertical column weight average error', 'units': 'mol m-2'})},
                attrs={'time': date})