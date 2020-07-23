import xarray as xr
import numpy as np
import glob
import pickle

from paths import *
import points_of_interest as poi
import create_xy_datasets as cxy

def average_data(dslist, city='toronto', res=1):
    """
    ds must be a list of datasets
    """
    # Lat/lon max/min
    xmn, xmx, ymn, ymx = -100, 100, -100, 100

    # Create a uniform lat/lon grid
    x_bnds = np.arange(xmn, xmx+res, res)
    y_bnds = np.arange(ymn, ymx+res, res)

    # arr will accumulate the values within each grid entry
    no2_arr = np.zeros([y_bnds.size, x_bnds.size])
    # dens_arr will count the number of observations that occur within that grid entry
    dens_arr = np.zeros([y_bnds.size, x_bnds.size], dtype=np.int32)

    for ds in dslist:
        no2 = np.array(ds.no2[0])
        x_coords = np.array(ds.x_coords)
        y_coords = np.array(ds.y_coords)

        x_flt = (x_coords > xmn) * (x_coords < xmx)
        y_flt = (y_coords > ymn) * (y_coords < ymx)

        filter_arr = y_flt * x_flt
        no2 = no2[filter_arr]
        x_coords = x_coords[filter_arr]
        y_coords = y_coords[filter_arr]

        for i in range(no2.size):
            # Find the indices in the lat/lon_bnds grid at which the
            # max/min lat/lon would fit (i.e. finding the grid squares that the data
            # point has values for)
            x_inds = np.searchsorted(x_bnds, x_coords[i])
            y_inds = np.searchsorted(y_bnds, y_coords[i])

            # Add the NO2 values that fit in those lat/lon grid squares to val_arr and
            # add 1 to dens_arr to increase the count of observations found in that
            # grid square
            no2_arr[y_inds, x_inds] += no2[i]
            dens_arr[y_inds, x_inds] += 1

    no2_arr_mean = np.divide(no2_arr, dens_arr, out=(
        np.zeros_like(no2_arr)), where=(dens_arr != 0))

    rotated_ds = xr.DataArray(np.array([no2_arr_mean]),
                              dims=('time', 'y', 'x'),
                              coords={'time': np.array(['May 2020']),
                                      'y': y_bnds,
                                      'x': x_bnds})

    return rotated_ds


if __name__ == '__main__':
    dslist = []

    city = 'toronto'
    f = '202005'
    f_str = '/' + f + '*_avg'
    fpath = winds_pkl + city + f_str
    for file in glob.glob(fpath):
        print(file)
        infile = open(file, 'rb')
        ds = pickle.load(infile)
        infile.close()
        cartesian_ds, rotated_ds = cxy.create_xy_dataset(
            ds, 'toronto', rotate_coords=True)
        dslist.append(rotated_ds)

    avg = average_data(dslist)