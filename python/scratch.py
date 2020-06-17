from paths import *
import os
from collections import namedtuple
import time


def get_city_files(f, city='toronto', extent=1, append_new=True):
    """
    Return .txt file containing .nc files with orbits over city given 
    file f. If .txt file exists, load append_new=True to append new files.

    Args:
        f (str): file name of TROPOMI NO2 .nc files.
        append_new (bool): if True, only append new files not found in existing
            .txt file. Default: False.
    """

    # Load path to NO2 files
    fdir = tropomi_no2
    fpath = os.path.join(fdir, f)

    # Load coordinates of city
    Point = namedtuple('Point', 'lon lat')
    cities = {'toronto': Point(-79.3832, 43.6532),
              'montreal': Point(-73.5673, 45.5017),
              'new_york': Point(-74.0060, 40.7128),
              'vancouver': Point(-123.1207, 49.2827),
              'los_angeles': Point(-118.2437, 34.0522)}

    if city not in cities.keys():
        return TypeError('Invalid city. City must be %s' % list(cities.keys()))

    else:
        extent = 5
        city_coords = cities[city]
        plot_limits = (city_coords.lon-extent,
                       city_coords.lon+extent,
                       city_coords.lat-extent,
                       city_coords.lat+extent)
        e, w, s, n = plot_limits

    # Load output file
    output_file = '{}/{}_inventory.txt'.format(city, city)
    output_fpath = os.path.join(inventories, output_file)

    # Open the text file
    file_object = open(output_fpath, "r+")

    # Keep track of start time of proeess
    start_time = time.time()

    # Iterate over all files in no2 directory
    files = sorted(glob.glob(fpath))

    if append_new:
        if 'OFFL' not in f:
            return('f must include \'OFFL\'')
        # Load text from toronto_inventory.txt
        text = file_object.readlines()

        # Obtain last date of observations already catalogued; only use
        # files with 'OFFL' because 'RPRO' stopped in 2019
        offl_files = []
        for file in text:
            if 'OFFL' in file:
                offl_files.append(file)
        last_date = offl_files[-1][53:61]

        j = 1
        # Check at the bottom of the list and append if date of file is 
        # greater than the last added file to the inventory
        for i in reversed(range(len(files))):
            date_of_obs = files[i][53:61]
            if date_of_obs > last_date:
                with xr.open_dataset(
                        files[i], group='/PRODUCT')['nitrogendioxide_tropospheric_column'] as ds:
                    # Keep track of start time of iteration
                    start_time_iter = time.time()

                    # Check if ds contains values over Toronto
                    extracted = ds.where(
                        (ds.longitude > e) &
                        (ds.longitude < w) &
                        (ds.latitude > s) &
                        (ds.latitude < n), drop=True)

                    # If extracted data is not empty, write the file name to
                    # the output_file
                    if len(extracted.data) != 0:
                        print('[{}] {} includes an orbit over {}'.format(j, files[i], city))
                        file_object.writelines([files[i], '\n'])

                    else:
                        print('[{}] {} does not include an orbit over {}'.format(j, files[i], city))
                   
                    print("--- %s seconds ---" % (time.time() - start_time_iter))
                    i += 1
    else:
        j = 1
        for i in range(len(files)):
            with xr.open_dataset(
                    file, group='/PRODUCT')['nitrogendioxide_tropospheric_column'] as ds:
                # Keep track of start time of iteration
                start_time_iter = time.time()

                # Check if ds contains values over Toronto
                extracted = ds.where(
                    (ds.longitude > e) &
                    (ds.longitude < w) &
                    (ds.latitude > s) &
                    (ds.latitude < n), drop=True)

                # If extract_toronto data is not empty, write the file name to
                # the output_file
                if len(extracted.data) != 0:
                    print('[{}] {} includes an orbit over {}'.format(j, files[i], city))
                    file_object.writelines([files[i], '\n'])

                else:
                    print('[{}] {} does not include an orbit over {}'.format(j, files[i], city))
                
                print("--- %s seconds ---" % (time.time() - start_time_iter))
                i += 1

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time elapsed:{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))
