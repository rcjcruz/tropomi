import os
import glob 
import xarray as xr 
import pickle
import grid_tropomi as gt 
from paths import *
import sys
from pprint import pprint


try:
    fpath = os.path.join(inventories, 'inventory_2020_W19_20.txt')
    files = []
    
    with open(fpath, 'r') as file_list:
        for test_file in file_list:
            date = test_file.strip()
            files.append(date)
    print(files)

except: 
    print('Did not find a text file containing file names (perhaps name does not match)')
    sys.exit()
    
files_dict = {}
for file in files:
    
    pkl_path = os.path.join(tropomi_pkl, file)
    
    with open(pkl_path, 'rb') as infile:
        ds = pickle.load(infile)
        files_dict[file] = ds


# create new dataset
week1920_ds = xr.concat(list(files_dict.values()), dim='time')
pprint(files_dict)