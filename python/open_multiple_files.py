import open_tropomi as ot 
import get_files as gf 
import numpy as np 
import xarray as xr 
import pandas as pd 
import time

# VARIABLES 
year = 2020
# month = 5
# day = 5
week_num = 17

##################
# Load files
start, end, calendar_week = gf.get_files(year=year, calendar_week=week_num)
 
try:
    file_list=open('inventory.txt','r')
except:
    print('Did not find a text file containing file names (perhaps name does not match)')
    sys.exit()

ds_list = []

startiest_time = time.time() 

for test_file in file_list:
    test_file = test_file.strip()
    start_time = time.time()
    ds_list.append(ot.dsread(test_file))
    print("--- %s seconds ---" % (time.time() - start_time))
print('Total time: %s', (time.time() - startiest_time))
print(len(ds_list))






# date_of_interest = '20200505'
# ds_list = []
# d = pd.to_datetime(ds[0].time.data).week

    
