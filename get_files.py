"""
.py script to create inventory.txt file with all the .nc files in 
/export/data/scratch/tropomi/no2/ directory.

"""

# Importing libraries
import glob
import os
import netCDF4 as nc 

# Remove deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Editable variables
outputfile = "/home/rcruz/scripts/inventory.txt" # file to save the results 

if os.path.getsize(outputfile) > 0:
    open(outputfile, 'w').close()

# Open the text file
file_object = open(outputfile, "w+")
# Create an empty list
ncfiles = []

# Iterate over the files in the no2 directory and append them to the list
for file in glob.glob("/export/data/scratch/tropomi/no2/*.nc"):
    # Only add datasets taken on May 5, 2020
    # if file [53:61] == 'YYYYMMDD': # looking for a specific day
    if file[53:59] == "202005": # looking for a specific month
        file_object.write(file + "\n")

file_object.close()