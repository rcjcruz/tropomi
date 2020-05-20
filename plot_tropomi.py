### Importing libraries
import glob
import netCDF4 as nc 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import numpy as np 
import numpy.ma as ma
import sys

# Remove deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# #This finds the user's current path so that all hdf4 files can be found
# try:
#     # inventory.txt must be in the same directory as plot_tropomi.py
#     fileList=open('inventory.txt','r')
# except:
#     print('Did not find a text file containing file names (perhaps name does not match)')
#     sys.exit()

# # Create an empty list
# ncfiles = []

# # Iterate over the files in the no2 directory and append them to the list
# for file in glob.glob("/export/data/scratch/tropomi/no2/*.nc"):
#     ncfiles.append(file)

# my_example_nc_file = ncfiles[1]

#This finds the user's current path so that all hdf4 files can be found
try:
    fileList=open('inventory.txt','r')
except:
    print('Did not find a text file containing file names (perhaps name does not match)')
    sys.exit()

# test_file = '/export/data/scratch/tropomi/no2/S5P_OFFL_L2__NO2____20200505T171512_20200505T185642_13270_01_010302_20200507T092201.nc'
for test_file in fileList:
    test_file=test_file.strip()
    short_file_name=test_file[33:]
    user_input=input('\nWould you like to create a map of\n' + test_file + '\n\n(Y/N/exit)')
    if(user_input == 'N' or user_input == 'n'):
        print('Skipping...')
    elif(user_input == 'exit'):
        sys.exit('The script has terminated at the user\'s request.')
    else:
        file = nc.Dataset(test_file, 'r')
        # read the data
        ds = file
        grp = 'PRODUCT'
        lons = ds.groups[grp].variables['longitude'][0][:][:] # longitude
        lats = ds.groups[grp].variables['latitude'][0][:][:] # latitude
        sds_name='nitrogendioxide_tropospheric_column'
        map_label='mol/m2'
        data = ds.groups[grp].variables[sds_name][0,:,:]

        # ### FROM THE TUTORIAL ONLINE -- Stereographic Projection
        # # Get some parameters for the Stereographic Projection
        # lon_0 = lons.mean()
        # lat_0 = lats.mean()
        # # m = Basemap(width=5000000,height=3500000,
        # #             resolution='l',projection='stere',\
        # #             lat_ts=40,lat_0=lat_0,lon_0=lon_0)

        # FROM READ_AND_MAP_TROPOMI_NO2_AI.PY -- cylindrical projection
        m = Basemap(projection='cyl', resolution='l',
                    llcrnrlat=-90, urcrnrlat = 90,
                    llcrnrlon=-180, urcrnrlon = 180)

        xi, yi = m(lons, lats)
        map_title = 'Tropospheric vertical column of nitrogen dioxide'

        # Choose plot colour
        # Note: values are logarithmically scaled
        cs = m.pcolor(xi,yi,np.squeeze(data),
                      norm=LogNorm(vmin=10e-11, 
                                   vmax=10e-3), 
                      cmap='jet')

        # Add Grid Lines
        m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-180, 180., 45.), labels=[0, 0, 0, 1])

        # Add Coastlines and Country Boundaries
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()

        # Add Colorbar
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label("mol/$\mathdefault{m^2}$")

        # Add Title
        plt.title('{0}\n {1}'.format(short_file_name, map_title))
        plt.autoscale()
        fig = plt.gcf()
        # Show the plot window
        plt.show()
        is_save=str(input('\nWould you like to save this map? Please enter Y or N \n'))
        if is_save == 'Y' or is_save == 'y':
            # Saves plot as a png in the tropomi_figures folder
            pngfile = '{0}.png'.format('tropomi_figures/' + short_file_name[:-3])
            fig.savefig(pngfile, dpi = 300)
        #Close the hdf4 file
        file.close()