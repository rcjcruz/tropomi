import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy import stats
from scipy.optimize import curve_fit    

from paths import *
import points_of_interest as poi
import grid_pixels_weighted as gpw

# Bin 2 - 2-4m/s for COVID-19 (March 2020 - June 2020)
bwinds = [20200305, 20200327, 20200407, 20200420, 20200424, 20200427, 20200505, 
          20200513, 20200516, 20200608, 20200609, 20200615, 20200616, 20200617, 
          20200618]

#define model function and pass independent variables x and y as a list
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Define 2-D Gaussian function for curve_fit.
    
    Reference: https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
    """
    
    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()

def fit_gauss(date, res, dist):
    """
    Fit 2-D Gaussian and return angle theta. 
    
    Args: 
    
    """
    # load averaged Cartesian data
    ds = gpw.grid_weighted('20200327', data_type='cartesian', res=5, dist=70, avg=True)
    x = ds.x.values
    y = ds.y.values
    x, y = np.meshgrid(x,y)
    no2 = ds.no2_avg

# if __name__ == '__main__':
date='20200327'
ds = gpw.grid_weighted('20200327', data_type='cartesian', res=5, dist=70, avg=True)
x = ds.x.values
y = ds.y.values
x, y = np.meshgrid(x,y)
no2 = ds.no2_avg

# im = ax.pcolormesh(ds.x.values, ds.y.values, ds.no2_avg, cmap='Blues')
new_no2 = no2.where(no2>= 0 * np.nanmax(no2))
new_no2 = new_no2.fillna(0).values.flatten()
# fig, ax = plt.subplots(1, 1)
# im = ax.pcolormesh(ds.x.values, ds.y.values, new_no2, cmap='Blues')


initial_guess = (3,0,0,10,10,0,10)

popt, pcov = curve_fit(twoD_Gaussian, (x, y), new_no2, p0=initial_guess)

gauss_angle = np.degrees(popt[5]) % 360
data_fitted = twoD_Gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
# plt.axis('off')
im = ax.pcolormesh(x, y, new_no2.reshape(x.shape), cmap=plt.cm.Blues)
# ax.set_title(pd.to_datetime(date, format='%Y%m%d'))
# ax.set_xlabel('x (km)')
# ax.set_ylabel('y (km)')
cs = ax.contour(x, y, data_fitted.reshape(x.shape), 5, colors='white')
plt.tight_layout()
plt.savefig('20200327.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# cb = create_colorbar(im=im, ax=ax, label='Mean NO$_2$ TVCD', orientation='vertical')
# cb.ax.yaxis.get_offset_text().set_visible(False)
# ax.set_xlabel(r'$\longleftarrow$ upwind      x (km)      downwind $\longrightarrow$')
# ax.set_ylabel(r'y (km) $\longrightarrow$')
# ax.scatter(0, 0, marker='*', color='black')
# plt.show()

#     alphabet_string = string.ascii_uppercase
#     alphabet_list = list(alphabet_string)

#     time = 'covid'
#     # for letter in alphabet_string[:9]:
#     # wind_dicts = average(time, data_type='rotated', dist=100, wind_type='G')