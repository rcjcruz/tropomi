import xarray as xr 
import matplotlib.pyplot as plt
import numpy as np 
import pickle
from scipy import optimize 

from paths import *
import plot_wind_tropomi as pwt 
import points_of_interest as poi 


def f(theta, p):
    a, e = p
    return a * (1 - e ** 2) / (1 - e * np.cos(theta))

def residuals(p, r, theta):
    return r - f(theta, p)

#define model function and pass independant variables x and y as a list
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Define 2-D Gaussian.
    
    Reference: https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
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



good_dates = ['20180501_avg', '20180502_avg', '20180504_avg', '20180505_avg',
              '20180507_avg', '20180508_avg', '20180509_avg', '20180513_avg',
              '20180516_avg', '20180517_avg', '20180518_avg', '20180520_avg',
              '20180521_avg', '20180523_avg', '20180525_avg', '20180527_avg',
              '20180528_avg', '20180529_avg', '20180530_avg', '20180531_avg',
              '20190312_avg', '20190317_avg', '20190318_avg', '20190319_avg',
              '20190320_avg', '20190323_avg', '20190325_avg', '20190326_avg',
              '20190327_avg', '20190329_avg', '20190401_avg', '20190403_avg',
              '20190404_avg', '20190406_avg', '20190410_avg', '20190413_avg',
              '20190415_avg', '20190417_avg', '20190422_avg', '20190424_avg',
              '20190425_avg', '20190428_avg', '20190505_avg', '20190506_avg',
              '20190507_avg', '20190508_avg', '20190511_avg', '20190517_avg',
              '20190519_avg', '20190521_avg', '20190523_avg', '20190524_avg',
              '20190525_avg', '20190526_avg', '20190527_avg', '20190531_avg',
              '20190602_avg', '20190603_avg', '20190606_avg', '20190607_avg',
              '20190608_avg', '20190609_avg', '20190611_avg', '20190612_avg',
              '20190614_avg', '20190617_avg', '20190618_avg', '20190621_avg',
              '20190622_avg', '20190623_avg', '20190624_avg', '20190625_avg',
              '20190626_avg', '20190627_avg', '20190628_avg', '20190629_avg',
              '20200301_avg', '20200305_avg', '20200307_avg', '20200308_avg',
              '20200309_avg', '20200315_avg', '20200316_avg', '20200321_avg',
              '20200322_avg', '20200325_avg', '20200327_avg', '20200329_avg',
              '20200401_avg', '20200402_avg', '20200405_avg', '20200406_avg',
              '20200408_avg', '20200409_avg', '20200410_avg', '20200411_avg',
              '20200412_avg', '20200418_avg', '20200420_avg', '20200422_avg',
              '20200423_avg', '20200425_avg', '20200427_avg', '20200428_avg',
              '20200501_avg', '20200503_avg', '20200504_avg', '20200505_avg',
              '20200506_avg', '20200507_avg', '20200513_avg', '20200516_avg',
              '20200519_avg', '20200520_avg', '20200521_avg', '20200522_avg',
              '20200523_avg', '20200524_avg', '20200525_avg', '20200526_avg',
              '20200531_avg', '20200601_avg', '20200602_avg', '20200603_avg',
              '20200604_avg', '20200605_avg', '20200606_avg', '20200607_avg',
              '20200608_avg', '20200609_avg', '20200610_avg', '20200611_avg',
              '20200612_avg', '20200613_avg', '20200614_avg', '20200615_avg',
              '20200616_avg', '20200617_avg', '20200618_avg', '20200619_avg',
              '20200620_avg', '20200621_avg', '20200622_avg', '20200624_avg',
              '20200625_avg', '20200626_avg', '20200627_avg', '20200628_avg',
              '20200629_avg', '20200630_avg']


city = 'toronto'
for date in [good_dates[88]]:
#     f_str = '/' + str(date) + '_avg'
    fpath = winds_pkl + city + '/' + date
    infile = open(fpath, 'rb')
    ds = pickle.load(infile)
    infile.close()
    
    no2 = ds.no2.values
    x = x.values
    y = y.values
    x, y = np.meshgrid(x,y)
    
    new_no2 = no2.where(no2 >= 1/2 * np.nanmax(no2))
    
    
    is_plot = input('[Y/N] Do you want to plot {}?'.format(date))
    if is_plot == 'Y' or 'y':
        pwt.plot_tropomi(ds, wind=True)
        
initial_guess = (3,-40,-40,10,10,0,10)