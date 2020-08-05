import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt 


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

# Create x and y indices
x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
x, y = np.meshgrid(x, y) # make a mesh grid of the coordinates 

# create data
data = twoD_Gaussian((x, y), amplitude=3, xo=100, yo=100, sigma_x=20, sigma_y=40, theta=0, offset=10)

# # plot twoD_Gaussian data generated above
# plt.figure()
# plt.imshow(data.reshape(201, 201))
# plt.colorbar()
# plt.show()

# add some noise to the data and try to fit the data generated beforehand
initial_guess = (3,100,100,20,40,0,10)

data_noisy = data + 0.2*np.random.normal(size=data.shape)

popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)

data_fitted = twoD_Gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
# ax.hold(True)
ax.pcolormesh(x, y, data_noisy.reshape(201, 201), cmap=jet)
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
plt.show()