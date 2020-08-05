import matplotlib.pyplot as plt 
import numpy as np 


x=np.arange(0,5)
y=np.arange(0,5)
yy, xx = np.meshgrid(y,x)
z=np.random.rand(5,5)
plt.pcolormesh(xx, yy, z)
plt.show()