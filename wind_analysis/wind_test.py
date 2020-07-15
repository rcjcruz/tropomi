import matplotlib.pyplot as plt
import numpy as np
import sympy
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())


X = np.array([-1,-1, -2])
Y = np.array([-1,-2, -1])

U = np.array([-1, -1, -1])
V = np.array([-1, -1, 0])

speed = np.sqrt(U**2 + V**2)
bearing = np.degrees(np.arctan2(V, U))
for i in range(len(bearing)):
    if bearing[i] < 0:
        bearing[i] += 360
            
            
def rotate_wind(pivot, point, angle):
    """
    Return xy-coordinate for an initial point (x, y) rotated by angle (in 
    radians) around a pivot (x, y).
    
    Args:
        pivot (tuple of floats): pivot point of rotation.
        point (tuple of floats): xy-coordinates of point to be rotated.
        angle (float): angle to rotate the point around the pivot. 
            Must be in radians.
    Returns:
        (xnew, ynew): xy-coordinates of rotated point.
    """
    # to rotate cw, need negative bearing
    s = sympy.sin(np.radians(-angle))
    c = sympy.cos(np.radians(-angle))
    
    # translate point back to origin
    x = point[0] - pivot[0]
    y = point[1] - pivot[1]
    
    # rotate clockwise to the x-axis
    xnew = x * c - y * s
    ynew = x * s + y * c
    
    # translate point back
    xnew += pivot[0]
    ynew += pivot[1]
    
    return (xnew, ynew)

# X_new = np.zeros_like(X, dtype=float)
# Y_new = np.zeros_like(Y, dtype=float)
# U_new = speed
# V_new = np.zeros_like(V, dtype=float)
# X_new[0], Y_new[0] = rotate_wind((0,0), (-1, -1), 180)
# X_new[1], Y_new[1] = rotate_wind((0,0), (-1, -2), 180)
# X_new[2], Y_new[2] = rotate_wind((0,0), (-2, -1), 180) 
pivot = (0,0)
# rotated xy coordinates
# X_new = np.zeros_like(X, dtype=float)
# Y_new = np.zeros_like(Y, dtype=float)
# U_new = np.zeros_like(U, dtype=float)
# V_new = np.zeros_like(V, dtype=float) # should always be 0 
# for i in range(len(X)):
#     for j in range(len(Y)):
#         point = (X[i][j], Y[i][j])
#         angle = bearing[i][j]
#         X_new[i][j], Y_new[i][j] = rotate_wind(pivot, point, angle)
#         U_new[i][j] = np.sqrt((X[i][j]) ** 2 + (Y[i][j]) ** 2)

X_new = np.zeros_like(X, dtype=float)
Y_new = np.zeros_like(Y, dtype=float)
U_new = np.zeros_like(U, dtype=float)
V_new = np.zeros_like(V, dtype=float) # should always be 0 
for i in range(len(X)):
    point = (X[i], Y[i])
    angle = bearing[i]
    X_new[i], Y_new[i] = rotate_wind(pivot, point, angle)
    U_new[i] = np.sqrt((X[i]) ** 2 + (Y[i]) ** 2)
        
# # point = (U[0][0], V[0][0])
# # pivot = (0,0)
# # angle = bearing[0][0]
# # new_point = rotate_wind(pivot, point, angle)

fig, (ax1, ax2) = plt.subplots(1,2)
q = ax1.quiver(X, Y, U, V)
# ax1.quiverkey(q, X=0.3, Y=1.1, U=10,
#              label='Quiver key, length = 10', labelpos='E')
q2 = ax2.quiver(X_new, Y_new, U_new, V_new)
plt.show()

# fig, ax = plt.subplots()
# # q = ax.quiver(X, Y, U, V)
# # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
# #              label='Quiver key, length = 10', labelpos='E')
# q2 = ax.quiver(X_new, Y_new, U_new, V_new)
# plt.show()