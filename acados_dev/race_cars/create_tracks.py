from tracks.readDataFcn import getTrack
from plotFcn import plotTrackProj

from matplotlib import pyplot as plt
import numpy as np

import casadi as ca
import scipy.interpolate as spi

trajectories = np.array([[[0, 0, 0, 0, 0, 0]]])
trajectories = np.tile(trajectories, (100,1))
# plotTrackProj(trajectories, 'LMS_Track.txt')
# plt.show()



datafile = "LMS_Track2.txt"
[s, x, y, psi, kappa] = getTrack(datafile)

while y[-1] > -1.65:
    x= np.append(x, x[-1])
    y= np.append(y, y[-1]-0.02)
    psi = np.append(psi, np.arctan2(y[-1]-y[-2], x[-1]-x[-2]))
    s = np.hypot

# plt.subplot(2, 1, 1)
# plt.plot(x, y)
# plt.subplot(2, 1, 2)
# plt.plot(s, psi, kappa)


def rounded_rectangle(w, h, r, num_points_corner=20):
    """
    Generate x, y waypoints for a rounded rectangle with the center of the top edge at (0, 0).

    Parameters:
        w (float): Width of the rectangle.
        h (float): Height of the rectangle.
        r (float): Radius of the corners (must satisfy r <= min(w, h)/2).
        num_points_corner (int): Number of points per quarter-circle corner.

    Returns:
        list: A list of (x, y) waypoints tracing the rounded rectangle.
    """
    if r > min(w, h) / 2:
        raise ValueError("Corner radius 'r' cannot be larger than half the rectangle's width or height.")
    
    # List to store (x, y) points
    points = []
    psi = []
    curvature = []
    
    # Shift the rectangle down by h/4 so top edge is at y = 0
    y_shift = -h / 4 
    
    # Define corner centers
    corners = [
        (w / 2 - r, y_shift + r),   # Top-right corner
        (w / 2 - r, y_shift - h + r),  # Top-left corner
        (-w / 2 + r, y_shift - h + r), # Bottom-left corner
        (-w / 2 + r, y_shift + r),   # Bottom-right corner
    ]
    
    # Define angles for quarter circles
    angles = [
        (0, -np.pi / 2),         # Top-right corner
        ( -np.pi / 2, -np.pi ),    # Top-left corner
        (np.pi, np.pi / 2), # Bottom-left corner
        (np.pi / 2, 0), # Bottom-right corner
    ]
    
    # Top edge center at (0,0)
    points.append((0, 0))
    
    # Top edge to the right
    points.extend([(x, 0.0 ) for x in np.linspace(0, w / 2 - r, num_points_corner, endpoint=False)])
    psi.extend([0]*num_points_corner)
    curvature.extend([0]*num_points_corner)

    # Generate points for each quarter circle and straight edges
    for i, (center, angle) in enumerate(zip(corners, angles)):
        # Generate quarter circle arc
        theta = np.linspace(angle[0], angle[1], num_points_corner)
        x_arc = center[0] - r * np.sin(theta) 
        y_arc = center[1] + r * np.cos(theta)
        for thetai in theta:
            if thetai > np.pi:
                psi.append(thetai - np.pi)
            else:
                psi.append(thetai)
        points.extend(zip(x_arc, y_arc))
        curvature.extend([-1/r]*num_points_corner)
        
        # Add straight edges (except after the last quarter circle)
        if i == 0:  # Right edge
            points.extend([(w / 2 , y) for y in np.linspace(y_shift + r, y_shift - h + r, num_points_corner, endpoint=False)])
            psi.extend([-np.pi/2]*num_points_corner)
            
        elif i == 1:  # Bottom edge
            points.extend([(x, y_shift - h ) for x in np.linspace(-w / 2 + r, w / 2 - r, num_points_corner, endpoint=False)][::-1])
            psi.extend([np.pi]*num_points_corner)
        elif i == 2:  # Left edge
            points.extend([(-w / 2 , y) for y in np.linspace(y_shift - h + r, y_shift + r, num_points_corner, endpoint=False)])
            psi.extend([np.pi/2]*num_points_corner)
        curvature.extend([0]*num_points_corner)
    points.extend([(x, 0.0) for x in np.linspace(-w / 2 +r , 0, num_points_corner, endpoint=False)])  # Top edge to the left
    psi.extend([0]*num_points_corner)
    x = [point[0] for point in points]
    y = [point[1] for point in points]
 

    # organize points in counter-clockwise order with 0,0 at beginning
    return x, y, psi, curvature



w = 2.5  # Width
h = 2 # Height
r = 0.25  # Corner radius
x,y,psi,curvature = rounded_rectangle(w, h, r, num_points_corner=100)
s = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
x = x[:-1]
y = y[:-1]

# Plot the rounded rectangle
plt.figure(figsize=(8, 6))
plt.plot(x, marker=".", linestyle="-", label="x")
plt.plot(y, marker=".", linestyle="-", label="y")
plt.title("Rounded Rectangle")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

plt.figure(figsize=(8, 6))
plt.plot(s, np.mod(psi, 2*np.pi), marker=".", linestyle="-")
plt.plot(s, curvature)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Rounded Rectangle")
plt.xlabel("s")
plt.grid()
plt.show()

# np.savetxt("acados_dev/race_cars/tracks/LMS_Track6.txt", np.column_stack((s, x, y, psi, curvature)), delimiter=" ", fmt="%s")
# plotTrackProj(trajectories, 'LMS_Track5.txt')
# plt.show()

# find if s is increasing

for i in range(len(s)-1):
    if s[i] >= s[i+1]:
        print(i)


