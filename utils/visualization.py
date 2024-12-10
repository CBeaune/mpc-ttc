import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy

def plot_robot(x, y, theta, w, h, ax, color, alpha = 1, plot_axes = False):
    """Plot a robot at position (x, y) with orientation theta."""
    angle = theta
    corner = np.array([-h/2, -w/2])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    transformed_corner = rotation_matrix @ corner + np.array([x, y])

    # plot axes of the robot as arrows
    if plot_axes:
        ax.plot([x, x + h/2*np.cos(theta)], [y, y + h/2*np.sin(theta)], color='r')
        ax.plot([x, x - w/2*np.sin(theta)], [y, y + w/2*np.cos(theta)], color='g')

    
    rect = plt.Rectangle((transformed_corner[0], transformed_corner[1]), h, w, angle=theta*180/np.pi, edgecolor=color, facecolor='none', alpha=alpha)
    ax.add_patch(rect)
    ax.plot(x, y, 'o', color=color)

def plot_cov_ellipse(center, angle, cov, p, ax,  edgecolor='r'):
    a=np.sqrt(-2*np.log(1-p))
    A = a * scipy.linalg.sqrtm(cov[:2,:2])
    # print("covariance: ", covariance[:2,:2])
     # Use scipy.linalg.eigh for real symmetric matrices
    w, v = scipy.linalg.eigh(A)
    
    v1 = v[:, 0].real
    v2 = v[:, 1].real
    
    f1 = A @ v1
    f2 = A @ v2
    width = np.linalg.norm(f1)
    height = np.linalg.norm(f2)
    # angle = np.arctan2(v1[1], v1[0])
    
    ellipse = Ellipse(center, 2 * width, 2 * height, angle=np.rad2deg(angle), edgecolor=edgecolor, facecolor='none')
    ax.add_patch(ellipse)