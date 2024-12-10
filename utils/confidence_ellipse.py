import numpy as np
import scipy.linalg
from math import *

def compute_axes(cov, p, axes_only = True):
    a=np.sqrt(-2*np.log(1-p))
    A = a * scipy.linalg.sqrtm(cov[:2,:2])
    # print("covariance: ", covariance[:2,:2])
    w, v = scipy.linalg.eigh(A)
    v1 = v[:, 0].real
    v2 = v[:, 1].real
    
    f1 = A @ v1
    f2 = A @ v2
    width = np.linalg.norm(f1)
    height = np.linalg.norm(f2)
    angle = np.arctan2(v1[1], v1[0])
    
    if axes_only:
        return width, height
    return width, height, angle

def compute_set(X_ego, X_obb, cov, R_x, R_y, p=0.1):
    a_x, a_y  = compute_axes(cov, p)
    

    x_obst = X_obb[0]
    y_obst = X_obb[1]
    psi_obst = X_obb[2]

    x_c = X_ego[0]
    y_c = X_ego[1]
    psi_c = X_ego[2]

    d = (((x_c - x_obst) * cos(psi_c-psi_obst) + (y_c - y_obst) * sin(psi_c-psi_obst) )/(R_x +a_x))**2 + \
    (((x_c - x_obst) * sin(psi_c-psi_obst) - (y_c - y_obst) * cos(psi_c-psi_obst) )/(R_y+a_y ))**2 - 1
  
    return d

def get_dist_ellipse(data, other, t, geometries, cov):
    assert other in data.columns, f"{other} not in data"
    assert t < len(data['tb0_0']['x_noisy']), f"t={t} is out of bounds"
    assert other in geometries, f"{other} not in geometries"
    assert geometries[other]['type'] == 'rectangle', f"{other} is not a rectangle"

    center1 = (data['tb0_0']['x_noisy'][t], data['tb0_0']['y_noisy'][t])
    angle1 = data['tb0_0']['theta_noisy'][t]
    width1, height1 = geometries['tb0_0']['width'], geometries['tb0_0']['height']

    center2 = (data[other]['x_noisy'][t], data[other]['y'][t])
    angle2 = data[other]['theta_noisy'][t]
    width2, height2 = geometries[other]['width'], geometries[other]['height']

    v1 = data['tb0_0']['v'][t]
    v2 = data[other]['v'][t]

    w1 = data['tb0_0']['w'][t]
    w2 = data[other]['w'][t]

    x1 = np.array([[center1[0]], [center1[1]], [angle1]])
    x2 = np.array([[center2[0]], [center2[1]], [angle2]])
    u1 = np.array([[v1], [w1]])
    u2 = np.array([[v2], [w2]])
    

    return compute_set(x1, x2, cov, height1 + height2, width1 + width2)


