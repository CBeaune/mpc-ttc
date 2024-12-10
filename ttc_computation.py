import numpy as np
from utils.dist_rectangles import get_corners, closest_points
from utils.confidence_ellipse import compute_axes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def compute_theta_dot(v_i, v_j, p_i, p_j, w_i, center2):
    v_i_bar = v_i + (p_i - np.array([center2[0], center2[1]])) * w_i
    theta_dot = (np.cross((p_j-p_i),  v_i_bar) + np.cross((p_i - p_j) , v_j)) / (np.linalg.norm(p_j - p_i)**2)
    return theta_dot

def ttc(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2):
    # Trouver les points les plus proches
    # Calcul des sommets des rectangles

    rect1 = get_corners(center1, 2*width1, 2*height1, angle1)
    rect2 = get_corners(center2, 2*width2, 2*height2, angle2)
    closest_pair, min_dist, projection = closest_points(rect1, rect2)

    p_i = closest_pair[0]
    p_j = closest_pair[1]



    v_i = np.array([v1*np.cos(angle1), v1*np.sin(angle1)])
    

    v_j = np.array([v2*np.cos(angle2), v2*np.sin(angle2)])

    w_i = w1

    loom = looming(p_i, p_j, v_i, v_j)
    
    
    # theta_dot = compute_theta_dot(v_i, v_j, p_i, p_j, w_i, center2)
   
    # p_i = center1
    # p_j = center2
    E = np.zeros((2,2))
    E[0,0] =  (width1 + width2)
    E[1,1] =  (width1 + width2)
    
    R1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R = R2
    E = E

    Sigma = np.linalg.inv(E)
    dij = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot = 2 * ((v_i - v_j).T @ Sigma  @ (p_i - p_j))/dij
    # ax.plot(t,  dij + dij_dot*t)
    # print(dij, dij_dot)


    return -(dij) /(dij_dot) if (dij_dot) < 0 else -1

def ttc_cov(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, ax=None, p=0.9):
    rect1 = get_corners(center1, 2*width1, 2*height1, angle1)
    rect2 = get_corners(center2, 2*width2, 2*height2, angle2)
    closest_pair, min_dist, projection = closest_points(rect1, rect2)

    p_i = closest_pair[0]
    p_j = closest_pair[1]

    v_i = np.array([v1*np.cos(angle1), v1*np.sin(angle1)])
    v_j = np.array([v2*np.cos(angle2), v2*np.sin(angle2)])
    w_i = w1
    w_j = w2
    
    loom = looming(p_i, p_j, v_i, v_j)
    
    
    # theta_dot = compute_theta_dot(v_i, v_j, p_i, p_j, w_i, center2)

    p_i = center1
    p_j = center2

    E = np.zeros((2,2))
    a_x, a_y  = compute_axes(cov, p)
    E[0,0] =   ((width1 + width2)+ a_x)
    E[1,1] =   ((width1 + width2)+ a_y)
    
    R1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R = R1
    E = R @ E @ R.T

    ell = Ellipse(center2, 2*(width2+ width2 + a_x  ), 2*(width2+ width2  + a_y), 
                    angle=angle2*180/np.pi, edgecolor='r', facecolor='none')
    # ax.add_patch(ell)

    Sigma = np.linalg.inv(E)
    dij = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot = 2 * (v_i - v_j).T @ Sigma @ (p_i - p_j)/dij
    # ax.plot(t,  dij + dij_dot*t)
    if dij_dot == 0:
        if dij-1 <= 0:
            return 0
    
    return (1-dij) /(dij_dot) if (dij_dot) < 0 else -1

def ttc_samples(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, ax=None):
    
    samples = np.random.multivariate_normal(center2, cov[:2,:2], 53050) # 53050 de Groot, 2021
    # ax.plot(samples[:,0], samples[:,1], 'ro')
    TTCs = []
    # get 50 closest samples to center1
    samples = samples[np.argsort(np.linalg.norm(samples - center1, axis=1))[:50]]

    # get 20 closest samples to center2
    samples = samples[np.argsort(np.linalg.norm(samples - center2, axis=1))[:20]] 
    

    for sample in samples:
        TTC= ttc(center1, width1, height1, angle1, v1, w1, sample, width2, height2, angle2, v2, w2)
        TTCs.append(TTC)
    return np.median(TTCs)

def looming(p_i, p_j, v_i, v_j):
    
    R = (p_j - p_i)/np.linalg.norm(p_j - p_i)
    v_ij = v_j - v_i
    if np.dot(v_ij, R) > 0:
        return False
    v_tan = np.linalg.norm(np.cross(v_ij, R))
    if v_tan/np.linalg.norm(p_i - p_j) < 0.01:
        return False
    return True

def ttc_cov_optimist(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, ax=None, p=0.9):
    rect1 = get_corners(center1, 2*width1, 2*height1, angle1)
    rect2 = get_corners(center2, 2*width2, 2*height2, angle2)
    closest_pair, min_dist, projection = closest_points(rect1, rect2)

    p_i = closest_pair[0]
    p_j = closest_pair[1]

    v_i = np.array([v1*np.cos(angle1), v1*np.sin(angle1)])
    v_j = np.array([v2*np.cos(angle2), v2*np.sin(angle2)])
    w_i = w1
    w_j = w2

    E = np.zeros((2,2))
    E[0,0] =  (width1 + width2)
    E[1,1] =  (width1 + width2)
    
    R1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R = R2
    E = E

    # Mean TTC
    Sigma = np.linalg.inv(E)
    dij = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot = 2 * ((v_i - v_j).T @ Sigma  @ (p_i - p_j))/dij

    # Ellipse TTc
    Ell = np.zeros((2,2))
    a_x, a_y  = compute_axes(cov, p)
    Ell[0,0] =   ((width1 + width2)+ a_x)
    Ell[1,1] =   ((width1 + width2)+ a_y)
    
    R1_ell = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2_ell = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R_ell = R1_ell
    E_ell = R_ell @ Ell @ R_ell.T



    Sigma = np.linalg.inv(E_ell)
    dij_ell = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot_ell = 2 * (v_i - v_j).T @ Sigma @ (p_i - p_j)/dij_ell

    if (2*dij_dot - dij_dot_ell) == 0:
        if (2*dij - dij_ell)-1 <= 0:
            return 0

    return ( 1 - (2*dij - dij_ell) / (2*dij_dot - dij_dot_ell)) if (2*dij_dot - dij_dot_ell) < 0 else -1

def ttc_2nd_order(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, ax=None,):
    v_i = np.array([v1*np.cos(angle1), v1*np.sin(angle1)])
    v_j = np.array([v2*np.cos(angle2), v2*np.sin(angle2)])
    w_i = w1
    w_j = w2
    p_i = center1
    p_j = center2

    E = np.zeros((2,2))
    E[0,0] =  (width1 + width2)
    E[1,1] =  (width1 + width2)
    
    R1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R = R2
    E = E

    Sigma = np.linalg.inv(E)
    dij = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot = 2 * ((v_i - v_j).T @ Sigma  @ (p_i - p_j))/dij
    dij_dot_dot = (((v_i - v_j).T @ Sigma  @ (v_i - v_j)) - dij_dot**2)/dij

    a = dij_dot_dot
    b = dij_dot
    c = dij - 1
    if b < 0:
        delta = b**2 - 4*a*c
        if delta < 0:
            return -1
        else:
            t1 = min((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))
            t2 = max((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))
            return t1 if t1 > 0 else t2 if t2 > 0 else -1
    return -1

def ttc_2nd_order_cov(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, ax=None, p=0.9):
    v_i = np.array([v1*np.cos(angle1), v1*np.sin(angle1)])
    v_j = np.array([v2*np.cos(angle2), v2*np.sin(angle2)])
    w_i = w1
    w_j = w2
    p_i = center1
    p_j = center2

    # Ellipse TTc
    Ell = np.zeros((2,2))
    a_x, a_y  = compute_axes(cov, p)
    Ell[0,0] =   ((width1 + width2)+ a_x)
    Ell[1,1] =   ((width1 + width2)+ a_y)
    
    R1_ell = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2_ell = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R_ell = R1_ell
    E_ell = R_ell @ Ell @ R_ell.T


    Sigma = np.linalg.inv(E_ell)
    dij = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot = 2 * ((v_i - v_j).T @ Sigma  @ (p_i - p_j))/dij
    dij_dot_dot = (((v_i - v_j).T @ Sigma  @ (v_i - v_j)) - dij_dot**2)/dij

    a = dij_dot_dot
    b = dij_dot
    c = dij - 1
    if b < 0:
        delta = b**2 - 4*a*c
        if delta < 0:
            return -1
        else:
            t1 = min((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))
            t2 = max((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))
            return t1 if t1 > 0 else t2 if t2 > 0 else -1
    return -1

def ttc_2nd_order_cov_optimist(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, ax=None, p=0.9):
    v_i = np.array([v1*np.cos(angle1), v1*np.sin(angle1)])
    v_j = np.array([v2*np.cos(angle2), v2*np.sin(angle2)])
    w_i = w1
    w_j = w2
    p_i = center1
    p_j = center2

    E = np.zeros((2,2))
    E[0,0] =  (width1 + width2)
    E[1,1] =  (width1 + width2)
    
    R1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R = R2
    E = E

    Sigma = np.linalg.inv(E)
    dij = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot = 2 * ((v_i - v_j).T @ Sigma  @ (p_i - p_j))/dij
    dij_dot_dot = (((v_i - v_j).T @ Sigma  @ (v_i - v_j)) - dij_dot**2)/dij

    # Ellipse TTc
    Ell = np.zeros((2,2))
    a_x, a_y  = compute_axes(cov, p)
    Ell[0,0] =   ((width1 + width2)+ a_x)
    Ell[1,1] =   ((width1 + width2)+ a_y)
    
    R1_ell = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    R2_ell = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    R_ell = R1_ell
    E_ell = R_ell @ Ell @ R_ell.T


    Sigma = np.linalg.inv(E_ell)
    dij_ell = np.sqrt((p_i - p_j).T @ Sigma @ (p_i - p_j))
    dij_dot_ell = 2 * ((v_i - v_j).T @ Sigma @ (p_i - p_j))/dij_ell
    dij_dot_dot_ell = (((v_i - v_j).T @ Sigma  @ (v_i - v_j)) - dij_dot_ell**2)/dij_ell

    a = 2*dij_dot_dot - dij_dot_dot_ell
    b = 2*dij_dot - dij_dot_ell
    c = (2*dij - dij_ell) - 1

    if b < 0:
        delta = b**2 - 4*a*c
        if delta < 0:
            return -1
        else:
            t1 = min((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))
            t2 = max((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))
            return t1 if t1 > 0 else t2 if t2 > 0 else -1
    return -1




    
    
    