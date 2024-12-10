# Trajectory generation for circular vehicles with w and v as control inputs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.confidence_ellipse import compute_axes

# Constants
Rego = 0.7
Robb = 0.5

# Initial conditions
X_0_ego = np.array([0, 0.2, 0.0 ])
u0_ego = np.array([0.4, 0.0])

X_0_obb = np.array([3, 2.0, np.pi])
u0_obb = np.array([0.4, 0.1])

def positions(X0, u0, t):
    theta = u0[1]*t + X0[2]
    x = u0[0]/u0[1]*(np.sin(theta) - np.sin(X0[2])) + X0[0] if u0[1]!=0 else u0[0]*np.cos(theta)*t + X0[0]
    y = -u0[0]/u0[1]*(np.cos(theta) - np.cos(X0[2])) + X0[1] if u0[1]!=0 else u0[0]*np.sin(theta)*t + X0[1]
    return np.array([x,y,theta])


# aproximate time to collision
v_0_ego = np.array([u0_ego[0]*np.cos(X_0_ego[2]), u0_ego[0]*np.sin(X_0_ego[2])])
v_0_obb = np.array([u0_obb[0]*np.cos(X_0_obb[2]), u0_obb[0]*np.sin(X_0_obb[2])])

a_0_ego = np.array([-u0_ego[1]*u0_ego[0]*np.sin(X_0_ego[2]) if u0_ego[1]!=0 else 0,
                     u0_ego[1]*u0_ego[0]*np.cos(X_0_ego[2]) if u0_ego[1]!=0 else 0])

a_0_obb = np.array([-u0_obb[1]*u0_obb[0]*np.sin(X_0_obb[2]) if u0_obb[1]!=0 else 0,
                        u0_obb[1]*u0_obb[0]*np.cos(X_0_obb[2]) if u0_obb[1]!=0 else 0])

t_prev = 0.0
X_t_ego = positions(X_0_ego, u0_ego, t_prev)

v_t_ego = np.array([u0_ego[0]*np.cos(X_t_ego[2]), u0_ego[0]*np.sin(X_t_ego[2])])
X_t_obb = positions(X_0_obb, u0_obb, t_prev)
v_t_obb = np.array([u0_obb[0]*np.cos(X_t_obb[2]), u0_obb[0]*np.sin(X_t_obb[2])])
d = np.sqrt((X_t_ego[:2] - X_t_obb[:2]).T @ (X_t_ego[:2] - X_t_obb[:2]))
d_dot = ((X_t_ego[:2] - X_t_obb[:2]).T @ (v_t_ego - v_t_obb))/d
d_dot_dot = 1/d * ((v_t_ego - v_t_obb).T @ (v_t_ego - v_t_obb) + (X_t_ego[:2] - X_t_obb[:2]).T @ (a_0_ego - a_0_obb) - d_dot**2)
t_coll = t_prev + (Rego+Robb-d)/d_dot if d_dot < 0 else np.inf
print(f't_coll: {t_coll}')

covariance = np.diag([0.05,0.01,0.000001])
alpha, beta, angle = compute_axes(covariance, 0.997, axes_only=False)
D = np.diag([(alpha + Rego + Robb)**2, (beta + Rego + Robb)**2])
Rot_angle = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Rot_ego = np.array([[np.cos(X_t_ego[2]), -np.sin(X_t_ego[2])], [np.sin(X_t_ego[2]), np.cos(X_t_ego[2])]])
Rot_obb = np.array([[np.cos(X_t_obb[2]), -np.sin(X_t_obb[2])], [np.sin(X_t_obb[2]), np.cos(X_t_obb[2])]])
Rot = Rot_obb @ Rot_angle
Sigma = Rot @ D @ Rot.T
d_ell_99 = np.sqrt((X_t_ego[:2] - X_t_obb[:2]).T @ np.linalg.inv(Sigma) @ (X_t_ego[:2] - X_t_obb[:2]))
d_dot_ell_99 = ((v_t_ego - v_t_obb).T @ np.linalg.inv(Sigma) @ (X_t_ego[:2] - X_t_obb[:2]))/d_ell_99
d_dot_dot_ell_99 = 1/d_ell_99 * ((v_t_ego - v_t_obb).T @ np.linalg.inv(Sigma) @ (v_t_ego - v_t_obb) + (X_t_ego[:2] - X_t_obb[:2]).T @ np.linalg.inv(Sigma) @ (a_0_ego - a_0_obb) - d_dot_ell_99**2)
from matplotlib.patches import Ellipse
e = Ellipse(xy=X_t_obb[:2], width=2 * alpha, height=2 * beta, angle=angle, edgecolor='r', facecolor='r', alpha=0.2)


alpha, beta, angle = compute_axes(covariance, 0.64, axes_only=False)

D = np.diag([(alpha + Rego + Robb)**2, (beta + Rego + Robb)**2])
Rot_angle = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Rot_ego = np.array([[np.cos(X_t_ego[2]), -np.sin(X_t_ego[2])], [np.sin(X_t_ego[2]), np.cos(X_t_ego[2])]])
Rot_obb = np.array([[np.cos(X_t_obb[2]), -np.sin(X_t_obb[2])], [np.sin(X_t_obb[2]), np.cos(X_t_obb[2])]])
Rot = Rot_obb @ Rot_angle
Sigma = Rot @ D @ Rot.T
d_ell_10 = np.sqrt((X_t_ego[:2] - X_t_obb[:2]).T @ np.linalg.inv(Sigma) @ (X_t_ego[:2] - X_t_obb[:2]))
d_dot_ell_10 = ((v_t_ego - v_t_obb).T @ np.linalg.inv(Sigma) @ (X_t_ego[:2] - X_t_obb[:2]))/d_ell_10
d_dot_dot_ell_10 = 1/d_ell_10 * ((v_t_ego - v_t_obb).T @ np.linalg.inv(Sigma) @ (v_t_ego - v_t_obb) + (X_t_ego[:2] - X_t_obb[:2]).T @ np.linalg.inv(Sigma) @ (a_0_ego - a_0_obb) - d_dot_ell_10**2)
e = Ellipse(xy=X_t_obb[:2], width=2 * alpha, height=2 * beta, angle=angle, edgecolor='r', facecolor='r', alpha=0.2)


tp = -2 * (-d_dot) * (-d + d_ell_10) / (1+d_dot**2)
d_p = d_ell_10 - 2 * (d_ell_10 - d)/ (1+d_dot**2)

def parameters(d, d_dot, d_ell, d_dot_ell):
    A = 2*d_dot/(1+d_dot**2)*(1+d_dot_ell*d_dot) - d_dot_ell
    B =     2*d/(1+d_dot**2)*(1+d_dot_ell*d_dot) - d_ell
    C =       2/(1+d_dot**2)*(1+d_dot_ell*d_dot) - 1
    return A, B, C

A_10, B_10, C_10 = parameters(d/(Rego+Robb), d_dot/(Rego+Robb), d_ell_10, d_dot_ell_10)
A_99, B_99, C_99 = parameters(d/(Rego+Robb), d_dot/(Rego+Robb), d_ell_99, d_dot_ell_99)

def distance_function(t, vi, vj, wi, wj , p_i, p_j):
    xi_t = vi * np.cos(p_i[2])*t + p_i[0] if wi == 0 else vi/wi*(np.sin(wi*t + p_i[2] ) - np.sin(p_i[2])) + p_i[0]
    yi_t = vi * np.sin(p_i[2])*t + p_i[1] if wi == 0 else vi/wi*(-np.cos(wi*t + p_i[2] ) + np.cos(p_i[2])) + p_i[1]

    xj_t = vj * np.cos(p_j[2])*t + p_j[0] if wj == 0 else vj/wj*(np.sin(wj*t + p_j[2] ) - np.sin(p_j[2])) + p_j[0]
    yj_t = vj * np.sin(p_j[2])*t + p_j[1] if wj == 0 else vj/wj*(-np.cos(wj*t + p_j[2] ) + np.cos(p_j[2])) + p_j[1]

    return np.sqrt((xi_t - xj_t)**2 + (yi_t - yj_t)**2)

def propagate_uncertainties(ego, obst, v_ego, v_obst, cov):
    assert ego.shape[0] == 3
    assert obst.shape[0] == 3
    assert v_ego.shape[0] == 2
    assert v_obst.shape[0] == 2
    assert cov.shape[0] > 1 and cov.shape[0] == cov.shape[1]

    x_ego, y_ego, theta_ego = ego
    x_obst, y_obst, theta_obst = obst
    v_x_ego, v_y_ego = v_ego
    v_x_obst, v_y_obst = v_obst

    p_i = np.array([x_ego, y_ego])
    p_j = np.array([x_obst, y_obst])
    den = ((p_i- p_j).T @ (v_ego - v_obst))**2
    dT_dx = (-x_ego + x_obst)/np.sqrt((x_ego - x_obst)**2 + (y_ego - y_obst)**2)
    dT_dy = (-y_ego + y_obst)/np.sqrt((x_ego - x_obst)**2 + (y_ego - y_obst)**2)
    J = np.array(
                [[dT_dx],
                [dT_dy]]
                )
    

    sigma = []
    if J.shape == (2,1):
        return np.sqrt(J.T @ cov[:2,:2] @ J)
    for i in range(J.shape[-1]):
        sigma.append(np.sqrt(J[:,:,i].T @ cov[:2,:2] @ J[:,:,i]))
    return np.array(sigma)

import seaborn as sns
with sns.axes_style("whitegrid"):
    plt.rcParams.update({
    
    "font.serif": [],  # Optionally specify a LaTeX serif font, e.g., Times
    "axes.labelsize": 16,  # Font size for axis labels
    "axes.titlesize": 18,  # Font size for titles
    "legend.fontsize": 14,  # Font size for legends
    "xtick.labelsize": 14,  # Font size for x-axis tick labels
    "ytick.labelsize": 14,  # Font size for y-axis tick labels
    })
    plt.figure(figsize=(9,6))
    plt.ylim(-0.1, 6.0)
    t = np.arange(0,10,0.01)
    distance = distance_function(t, u0_ego[0], u0_obb[0], u0_ego[1], u0_obb[1], X_0_ego, X_0_obb)
    plt.plot(t, distance/(Rego+Robb), label = 'normalized distance function')
    line0 = ((d_dot*(t-t_prev) + d)/(Rego+Robb))
    plot_tcoll = ((Rego+Robb) - d)/d_dot
    plt.plot(t,line0, label='1st order approximation')
    plt.scatter(plot_tcoll, 1, c= sns.color_palette()[1], label=' estimated time to collision (1st order)')
    line0 = ((d_dot_dot*(t-t_prev)**2 +d_dot*(t-t_prev) + d)/(Rego+Robb))
    a = d_dot_dot/(Rego+Robb)
    b = d_dot/(Rego+Robb)
    c = d/(Rego+Robb) - 1

    delta = b**2 - 4*a*c
    plot_tcoll = (-b - np.sqrt(delta))/2/a if delta >= 0 else np.inf
    plt.scatter(plot_tcoll, 1,  c= sns.color_palette()[2], label=' estimated time to collision (2nd order)')
    plt.plot(t,line0, label='2nd order approximation')
    plt.plot([0,10],[1,1], 'g--', label = 'collision threshold')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized distance')
    plt.legend()
    
    plt.tight_layout()

# uncertainties = propagate_uncertainties(X_0_ego, X_0_obb, v_0_ego, v_0_obb, covariance)
# print(f'uncertainties: {np.sqrt(uncertainties)}')

with sns.axes_style("whitegrid"):
    plt.rcParams.update({
    
    "font.serif": [],  # Optionally specify a LaTeX serif font, e.g., Times
    "axes.labelsize": 16,  # Font size for axis labels
    "axes.titlesize": 18,  # Font size for titles
    "legend.fontsize": 12,  # Font size for legends
    "xtick.labelsize": 14,  # Font size for x-axis tick labels
    "ytick.labelsize": 14,  # Font size for y-axis tick labels
    })
    plt.figure(figsize=(9,6))
    plt.ylim(-0.1, 5)
    t = np.arange(0,10,0.01)
    distance = distance_function(t, u0_ego[0], u0_obb[0], u0_ego[1], u0_obb[1], X_0_ego, X_0_obb)
    plt.plot(t, distance/(Rego+Robb), label = 'normalized distance function', linewidth=3)
    line0 = (( d_dot_dot*(t-t_prev)**2 + d_dot*(t-t_prev) + d)/(Rego+Robb))
    line1_10 =  d_dot_dot_ell_10*(t-t_prev)**2 + d_dot_ell_10*(t-t_prev) + d_ell_10 
    line1_99 =  d_dot_dot_ell_99*(t-t_prev)**2 + d_dot_ell_99*(t-t_prev) + d_ell_99

    line2_10 = 2* line0 - line1_10
    line2_99 = 2* line0 - line1_99

    # plt.plot(t,line0, label='2nd order approximation')
    # plt.plot(t, line1_10,  label='1 $\sigma$ 2nd order', color=sns.color_palette()[1])
    # plt.plot(t, line2_10,   color=sns.color_palette()[1])
    # plt.fill_between(t, line1_10, line2_10, color=sns.color_palette()[1], alpha=0.2)
    # plt.plot(t, line1_99,  label='3 $\sigma$ 2nd order', color=sns.color_palette()[3])
    # plt.plot(t, line2_99,   color=sns.color_palette()[3])
    # plt.fill_between(t, line1_99, line2_99, color=sns.color_palette()[3], alpha=0.2)
    # plt.plot([0,10],[1,1], 'g--', label = 'collision threshold')

    distance_cov = []
    xi = positions(X_0_ego,u0_ego,t) 
    xj = positions(X_0_obb,u0_obb,t) 
    np.random.seed(0)
    xj[:2] += np.random.multivariate_normal([0,0], covariance[:2,:2], (1000)).T
    distance_cov = np.linalg.norm(xi[:2] - xj[:2], axis=0)
    plt.scatter(t, np.array(distance_cov)/(Rego+Robb), alpha = 0.2, c=sns.color_palette()[0],label = 'distance with gaussian noise')
    
    # uncertainties propagation
    print(positions(X_0_ego,u0_ego,t).shape)


    line7 = line0 +  propagate_uncertainties(X_0_ego , X_0_obb ,
                                                        v_0_ego, v_0_obb, covariance).T
    line8 = line0 -  propagate_uncertainties(positions(X_0_ego,u0_ego,t) , positions(X_0_obb,u0_obb,t) ,
                                                        v_0_ego, v_0_obb, covariance).T
    plt.plot(t, line7.reshape(1000,), '--', label='$\sigma$ uncertainties', color=sns.color_palette()[3])
    plt.plot(t, line8.reshape(1000,), '--',  color=sns.color_palette()[3])
    plt.fill_between(t, line7.reshape(1000,), line8.reshape(1000,), color=sns.color_palette()[3], alpha=0.2)

    line5 = line0 +  2*propagate_uncertainties(X_0_ego , X_0_obb ,
                                                      v_0_ego, v_0_obb, covariance).T
    line6 = line0 -  2*propagate_uncertainties(X_0_ego , X_0_obb ,
                                                      v_0_ego, v_0_obb, covariance).T
    plt.plot(t, line5.reshape(1000,), '--', label='2 $\sigma$ uncertainties', color=sns.color_palette()[1])

    plt.plot(t, line6.reshape(1000,), '--',  color=sns.color_palette()[1])
    plt.fill_between(t, line5.reshape(1000,), line6.reshape(1000,), color=sns.color_palette()[1], alpha=0.2)

    line3 = line0 +  3*propagate_uncertainties(X_0_ego , X_0_obb ,
                                                      v_0_ego, v_0_obb, covariance).T

    line4 = line0 - 3*propagate_uncertainties(X_0_ego , X_0_obb ,
                                                      v_0_ego, v_0_obb, covariance).T
    plt.plot(t, line3.reshape(1000,), '--', label='3 $\sigma$ uncertainties', color=sns.color_palette()[2])
    plt.plot(t, line4.reshape(1000,), '--',  color=sns.color_palette()[2])
    plt.fill_between(t, line3.reshape(1000,), line4.reshape(1000,), color=sns.color_palette()[2], alpha=0.2)

    


    plt.xlabel('Time [s]')
    plt.ylabel('Normalized distance')
    plt.legend()

    plt.figure()
    plt.scatter(positions(X_0_ego,u0_ego,t)[0], positions(X_0_ego,u0_ego,t)[1])
    plt.scatter(positions(X_0_obb,u0_obb,t)[0], positions(X_0_obb,u0_obb,t)[1])
    plt.scatter(xi[0], xi[1])
    plt.scatter(xj[0], xj[1])
    plt.show()


t_coll = (1-d)/d_dot if d_dot < 0 else np.inf
t_coll_99 = (1-d_ell_99)/d_dot_ell_99 if d_dot_ell_99 < 0 else np.inf




# if t_coll != np.inf:
#     t = np.arange(t_prev,t_coll,0.01)
#     line1_10 = (2*d_dot - d_dot_ell_10)*(t-t_prev) + (2*d -d_ell_10) 
#     line2_10 = d_dot_ell_10*(t-t_prev) + d_ell_10 
#     ttc1_99 = (1-(2*d-d_ell_10))/(2*d_dot - d_dot_ell_10) if 2*d_dot - d_dot_ell_10 < 0 else np.inf
#     ttc2_99 = (1-d_ell_10)/(d_dot_ell_10) if d_dot_ell_10 < 0 else np.inf
#     print(f"ttc1_99: {t_prev + ttc1_99}, ttc2_99: {t_prev + ttc2_99}")
#     ax1.plot(t,line1_10, 'b--')
#     ax1.plot(t,line2_10, 'b--')
#     ax1.fill_between(t, line1_10, line2_10, color='b', alpha=0.2)

#     line1_99 = (2*d_dot - d_dot_ell_99)*(t-t_prev) + (2*d -d_ell_99) 
#     ttc1_99 = (1-(2*d-d_ell_99))/(2*d_dot - d_dot_ell_99) if 2*d_dot - d_dot_ell_99 < 0 else np.inf
#     ttc2_99 = (1-d_ell_99)/(d_dot_ell_99) if d_dot_ell_99 < 0 else np.inf
#     print(f"ttc1_99: {t_prev + ttc1_99}, ttc2_99: {t_prev + ttc2_99}")
#     line2_99 = d_dot_dot_ell_99 * (t-t_prev)**2 + d_dot_ell_99*(t-t_prev) + d_ell_99 
#     ax1.plot(t,line1_99, 'r--')
#     ax1.plot(t,line2_99, 'r--')
#     ax1.fill_between(t, line1_99, line2_99, color='r', alpha=0.2)

#     line0 = ((d_dot_dot*(t-t_prev)**2 +d_dot*(t-t_prev) + d)/(Rego+Robb))
#     ax1.plot(t,line0, 'k')

#     ax1.plot([t_coll, t_coll],[0,10], 'g--')
# ax1.plot([0,10],[1,1], 'b--')
# plt.show()
# print(f"Time to collision: {t_coll}")

# for t in np.arange(0,min(t_coll,15),0.01):

#     X_ego = positions(X_0_ego,u0_ego,t) 
#     X_obb = positions(X_0_obb,u0_obb,t) 
#     X_obb[:2] += np.random.multivariate_normal([0,0], covariance[:2,:2])



#     distance = np.linalg.norm(X_ego[:2] - X_obb[:2])
#     ax1.plot(t,distance/(Rego+Robb), 'r.')
    


#     ax2.plot(X_ego[0], X_ego[1], 'ro')
#     circle_ego = patches.Circle((X_ego[0], X_ego[1]), Rego, fill=False, alpha = 0.1)
#     ax2.add_patch(circle_ego)
#     ax2.plot(X_obb[0], X_obb[1], 'bo')
#     circle_obb = patches.Circle((X_obb[0], X_obb[1]), Robb, fill=False, alpha = 0.1)
#     ax2.add_patch(circle_obb)
#     ellipse_obb = patches.Ellipse((X_obb[0], X_obb[1]), 2*alpha, 2*beta, angle=(angle + X_obb[2] )*180/np.pi, edgecolor='r', facecolor='r', alpha = 0.1)
#     ax2.add_patch(ellipse_obb)




plt.show()