import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from utils.visualization import plot_robot, plot_cov_ellipse
from utils.control import pid_control, simulate, closest_waypoint
from utils.cubic_spline_planner import calc_spline_course
from utils.lqr_speed_steer_control import do_simulation, calc_speed_profile
from utils.rectangles import get_corners, closest_points

from ttc_computation import ttc, ttc_cov, ttc_samples

# Python script to launch a scenario simulation of 3 Turtlebot robots
# with a given number of steps and a given time step
# The script will plot the trajectories of the robots
# The robots will follow a previously planned path with PID control
# The path is defined by a list of waypoints (x, y, theta)
# We control the rosbots with steering angle and velocity

# === Parameters =====
N = 10
dt = 0.01
width = 0.132
height = 0.178

PLOT = True

# === Choice of scenario =====
np.random.seed(15) # 15 : collision # 8 : no collision


# === Simulation parameters =====
# Randomize the target speeds
target_speeds = np.random.uniform(0.01, 0.25, size=3)


# Waypoints for crossroads scenario
# Define if cross to left or right
choice = 1
wpt_robot0 = np.array([[i, 0.0 , 0] for i in np.linspace(0, 1.5, 20)])
if choice ==1 :
    wpt_robot0 = np.append(wpt_robot0, np.array([[2.0*(1.000001+i*0.0000001), choice*i, choice*np.pi/2 ] for i in np.linspace(0, 2.0, 10)]), axis=0)
else:
    wpt_robot0 = np.append(wpt_robot0, np.array([[1.5*(1.000001+i*0.0000001), choice*i, -choice*np.pi/2 ] for i in np.linspace(0, 2.0, 10)]), axis=0)

# robot 1 on crossroads
wpt_robot1 = np.array([[1.5*(1.000001+i*0.0000001), 0.75*(1-i),  -np.pi/2] for i in np.linspace(0, 3, 10)])

# robot 2 on same lane as robot 0 stopped
wpt_robot2 = np.array([[2.0, -0.75+i,  np.pi/2] for i in np.linspace(0, 3.75, 10)])


X_robot0 = []
X_robot1 = []
X_robot2 = []
dict_robot = {0: X_robot0, 1: X_robot1, 2: X_robot2}

# === If not cross =====
cross = True
if cross is False:
    target_speeds[1] = 0.0
    target_speeds[2] = 0.0
    
noise = np.array([[0.01, 0.01], [0.0, 0.0], [0.0, 0.0]])

# === Control =====
for i, wpt_target in enumerate([wpt_robot0, wpt_robot1, wpt_robot2]):
    # Compute the trajectory of the robot

    cx, cy, cyaw, ck, s = calc_spline_course(
            wpt_target[:,0], wpt_target[:,1], ds=0.01)
    target_speed = target_speeds[i]  # simulation parameter  m/s

    sp = calc_speed_profile(cyaw, target_speed)

    goal  = wpt_target[-1,:]
    t, x, y, yaw, v, delta = do_simulation(cx, cy, cyaw, ck, sp, wpt_target[0,:], goal, noise[i])
    dict_robot[i] = np.zeros((len(t), 5))
    dict_robot[i][:,0] = x
    dict_robot[i][:,1] = y
    dict_robot[i][:,2] = yaw
    dict_robot[i][:,3] = v
    dict_robot[i][:,4] = delta

   
X_robot0 = dict_robot[0]
X_robot1 = dict_robot[1]
X_robot2 = dict_robot[2]
t = np.arange(0, len(X_robot0)-1, dt)


# === Plot trajectories =====
# Show robot motion
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots()
    
collision = False

ttc_norm = np.zeros((2, X_robot0.shape[0]))
ttc_cov_norm = np.zeros((2, X_robot0.shape[0]))
ttc_samples_norm = np.zeros((2, X_robot0.shape[0]))

min_distances = np.ones((2,2))*np.inf
t_coll = X_robot0.shape[0]
for i in tqdm.tqdm(range(X_robot0.shape[0]), desc="Simulation"):
    ax.cla()

    ax.axis("equal")

    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    

    # plot ego robot
    X_robot0 = dict_robot[0] 
    if PLOT:
        ax.plot(wpt_robot0[:,0],wpt_robot0[:,1], "-",color = sns.color_palette()[0], label="course ego robot")
        ax.plot(wpt_robot1[:,0],wpt_robot1[:,1], "-",color = sns.color_palette()[1], label="course robot 1")

        ax.legend()
        plot_robot(X_robot0[i,0],X_robot0[i,1],X_robot0[i,2], 2*0.178, 2*0.132, ax, color = sns.color_palette()[0])
    rect1 = get_corners([X_robot0[i,0], X_robot0[i,1]], 2*0.178, 2*0.132,  X_robot0[i,2])

    cov = np.ones((2,2)) * 0.0001 

    for j in range(2):
        # add noise depending on cov
        alpha = np.random.multivariate_normal([0, 0], cov)
        X_robot = dict_robot[j+1] 
        
        if i == X_robot.shape[0]:
            X_robot = np.append(X_robot, [X_robot[-1,:]], axis=0)
            dict_robot[j+1] = X_robot
        if PLOT : plot_robot(X_robot[i,0],X_robot[i,1],X_robot[i,2], 2*0.178,2 *0.132,  ax, color = sns.color_palette()[j+1])
        rect2 = get_corners([X_robot[i,0], X_robot[i,1]], 2*0.178, 2*0.132,  X_robot[i,2])

        X_robot[i,:2] += alpha

        if PLOT :plot_robot(X_robot[i,0],X_robot[i,1],X_robot[i,2], 2*0.178,2 *0.132,  ax, color = sns.color_palette()[j+1])
        closest_pair, min_dist, _ = closest_points(rect1, rect2)
        if min_dist < min_distances[j,1]:
            min_distances[j,1] = min_dist
            min_distances[j,0] = i

        if PLOT: ax.plot([closest_pair[0][0], closest_pair[1][0]], [closest_pair[0][1], closest_pair[1][1]], 'r--')


        # plot covariance ellipse around robots 
        if PLOT:
            plot_cov_ellipse([X_robot[i,0], X_robot[i,1]],X_robot[i,2], cov, 0.9, ax, edgecolor=sns.color_palette()[j+1])
            ax.set_xlim(X_robot0[i,0]-1, X_robot0[i,0]+1)
            ax.set_ylim(X_robot0[i,1]-1, X_robot0[i,1]+1)
            ax.grid(True)
        # plt.pause(dt/100)

        h = 0.132
        w = 0.178

        ttc_norm[j][i]=ttc(np.array([X_robot0[i,0], X_robot0[i,1]]), w,h , X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                   np.array([X_robot[i,0], X_robot[i,1]]), w,h, X_robot[i,2], X_robot[i,3], 0.0)
        ttc_cov_norm[j][i] = ttc_cov(np.array([X_robot0[i,0], X_robot0[i,1]]), w,h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                   np.array([X_robot[i,0], X_robot[i,1]]), w,h, X_robot[i,2], X_robot[i,3], 0.0, cov, ax)
        ttc_samples_norm[j][i] = ttc_samples(np.array([X_robot0[i,0], X_robot0[i,1]]), w,h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                     np.array([X_robot[i,0], X_robot[i,1]]), w,h, X_robot[i,2], X_robot[i,3], 0.0, cov)
        
        
        # print("TTC between robot 0 and robot ", j+1, " : ", ttc_norm, ttc_cov_norm, ttc_samples_norm)
        # print("speeds", X_robot0[i,3], X_robot[i,3])
        
        if min_dist < 0.08:
            print("Collision detected with robot ", j+1)
            collision = True
            t_coll = i
            break
    if collision:
        break


# === Save the trajectories =====
dict_robot[0] = X_robot0[:t_coll+1, :]
dict_robot[1] = X_robot1[:t_coll+1, :]
dict_robot[2] = X_robot2[:t_coll+1, :]
# save with pickle
import pickle
with open("data/robots_crossroads.pkl", "wb") as f:
    pickle.dump(dict_robot, f)

# === Save the ttc values =====

with open("data/ttc_crossroads.pkl", "wb") as f:
    pickle.dump([ttc_norm, ttc_cov_norm, ttc_samples_norm], f)

# save the min distances
with open("data/min_distances_crossroads.pkl", "wb") as f:
    pickle.dump(min_distances, f)

# === Plot the ttc values =====

    
# fig.suptitle(f'Replay Speed: {100}')
# with sns.axes_style("whitegrid"):
#     fig, [ax1, ax2] = plt.subplots(2,1)
# for i in tqdm.tqdm(range(X_robot0.shape[0]), desc="Plotting TTC"):
#     for j in range(2):
#         if j == 0:
#             ax1.scatter(i,ttc_norm[j,i] , color= 'r', label="ttc")
#             ax1.scatter(i,ttc_cov_norm[j,i] , color='b', label="ttc_cov")
#             ax1.scatter(i,ttc_samples_norm[j,i] , color='g', label="ttc_samples")
#             ax1.set_ylim(-1,10)
#             ax1.plot([min_distances[j,0], min_distances[j,0]], [-1, 10], 'k--')

#         if j == 1:
#             ax2.scatter(i,ttc_norm[j,i] , color= 'r', label="ttc")
#             ax2.scatter(i,ttc_cov_norm[j,i] , color='b', label="ttc_cov")
#             ax2.scatter(i,ttc_samples_norm[j,i] , color='g', label="ttc_samples")
#             ax2.set_ylim(-1,10)
#             ax2.plot([min_distances[j,0], min_distances[j,0]], [0, 10], 'k--')

# plt.grid(True)
# plt.show()
