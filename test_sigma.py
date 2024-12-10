import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
dt = 0.1
w = 0.178
h = 0.132


fig, ax = plt.subplots()
computation_time = []
# === Choice of scenario =====
for seed in range(10):

    np.random.seed(seed) # 15 : good overtaking # 8 : collision in overtaking

    for test in range(2):


        # === Simulation parameters =====
        # Randomize the target speeds
        target_speeds = np.random.uniform(0.01, 0.25, size=3)


        # Waypoints for 2 lanes scenario
        wpt_robot0 = np.array([[i, 0.0 , 0] for i in np.linspace(0, 3.5, 10)])

        # robot 1 on other lane
        wpt_robot1 = np.array([[4-i, 0.75,  -np.pi] for i in np.linspace(0, 3, 10)])

        # robot 2 on same lane as robot 0 stopped
        wpt_robot2 = np.array([[ 2+(i), 0.0, 0] for i in np.linspace(0, 3, 5)])

        X_robot0 = []
        X_robot1 = []
        X_robot2 = []
        dict_robot = {0: X_robot0, 1: X_robot1, 2: X_robot2}

        # === If overtaking =====
        overtaking = True

        if overtaking:
            target_speeds[2] = 0.0
            
            x_obst = 2.0
            y_obst = 0.0
            theta_obst = np.pi/6
            s_min = 2*w*np.cos(theta_obst) + 2*h*np.sin(theta_obst)
            s_max = 0.75

            n_min = 2*h*np.sin(theta_obst) + 2*w*np.cos(theta_obst)+ 2*w
            n_max = max(2*h*np.sin(theta_obst) + 2*w*np.cos(theta_obst) + 2*w, 0.75)

            pA = [x_obst  , y_obst + n_min,0]
            pB = [x_obst + s_max , y_obst ,0]

            wpt_robot_prev = wpt_robot0[wpt_robot0[:,0] < x_obst - s_max ,:]
            wpt_robot_past = wpt_robot0[wpt_robot0[:,0] > x_obst + s_max ,:]
            waypoints = np.append(wpt_robot_prev, np.array([pA, pB]), axis=0)
            waypoints = np.append(waypoints, wpt_robot_past, axis=0)

            from scipy.interpolate import BSpline
            t = np.linspace(0, 1, len(waypoints))  # Paramètre t (temps) pour chaque point
            # Générer des nœuds pour la B-spline
            knots = np.linspace(0, 1, waypoints.shape[0] - 2 + 1)
            knots = np.concatenate(([0] * 2, knots, [1] * 2))  # Ajouter nœuds pour continuité

            spline_x = BSpline(knots, waypoints[:,0],2)
            spline_y = BSpline(knots, waypoints[:,1],2)

            # Générer des points sur la spline
            t_spline = np.linspace(0, 1, 100)
            x_spline = spline_x(t_spline)
            y_spline = spline_y(t_spline)
            theta_spline = np.arctan2(np.diff(y_spline), np.diff(x_spline))
            theta_spline = np.insert(theta_spline, 0, theta_spline[0])
            wpt_robot0 = np.concatenate((np.array([x_spline]).T, np.array([y_spline]).T, np.zeros((100,1))), axis=1)

        # === Control =====
        for i, wpt_target in enumerate([wpt_robot0, wpt_robot1, wpt_robot2]):
            # Compute the trajectory of the robot

            cx, cy, cyaw, ck, s = calc_spline_course(
                    wpt_target[:,0], wpt_target[:,1], ds=0.05)
            target_speed = target_speeds[i]  # simulation parameter  m/s

            sp = calc_speed_profile(cyaw, target_speed)
            goal  = wpt_target[-1,:]
            t, x, y, yaw, v, delta = do_simulation(cx, cy, cyaw, ck, sp, wpt_target[0,:], goal)
            dict_robot[i] = np.zeros((len(t), 5))
            dict_robot[i][:,0] = x
            dict_robot[i][:,1] = y
            dict_robot[i][:,2] = yaw
            dict_robot[i][:,3] = v
            dict_robot[i][:,4] = delta

        X_robot0 = dict_robot[0]
        X_robot1 = dict_robot[1]
        X_robot2 = dict_robot[2]
        if overtaking:
            dict_robot[2] = np.array([[2, 0, np.pi/6, 0, 0]])
        t = np.arange(0, len(X_robot0)-1, dt)


        # === Plot trajectories =====
        # Show robot motion
        # with sns.axes_style("whitegrid"):
        #     fig, ax = plt.subplots()

        collision = False
        ttc_cov_norm1 = np.zeros((2, X_robot0.shape[0]))
        ttc_cov_norm2 = np.zeros((2, X_robot0.shape[0]))
        ttc_cov_norm3 = np.zeros((2, X_robot0.shape[0]))

        ct_cov_norm1 = 0
        ct_samples_norm = 0
        ct_cov_norm3 = 0

        for i in range(X_robot0.shape[0]):
            # ax.cla()

            # ax.axis("equal")

            # plt.gcf().canvas.mpl_connect(
            #     'key_release_event',
            #     lambda event: [exit(0) if event.key == 'escape' else None])
            

            # plot ego robot
            X_robot0 = dict_robot[0] 
            # ax.plot(wpt_robot0[:,0],wpt_robot0[:,1], "-",color = sns.color_palette()[0], label="course ego robot")
            # ax.plot(wpt_robot1[:,0],wpt_robot1[:,1], "-",color = sns.color_palette()[1], label="course robot 1")

            # ax.legend()
            # plot_robot(X_robot0[i,0],X_robot0[i,1],X_robot0[i,2], 2*w, 2*h, ax, color = sns.color_palette()[0])
            rect1 = get_corners([X_robot0[i,0], X_robot0[i,1]], 2*w, 2*h,  X_robot0[i,2])

            cov = np.eye(2) *( 0.005 )**2
            cov[1,1] = 0.01**2

            for j in range(2):
                # add noise depending on cov
                alpha = np.random.multivariate_normal([0, 0], cov)
                X_robot = dict_robot[j+1] 
                
                if i == X_robot.shape[0]:
                    X_robot = np.append(X_robot, [X_robot[-1,:]], axis=0)
                    dict_robot[j+1] = X_robot
                # plot_robot(X_robot[i,0],X_robot[i,1],X_robot[i,2], 2*w,2 *h,  ax, color = sns.color_palette()[j+1])
                rect2 = get_corners([X_robot[i,0], X_robot[i,1]], 2*w, 2*h,  X_robot[i,2])

                X_robot[i,:2] += alpha

                # plot_robot(X_robot[i,0],X_robot[i,1],X_robot[i,2], 2*w,2 *h,  ax, color = sns.color_palette()[j+1])
                closest_pair, min_dist, _ = closest_points(rect1, rect2)

                # ax.plot([closest_pair[0][0], closest_pair[1][0]], [closest_pair[0][1], closest_pair[1][1]], 'r--')


                

                # plot covariance ellipse around robots 
                # plot_cov_ellipse([X_robot[i,0], X_robot[i,1]],X_robot[i,2], cov, 0.9, ax, edgecolor=sns.color_palette()[j+1])
                # ax.set_xlim(X_robot0[i,0]-1, X_robot0[i,0]+1)
                # ax.set_ylim(X_robot0[i,1]-1, X_robot0[i,1]+1)
                # ax.grid(True)


                t1 = time.time()
                ttc_cov_norm1[j][i] = ttc_cov(np.array([X_robot0[i,0], X_robot0[i,1]]), w,h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                        np.array([X_robot[i,0], X_robot[i,1]]), w,h, X_robot[i,2], X_robot[i,3], 0.0, cov, ax, 0.68)
                t2 = time.time()
                ttc_cov_norm2[j][i] = ttc_samples(np.array([X_robot0[i,0], X_robot0[i,1]]), w,h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                            np.array([X_robot[i,0], X_robot[i,1]]), w,h, X_robot[i,2], X_robot[i,3], 0.0, cov, ax)
                t3 = time.time()
                ttc_cov_norm3[j][i] = ttc_cov(np.array([X_robot0[i,0], X_robot0[i,1]]), w,h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                                np.array([X_robot[i,0], X_robot[i,1]]), w,h, X_robot[i,2], X_robot[i,3], 0.0, cov, ax, 0.997)
                t4 = time.time()
                
                ct_cov_norm1 += t2 - t1
                ct_samples_norm += t3 - t2
                ct_cov_norm3 += t4 - t3
                
                # print("TTC between robot 0 and robot ", j+1, " : ", ttc_norm, ttc_cov_norm, ttc_samples_norm)
                # print("speeds", X_robot0[i,3], X_robot[i,3])
                
                if min_dist < 0.08:
                    ttc_real = i
                    print("Collision detected with robot ", j+1)
                    collision = True
                    break
            
            # plt.pause(dt/100)
            if collision:
                break
            
            # fig.suptitle(f'Replay Speed: {100}')
    
    # ax.plot(seed, ct_cov_norm1/min(ttc_real, X_robot[0].shape[0]), 'o', label="ttc_cov 1*sigma", color = sns.color_palette()[0])
    # ax.plot(seed, ct_samples_norm/min(ttc_real, X_robot[0].shape[0]),'o', label="ttc_samples", color = sns.color_palette()[1])
    # ax.plot(seed, ct_cov_norm3/min(ttc_real, X_robot[0].shape[0]), 'o',label="ttc_cov 3*sigma", color = sns.color_palette()[2])

        computation_time.append([[ct_cov_norm1/min(ttc_real, X_robot[0].shape[0])],
                                [ct_samples_norm/min(ttc_real, X_robot[0].shape[0])],
                                 [   ct_cov_norm3/min(ttc_real, X_robot[0].shape[0])]])

    computation_time_array = np.array(computation_time)   
    
    ax.boxplot(computation_time_array[:,0].squeeze(), positions=[seed])
    ax.boxplot(computation_time_array[:,1].squeeze(), positions=[seed])
    ax.boxplot(computation_time_array[:,2].squeeze(), positions=[seed])
    

    # print("Time for ttc_cov_norm1: ", ct_cov_norm1/min(ttc_real, X_robot0.shape[0]) )
    # print("Time for ttc_samples_norm: ", ct_samples_norm/min(ttc_real, X_robot0.shape[0]))
    # print("Time for ttc_cov_norm3: ", ct_cov_norm3/min(ttc_real, X_robot0.shape[0]))

    # with sns.axes_style("whitegrid"):
    #     fig, [ax1, ax2] = plt.subplots(2,1)
    # ax1.plot( np.arange(X_robot0.shape[0]), ttc_cov_norm1[0], color=sns.color_palette()[1], label="ttc_cov 1*sigma")
    # ax1.plot( np.arange(X_robot0.shape[0]), ttc_cov_norm2[0], color=sns.color_palette()[2], label="ttc")
    # ax1.plot( np.arange(X_robot0.shape[0]), ttc_cov_norm3[0], color=sns.color_palette()[3], label="ttc_cov 3*sigma")

    # ax2.plot( np.arange(X_robot0.shape[0]), ttc_cov_norm1[1], color=sns.color_palette()[1], label="ttc_cov 1*sigma")
    # ax2.plot( np.arange(X_robot0.shape[0]), ttc_cov_norm2[1], color=sns.color_palette()[2], label="ttc")
    # ax2.plot( np.arange(X_robot0.shape[0]), ttc_cov_norm3[1], color=sns.color_palette()[3], label="ttc_cov 3*sigma")

    # # for i in range(X_robot0.shape[0]):
    # #     for j in range(2):
    # #         if j == 0:
    # #             ax1.scatter(i,ttc_norm[j,i] , color= 'r', label="ttc")
    # #             ax1.scatter(i,ttc_cov_norm[j,i] , color='b', label="ttc_cov")
    # #             ax1.scatter(i,ttc_samples_norm[j,i] , color='g', label="ttc_samples")
    # ax1.set_ylim(-1,8)
    # ax1.set_xlim(0, ttc_real+20)
    # ax1.set_ylabel("TTC between robot 0 and robot 1 (s)")

    # #         if j == 1:
    # #             ax2.scatter(i,ttc_norm[j,i] , color= 'r', label="ttc")
    # #             ax2.scatter(i,ttc_cov_norm[j,i] , color='b', label="ttc_cov")
    # #             ax2.scatter(i,ttc_samples_norm[j,i] , color='g', label="ttc_samples")
    # ax2.set_ylim(-1,8)
    # ax2.set_ylabel("TTC between robot 0 and robot 2 (s)")
    # ax2.set_xlim(0, ttc_real+20)
    # if collision:
    #     if j+1 == 1:
    #         ax1.axvline(x=ttc_real, color='gray', linestyle='--', label="Collision detected")
    #     else:
    #         ax2.axvline(x=ttc_real, color='gray', linestyle='--', label="Collision detected")

    # ax1.set_xlabel("Simulation Time (iterations)")
    # ax2.set_xlabel("Simulation Time (iterations)")

    # ax1.legend()
    # ax2.legend()
    # plt.grid(True)
    # plt.show()
    plt.pause(0.1)
plt.show()