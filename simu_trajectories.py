import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import os 
import pickle as pkl

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
VIZ=False

N = 200
dt = 0.1
T = N*dt
w = 0.132
h = 0.178


# === Functions =====
def init(parameters):
    SCENARIO = parameters["SCENARIO"]
    target_speeds = np.random.uniform(0.1, 0.65, size=3)

    

    if SCENARIO == "crossing":
        CROSS = parameters["CROSS"]
        choice = parameters["DIR"]
        assert choice in [0, 1]

        wpt_robot0 = np.array([[i, 0.0 , 0] for i in np.linspace(0, 1.5, 20)])
        if choice ==1 :
            wpt_robot0 = np.append(wpt_robot0, np.array([[2.02*(1.000001+i*0.0000001), choice*i, choice*np.pi/2 ] for i in np.linspace(0, 2.0, 10)]), axis=0)
        elif choice == 0:
            wpt_robot0 = np.append(wpt_robot0, np.array([[1.5*(1.000001+i*0.0000001), -1*i, -choice*np.pi/2 ] for i in np.linspace(0, 2.0, 10)]), axis=0)

        # robot 1 on crossroads
        wpt_robot1 = np.array([[1.5*(1.000001+i*0.0000001), 0.75*(1-i),  -np.pi/2] for i in np.linspace(0, 3, 10)])

        # robot 2 on same lane as robot 0 stopped
        wpt_robot2 = np.array([[2.02, -2.0+i,  np.pi/2] for i in np.linspace(0, 5.0, 10)])

        # === If not cross =====
        if CROSS is False:
            target_speeds[1] = 0.0
            target_speeds[2] = 0.0

    
    
    elif SCENARIO == "twolanes":
        OVERTAKING = parameters["OVERTAKING"]
        
        target_speeds[2] = np.random.uniform(0.01,0.1)
        # vehicle 1 is either stopped or moving at a constant speed
        target_speeds[1] = np.random.uniform(0.0,0.5)
        # Waypoints for 2 lanes scenario
        wpt_robot0 = np.array([[i, 0.0 , 0] for i in np.linspace(0, 3.5, 10)])

        # robot 1 on other lane
        wpt_robot1 = np.array([[4-i, 0.75,  -np.pi] for i in np.linspace(0, 3, 10)])

        # robot 2 on same lane as robot 0 stopped
        wpt_robot2 = np.array([[ 2+(i), 0.0, 0] for i in np.linspace(0, 3, 5)])
        
        if OVERTAKING:
            target_speeds[0] = np.random.uniform(0.4,0.65)
            target_speeds[2] = 0.0
            
            x_obst = 2.1
            y_obst = 0.0
            theta_obst = np.pi/6
            s_min = 2*w*np.cos(theta_obst) + 2*h*np.sin(theta_obst)
            s_max = 1.0

            n_min = 2*h*np.sin(theta_obst) + 2*w*np.cos(theta_obst)+ 2*w
            n_max = max(2*h*np.sin(theta_obst) + 2*w*np.cos(theta_obst) + 2*w, 1.0)

            pA = [x_obst  , y_obst + n_max,0]
            pB = [x_obst + s_max , y_obst ,0]

            wpt_robot_prev = wpt_robot0[wpt_robot0[:,0] < x_obst - s_max ,:]
            wpt_robot_past = wpt_robot0[wpt_robot0[:,0] > x_obst + s_max + 0.25 ,:]
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

    INACTIVE_ROBOT = parameters["INACTIVE_ROBOT"]
    if INACTIVE_ROBOT is not None:
        target_speeds[INACTIVE_ROBOT] = 0.0

    X_robot0 = []
    X_robot1 = []
    X_robot2 = []
    if SCENARIO == "twolanes":
        obstacle = 1
    elif SCENARIO == "crossing":
        if INACTIVE_ROBOT == 1:
            obstacle = 2
        elif INACTIVE_ROBOT == 2:
            obstacle = 1
    dict_robot = {0: X_robot0, 1: X_robot1, 2: X_robot2, 'obstacle': obstacle}

    noise = np.array([[0.01, 0.01], [0.0, 0.0], [0.0, 0.0]])
    
    # === Control =====
    for i, wpt_target in enumerate([wpt_robot0, wpt_robot1, wpt_robot2]):
        # Compute the trajectory of the robot

        cx, cy, cyaw, ck, s = calc_spline_course(
                wpt_target[:,0], wpt_target[:,1], ds=0.01)
        target_speed = target_speeds[i]  # simulation parameter  m/s

        sp = calc_speed_profile(cyaw, target_speed)

        goal  = wpt_target[-1,:]
        t, x, y, yaw, v, delta = do_simulation(cx, cy, cyaw, ck, sp, wpt_target[0,:], goal, noise[i], T = T)
        dict_robot[i] = np.zeros((len(t), 5))
        dict_robot[i][:,0] = x
        dict_robot[i][:,1] = y
        dict_robot[i][:,2] = yaw
        dict_robot[i][:,3] = v
        dict_robot[i][:,4] = delta
    return dict_robot

    
    

def simulation(dict_robot):
    collision = False
    X_robot0 = dict_robot[0]
    X_robot1 = dict_robot[1]
    X_robot2 = dict_robot[2]
    cov = np.ones((2,2)) * 0.0001 

    t = np.arange(0, len(X_robot0)-1, dt)
    t_coll = X_robot0.shape[0]
    min_distance = np.inf

    for i in range(20,X_robot0.shape[0]):
        
        rect1 = get_corners([X_robot0[i,0], X_robot0[i,1]], 2*h, 2*w,  X_robot0[i,2])
        
        # alpha = np.random.multivariate_normal([0, 0], cov)
        
        for j in range(2):
            X_robot = dict_robot[j+1] 
            
            
            if i >= X_robot.shape[0]:
                X_robot = np.append(X_robot, [X_robot[-1,:]], axis=0)
                dict_robot[j+1] = X_robot
            
            # X_robot[i,:2] += alpha
            rect2 = get_corners([X_robot[i,0], X_robot[i,1]], 2*h, 2*w,  X_robot[i,2])

            closest_pair, min_dist, _ = closest_points(rect1, rect2)
            min_distance = min(min_distance, min_dist)
            if min_dist < 0.05:
                # print(f"Collision detected with robot {j+1} at t={i*dt} ")
                collision = True
                t_coll = i
                break
        if collision:
            break
    
    # === Save the trajectories =====
    dict_robot[0] = X_robot0[20:t_coll+1, :]
    dict_robot[1] = X_robot1[20:t_coll+1, :]
    dict_robot[2] = X_robot2[20:t_coll+1, :]

    if t_coll < 6.0/dt:
        return dict_robot, None, None

    if collision:
        type_scenario = "collision"
    elif min_distance < 0.25:
        type_scenario = "close"
    else:
        type_scenario = "clear"

    return dict_robot, type_scenario, min_distance

def plot_results(dict_robot, VIZ = False):
    X_robot0 = dict_robot[0]
    X_robot1 = dict_robot[1]
    X_robot2 = dict_robot[2]
    t = np.arange(0, len(X_robot0)-1, dt)
    t_coll = dict_robot[0].shape[0]

    # === Plot_trajectories =====
    fig, ax = plt.subplots()
    
    
    for i in range(40, X_robot0.shape[0]):
        with sns.axes_style("whitegrid"):
            plot_robot(X_robot0[i,0],X_robot0[i,1],X_robot0[i,2], 2*h, 2*w, ax, color = sns.color_palette()[0], alpha = i/t_coll)
            if i == X_robot1.shape[0]:
                X_robot1 = np.append(X_robot1, [X_robot1[-1,:]], axis=0)
            plot_robot(X_robot1[i,0],X_robot1[i,1],X_robot1[i,2], 2*h, 2*w, ax, color = sns.color_palette()[1], alpha = i/t_coll)
            if i == X_robot2.shape[0]:
                X_robot2 = np.append(X_robot2, [X_robot2[-1,:]], axis=0)
            plot_robot(X_robot2[i,0],X_robot2[i,1],X_robot2[i,2], 2*h, 2*w, ax, color = sns.color_palette()[2], alpha = i/t_coll)
        
            if VIZ:
                    plt.pause(dt/10)
            # add control key to pause the simulation

        ax.axes.set_aspect('equal')
    ax.grid(True)
    


def main():
    # === Parameters =====

    # PLOT = True
    # CROSS = True
    # OVERTAKING = True
    INACTIVE_ROBOT = None

    parameters0 = {"SCENARIO": "twolanes", "OVERTAKING": False, "INACTIVE_ROBOT": None}
    parameters1 = {"SCENARIO": "twolanes","OVERTAKING": True, "INACTIVE_ROBOT": None}
    parameters2 = {"SCENARIO": "crossing", "CROSS": True, "DIR": 1, "INACTIVE_ROBOT": 1}
    parameters3 = {"SCENARIO": "crossing", "CROSS": True, "DIR": 0, "INACTIVE_ROBOT": 1}
    parameters4 = {"SCENARIO": "crossing", "CROSS": True, "DIR": 0, "INACTIVE_ROBOT": 2}
    parameters5 = {"SCENARIO": "crossing", "CROSS": True, "DIR": 1, "INACTIVE_ROBOT": 2}

    save_dir = "data4/"

    for parameters in tqdm.tqdm([parameters2,   ]): #parameters0,parameters3, parameters4, parameters5, parameters1,
        count = {"collision": 0 ,"close": 0,"clear":0}
        seed = 0
        # === Choice of scenario =====
        for seed in tqdm.tqdm(range(600)):
            if(count["collision"] < 30 or count["close"] < 30 or count["clear"] < 30):
                np.random.seed(seed) # 15 : collision # 8 : no collision
                dict_robot = init(parameters)
                dict_robot, type, min_distance = simulation(dict_robot)
                if type:
                    count[type] += 1
                    if count[type] <= 30:
                        # Save the trajectories
                        if parameters['SCENARIO'] == "twolanes":
                            dir = os.path.join(save_dir,f"{parameters['SCENARIO']}",f"{type}")
                        else:
                            dir = os.path.join(save_dir,f"{parameters['SCENARIO']}_{parameters['DIR']}_{parameters['INACTIVE_ROBOT']}",f"{type}")
                        if os.path.exists(dir) is False:
                            os.makedirs(dir)
                        pkl.dump(dict_robot, open(dir + f"/scenario_{count[type]}.pkl", "wb"))
                    else:
                        # randomize c in [1,20]
                        c = np.random.randint(1,30)
                        # Save the trajectories
                        if parameters['SCENARIO'] == "twolanes":
                            dir = os.path.join(save_dir,f"{parameters['SCENARIO']}",f"{type}")
                        else:
                            dir = os.path.join(save_dir,f"{parameters['SCENARIO']}_{parameters['DIR']}_{parameters['INACTIVE_ROBOT']}",f"{type}")

                        pkl.dump(dict_robot, open(dir + f"/scenario_{c}.pkl", "wb"))

                seed+=1
            else:
                break
        print(count)

if __name__ == "__main__":
    main()


