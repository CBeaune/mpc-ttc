# Import necessary libraries
import time
import os
import numpy as np
from acados_settings_dev import acados_settings, acados_settings_ttc

from plotFcn import plotTrackProj, plotTrackProjfinal, plotDist, plotRes, plotTTC
from tracks.readDataFcn import getTrack
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm, eigh, inv
import tqdm
from time2spatial import transformProj2Orig, transformOrig2Proj
from utils import compute_ellipse_parameters
import pickle
import json

class Simulation:
    """
    A class to represent a simulation of a race car navigating a track with obstacles.
    Attributes
    ----------
    TRACK_FILE : str
        The file name of the track.
    PREDICTION_HORIZON : float
        The prediction horizon for the simulation.
    TIME_STEP : float
        The time step for the simulation.
    NUM_DISCRETIZATION_STEPS : int
        The number of discretization steps.
    MAX_SIMULATION_TIME : float
        The maximum simulation time.
    REFERENCE_VELOCITY : float
        The reference velocity of the car.
    REFERENCE_PROGRESS : float
        The reference progress of the car.
    OBSTACLE_WIDTH : float
        The width of the obstacle.
    OBSTACLE_LENGTH : float
        The length of the obstacle.
    INITIAL_OBSTACLE_POSITION : np.array
        The initial position of the obstacle.
    DIST_THRESHOLD : float
        The distance threshold for obstacle avoidance.
    N_OBSTACLES : int
        The maximum number of obstacles considered.
    Q_SAFE : list
        The weight matrix for safe conditions.
    QE_SAFE : list
        The weight matrix for safe conditions at the end of the horizon.
    Q_OBB : list
        The weight matrix for obstacle avoidance.
    QE_OBB : list
        The weight matrix for obstacle avoidance at the end of the horizon.
    Zl_SAFE : np.array
        The slack variable for safe conditions.
    Zl_OBB : np.array
        The slack variable for obstacle avoidance.
    Methods
    -------
    __init__():
        Initializes the simulation parameters and structures.
    dist(X, X_obb):
        Calculates the distance between the car and the obstacle.
    evolution_function(X_obb0, i, cov_noise=np.zeros((2, 2))):
        Evolves the obstacle position over time.
    initialize_simulation():
        Initializes the simulation parameters and structures.
    run():
        Runs the simulation.
    plot_results():
        Plots the results of the simulation.
    """

    
    # TRACK_FILE = "LMS_Track6.txt"
    # PREDICTION_HORIZON = 5.0
    # TIME_STEP = 0.1
    # NUM_DISCRETIZATION_STEPS = int(PREDICTION_HORIZON / TIME_STEP)
    # MAX_SIMULATION_TIME = 20.0
    
    # DIST_THRESHOLD = 1.0
    # # Multiple obstacles 
    # N_OBSTACLES_MAX = 3 # Maximum number of obstacles considered
    # OBSTACLE_WIDTH = 0.15
    # OBSTACLE_LENGTH = 0.25
    # # Initial positions of the obstacles [x, y, psi, v, length, width, sigmax, sigmay, sigmaxy]
    

    # Q_SAFE = [2e3, 5e2, 1e-7, 1e-8, 1e-1, 5e-3, 1e-1, 5e1]
    # QE_SAFE = [ 5e3, 1e1, 1e0, 1e-8, 5e-3, 2e1]


    # Q_OBB = [1e3, 1e-1, 1e-7, 1e-8, 1e-1, 5e-3, 1e-3, 5e-2]
    # QE_OBB = [5e3, 1e-1, 1e0, 1e-8, 5e-3, 5e-3]

    # Sgoal = 1.75 # Goal position




    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    def __init__(self, SAVE, params_file, scenario=1, seed=0):
        self.SCENARIO=scenario
        self.seed = seed

        # Load parameters from JSON file
        with open(f'{params_file}', 'r') as f:
            params = json.load(f)
        self.SCENARIO = params["scenario"]
        self.TRACK_FILE = params["TRACK_FILE"]
        self.PREDICTION_HORIZON = params["PREDICTION_HORIZON"]
        self.TIME_STEP = params["TIME_STEP"]
        self.NUM_DISCRETIZATION_STEPS = int(self.PREDICTION_HORIZON / self.TIME_STEP)
        self.MAX_SIMULATION_TIME = params["MAX_SIMULATION_TIME"]
        self.DIST_THRESHOLD = params["DIST_THRESHOLD"]
        self.N_OBSTACLES_MAX = params["N_OBSTACLES_MAX"]
        self.OBSTACLE_WIDTH = params["OBSTACLE_WIDTH"]
        self.OBSTACLE_LENGTH = params["OBSTACLE_LENGTH"]
        self.Q_SAFE = params["Q_SAFE"]
        self.QE_SAFE = params["QE_SAFE"]
        self.Q_OBB = params["Q_OBB"]
        self.QE_OBB = params["QE_OBB"]
        self.Sgoal = params["Sgoal"]
        self.cov_noise = np.diag(params["cov_noise"]) # np.diag([0.05**2, 0.05**2]) #np.diag([0.0**2, 0.0**2])
        self.ttc =bool(params["ttc"])
        print(f" ttc : {self.ttc}")
    
        if self.SCENARIO == 1:
            self.x0 = np.array([np.random.uniform(-1.0,-0.8), np.random.uniform(0.0,0.0), 0.0, 0.0, 0.0, 0.0])
            self.REFERENCE_VELOCITY = np.random.uniform(0.21,0.21)
            self.INITIAL_OBSTACLE_POSITION = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1),
                                                0.0, np.random.uniform(0.0, 0.05),
                                                self.OBSTACLE_LENGTH, self.OBSTACLE_WIDTH, 0, 0, 0]) # 5e-4, 5e-3, 5e-8
            self.INITIAL_OBSTACLE_POSITION2 = np.array([np.random.uniform(0.9, 1.1), np.random.uniform(0.2, 0.3),
                                                    -np.pi, np.random.uniform(0.1, 0.25), 
                                                self.OBSTACLE_LENGTH, self.OBSTACLE_WIDTH, 0, 0, 0])
        elif self.SCENARIO == 3:
            self.x0 = np.array([np.random.uniform(0.0, 0.5), np.random.uniform(0.0,0.0), 0.0, 0.0, 0.0, 0.0])
            self.REFERENCE_VELOCITY = np.random.uniform(0.21,0.21)
            self.INITIAL_OBSTACLE_POSITION = np.array([np.random.uniform(1.45, 1.55), np.random.uniform(0.5, 1.0),
                                                -np.pi/2, np.random.uniform(0.0, 0.15),
                                                self.OBSTACLE_LENGTH, self.OBSTACLE_WIDTH, 0, 0, 0]) # 5e-4, 5e-3, 5e-8
            self.INITIAL_OBSTACLE_POSITION2 = np.array([np.random.uniform(1.7, 1.8), np.random.uniform(-1, -0.5),
                                                    np.pi/2, np.random.uniform(0.0, 0.1), 
                                                self.OBSTACLE_LENGTH, self.OBSTACLE_WIDTH, 0, 0, 0])
        elif self.SCENARIO == 2:
            self.x0 = np.array([np.random.uniform(-0.3, 0.5), np.random.uniform(0.0,0.0), 0.0, 0.0, 0.0, 0.0])
            self.REFERENCE_VELOCITY = np.random.uniform(0.21,0.21)
            self.INITIAL_OBSTACLE_POSITION = np.array([np.random.uniform(1.2,1.3), np.random.uniform(0.5, 1.0),
                                                -np.pi/2, np.random.uniform(0.0, 0.2),
                                                self.OBSTACLE_LENGTH, self.OBSTACLE_WIDTH, 0, 0, 0]) # 5e-4, 5e-3, 5e-8
            self.INITIAL_OBSTACLE_POSITION2 = np.array([np.random.uniform(1.57,1.6), np.random.uniform(-1.0, -0.5),
                                                    np.pi/2, np.random.uniform(0.0, 0.1), 
                                                self.OBSTACLE_LENGTH, self.OBSTACLE_WIDTH, 0, 0, 0])

        self.REFERENCE_PROGRESS = self.REFERENCE_VELOCITY * self.PREDICTION_HORIZON

        # INITIAL_OBSTACLE_POSITION3 = np.array([1.556, -0.5, np.pi/2, 0.00, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0])
        self.INITIAL_OBSTACLES = [self.INITIAL_OBSTACLE_POSITION,
                                   self.INITIAL_OBSTACLE_POSITION2, 
                                   self.INITIAL_OBSTACLE_POSITION,] #, INITIAL_OBSTACLE_POSITION3
        self.N_OBSTACLES = len(self.INITIAL_OBSTACLES)
        assert self.N_OBSTACLES == self.N_OBSTACLES_MAX, f"Number of obstacles should be equal to {self.N_OBSTACLES_MAX}"




        [self.Sref, self.constraint, self.model, self.acados_solver, self.nx, self.nu, self.Nsim,
            self.simX, self.predSimX, self.simU, self.sim_obb, self.predSim_obb, self.xN] = self.initialize_simulation(self.ttc)
        self.s0 = self.model.x0[0]
        self.obstacles = self.INITIAL_OBSTACLES
        self.obstacles0 = self.INITIAL_OBSTACLES

        # Log variables
        self.tcomp = np.zeros(self.Nsim)
        self.tcomp_sum = 0
        self.tcomp_max = 0
        self.tpred_sum = 0
        self.t_update = 0
        self.t_obb = 0
        self.final_t = self.Nsim
        self.min_dist = np.inf
        self.freeze = 0 # number of time steps the car is stuck
        self.relaunch = 0 # number of times the car is relaunched after 10 time steps stuck
        self.collision = False


        self.track_width = 0.3

        self.closest_obstacle_index = np.ones(self.Nsim)*-1
        
        

        # self.n_xobb = self.INITIAL_OBSTACLE_POSITION.shape[0]
        self.n_params = self.n_xobb * self.N_OBSTACLES

        if SAVE:
            self.folder_name =  f"results/ttc/scenario_{self.SCENARIO}/seed_{self.seed}" if self.ttc \
                else f"results/dist/scenario_{self.SCENARIO}/seed_{self.seed}/"
            # SAVE GIF and FIGURE
            self.SAVE_GIF_NAME = "sim"
            self.SAVE_FIG_NAME = "sim"
        else:
            self.folder_name =  None
            self.SAVE_GIF_NAME = None
            self.SAVE_FIG_NAME = None
    
    def save_params(self):
        """Save the parameters of the simulation."""
        
        # Save useful params in pickle file

        save_path = f"{self.folder_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + "params.pkl", "wb") as f:
            
            dict = {# Simulation parameters
                    "Nsim": self.Nsim,
                    "MAX_SIMULATION_TIME": self.MAX_SIMULATION_TIME,
                    "TIME_STEP": self.TIME_STEP,
                    "PREDICTION_HORIZON": self.PREDICTION_HORIZON,
                    "TRACK_FILE": self.TRACK_FILE,
                    "REFERENCE_VELOCITY": self.REFERENCE_VELOCITY,
                    "REFERENCE_PROGRESS": self.REFERENCE_PROGRESS,
                    "DIST_THRESHOLD": self.DIST_THRESHOLD,
                    "tcomp_sum": self.tcomp_sum,
                    "tcomp_max": self.tcomp_max,
                    "Q_SAFE": self.Q_SAFE,
                    "QE_SAFE": self.QE_SAFE,
                    "Q_OBB": self.Q_OBB,
                    "QE_OBB": self.QE_OBB,
                    "cov_noise": self.cov_noise,
                    "Sref": self.Sref,
                    # Log variables
                    "final_t": self.final_t,
                    "min_dist": self.min_dist,
                    "freeze": self.freeze,
                    "collision": self.collision,
                    "tcomp": self.tcomp,
                    "seed": self.seed,
                    # Simulation variables
                    "simX": self.simX,
                    "simU": self.simU,
                    "sim_obb": self.sim_obb,
                    "predSimX": self.predSimX,
                    "predSim_obb": self.predSim_obb,
                    }


            pickle.dump(dict, f)
    
    def set_init_pose(self, x0):
        """Set the initial pose of the car."""
        self.x0 = x0

    def dist(self, X, X_obb):
        """Calculate the distance between the car and the obstacle."""
        assert len(X) == 6, f"X has to be 6 dimensional, got {len(X)}"
        assert len(X_obb) == self.n_xobb, f"X_obb has to be {self.n_xobb} dimensional, got {len(X_obb)}"
        x, y, psi, v = transformProj2Orig(X[0], X[1], X[2], X[3])
        X_c = np.array([x, y, psi, v], dtype=object)
        return np.sqrt((X_c[0] - X_obb[0])**2 + (X_c[1] - X_obb[1])**2)

    def evolution_function(self, X_obb0, i, cov_noise=np.zeros((2, 2))):
        """Evolve the obstacle position over time."""
        noise = np.random.multivariate_normal([0, 0], cov_noise)
        x = X_obb0[0] + i * self.TIME_STEP * X_obb0[3] * np.cos(X_obb0[2]) 
        y = X_obb0[1] + i * self.TIME_STEP * X_obb0[3] * np.sin(X_obb0[2]) 
        cov  = np.array([[X_obb0[6], X_obb0[8]], [X_obb0[8], X_obb0[7]]]) + i*self.TIME_STEP*cov_noise
        return [x, y, X_obb0[2], X_obb0[3], X_obb0[4], X_obb0[5], cov[0,0], cov[1,1],cov[1,0]]
    
    def update_obstacle_positions(self, i, obstacles):
        """Update the positions of all obstacles."""
        return [self.evolution_function(obstacle, i, self.cov_noise) for obstacle in obstacles]
    


    def initialize_simulation(self, ttc=True):
        """Initialize the simulation parameters and structures."""
        track = self.TRACK_FILE
        Sref, _, _, _, _ = getTrack(track)
        self.L_TRACK = Sref[-1]
        if ttc:
            constraint, model, acados_solver = acados_settings_ttc(self.PREDICTION_HORIZON, self.NUM_DISCRETIZATION_STEPS, track,
                                                            self.x0)
        else:
            constraint, model, acados_solver = acados_settings(self.PREDICTION_HORIZON, self.NUM_DISCRETIZATION_STEPS, track,
                                                            self.x0)
        nx = model.x.rows()
        nu = model.u.rows()
        Nsim = int(self.MAX_SIMULATION_TIME * self.NUM_DISCRETIZATION_STEPS / self.PREDICTION_HORIZON)
        simX = np.zeros((Nsim, nx))
        predSimX = np.zeros((Nsim, self.NUM_DISCRETIZATION_STEPS, nx))
        simU = np.zeros((Nsim, nu))
        self.n_xobb = self.INITIAL_OBSTACLE_POSITION.shape[0]

        sim_obb = np.zeros((self.N_OBSTACLES, Nsim, self.n_xobb))
        sim_obb[:, :, 4] = self.OBSTACLE_LENGTH
        sim_obb[:, :, 5] = self.OBSTACLE_WIDTH
        sim_obb[:, :, 6] = self.INITIAL_OBSTACLE_POSITION[6] # sigmax
        sim_obb[:, :, 7] = self.INITIAL_OBSTACLE_POSITION[7] # sigmay
        sim_obb[:, :, 8] = self.INITIAL_OBSTACLE_POSITION[8] # sigmaxy

        predSim_obb = np.zeros((self.N_OBSTACLES, Nsim, self.NUM_DISCRETIZATION_STEPS, self.n_xobb))
        predSim_obb[:, :, :, 4] = self.OBSTACLE_LENGTH
        predSim_obb[:, :, :, 5] = self.OBSTACLE_WIDTH
        predSim_obb[:, :, :, 6] = self.INITIAL_OBSTACLE_POSITION[6] # sigmax
        predSim_obb[:, :, :, 7] = self.INITIAL_OBSTACLE_POSITION[7] # sigmay
        predSim_obb[:, :, :, 8] = self.INITIAL_OBSTACLE_POSITION[8] # sigmaxy

        xN = np.zeros((self.NUM_DISCRETIZATION_STEPS, 3))
        for i in range(self.NUM_DISCRETIZATION_STEPS):
            xN[i,:] = constraint.pose(acados_solver.get(i, "x"))
        return [Sref, constraint, model, acados_solver, nx, nu, Nsim, simX, predSimX, simU, sim_obb, predSim_obb, xN]
    
    def closest_obstacle(self):
        """Find the closest obstacle to the car front."""
        transforms = np.array([transformOrig2Proj(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.TRACK_FILE) for obstacle in self.obstacles])
        # print(transforms)
        s_obb = transforms[:, 0]
        n_obb = transforms[:, 1]
        valid_indexes_s = np.where(s_obb > self.s0)[0]
        # print("Valid indexes s ", valid_indexes_s)
        valid_indexes_dist = np.where(np.abs(s_obb-self.s0)<self.DIST_THRESHOLD)[0]
        # print("Valid indexes dist ", valid_indexes_dist)
        valid_indexes_n = np.where(n_obb < self.track_width/2)[0]
        # print("Valid indexes n ", valid_indexes_n)
        from functools import reduce
        valid_indexes = reduce(np.intersect1d, (valid_indexes_s,  valid_indexes_dist, valid_indexes_n))
        # print("Valid indexes ", valid_indexes)
        if len(valid_indexes) == 0:
            # print("No valid indexes")
            return -1
        # if index is not valid, s_obb = np.inf
        s_obb = np.array([np.inf if k not in valid_indexes else s_obb[k] for k in range(self.N_OBSTACLES)])
        min_index = np.argmin(s_obb)

        return min_index
    

    def run(self):
        """Run the simulation."""
        print(f"Initial state: {self.x0}")
        print(f"Initial pose: {self.constraint.pose(self.x0)}")
        print(f"Initial obstacle 1 pose: {self.constraint.obb_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial obstacle 2 pose: {self.constraint.obb1_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial obstacle 3 pose: {self.constraint.obb2_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial distance: {[np.linalg.norm(self.xN[0][:2] - np.array(self.obstacles)[k][:2]) for k in range(self.N_OBSTACLES)]}")

        for k in range(self.N_OBSTACLES):
            new_obstacles = np.array([self.update_obstacle_positions(j, self.obstacles) for j in range(self.NUM_DISCRETIZATION_STEPS)])
            self.predSim_obb[k, 0, :, :] = new_obstacles[:,k,:]
        
        cov = np.diag([0.01**2, 0.01**2, 0, 0, 0, 0, 0, 0, 0])

        self.sim_obb[:, 0, :] = self.obstacles + np.random.multivariate_normal(np.zeros((9,)), cov, self.N_OBSTACLES)
        for i in tqdm.tqdm(range(self.Nsim)):


            t = time.time()           
            t0_obb = time.time()
            dist_obstacle_N = np.array([[np.linalg.norm(self.xN[k][:2] - self.predSim_obb[m, i, k, :2]) for m in range(self.N_OBSTACLES)]
                                         for k in range(self.NUM_DISCRETIZATION_STEPS)])
            real_dist_obstacle_N = np.array([np.linalg.norm(self.simX[i, :2] - self.sim_obb[m, i, :2]) for m in range(self.N_OBSTACLES)])
            if np.any(real_dist_obstacle_N) < self.min_dist:
                self.min_dist = np.min(dist_obstacle_N[0])
            self.t_obb += time.time() - t0_obb

            if np.any(dist_obstacle_N[0]) < 0.15:
                print("Collision detected")
                self.collision = True
                break

            t0_pred = time.time()
            obb_J = [self.update_obstacle_positions(j, self.obstacles) for j in range(self.NUM_DISCRETIZATION_STEPS+1)]

            idx = self.closest_obstacle()
            # print(f"Closest obstacle index: {idx}")
            self.closest_obstacle_index[i] = idx


            for j in range(self.NUM_DISCRETIZATION_STEPS):
                # if idx is not None:

                #     if self.REFERENCE_VELOCITY - self.obstacles[idx][3] < 0.1:
                #         # print(f"Closest obstacle index: {idx}")
                #         vref = self.obstacles[idx][3]
                #     else :
                #         vref = self.REFERENCE_VELOCITY
                #     sref = self.s0 + vref * self.PREDICTION_HORIZON
                # else:
                vref = self.REFERENCE_VELOCITY
                sref =  self.s0 + self.REFERENCE_PROGRESS

                if i > 0:
                    X = self.acados_solver.get(j, "x")
                    self.xN[j,:] = self.constraint.pose(X)
                    self.predSimX[i, j, :] = X
                Q = self.Q_SAFE
                Qe = self.QE_SAFE
                
                # Update the obstacle positions for each obstacle at each time step

                yref = np.array([self.s0 + (sref - self.s0) * j / self.NUM_DISCRETIZATION_STEPS, 0, 0, vref, 0, 0, 0, 0])
                self.predSim_obb[:, i, j, :] = obb_J[j]
                self.acados_solver.set(j, "p", np.array(np.array(obb_J[j]).reshape((self.n_params))))

                # Update the reference state and cost weights
                self.acados_solver.set(j, "yref", yref)
                self.acados_solver.cost_set(j, 'W', np.diag(Q))
            
            # Update the obstacle positions for each obstacle for the last time step
            self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "p", 
                                   np.array(obb_J[self.NUM_DISCRETIZATION_STEPS]).reshape((self.n_params)))

            # Update the reference state and cost weights for the last time step
            yref_N = np.array([sref, 0, 0, vref, 0, 0])
            self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "yref", yref_N)
            self.acados_solver.cost_set(self.NUM_DISCRETIZATION_STEPS, 'W', np.diag(Qe))
            
            
            
            self.tpred_sum += time.time() - t0_pred
            
            status = self.acados_solver.solve()
            x0 = self.acados_solver.get(1, "x")
            u0 = self.acados_solver.get(0, "u")
            self.simX[i, :] = self.acados_solver.get(0, "x")
            current_state = self.simX[i-1, :] if i > 0 else self.x0
            self.simU[i, :] = u0
            if status != 0:
                self.freeze += 1
                # print(f"acados returned status {status} at time  {i * self.TIME_STEP}")
                self.acados_solver.reset()
                self.acados_solver.load_iterate("acados_ocp_iterate.json")
                if self.freeze > 10:
                    print("Car is stuck")
                    self.relaunch += 1
                    self.freeze = 0
                    relaunch_time = time.time()
                    if self.ttc:
                        self.constraint, self.model, self.acados_solver = acados_settings_ttc(self.PREDICTION_HORIZON,
                                                                                               self.NUM_DISCRETIZATION_STEPS,
                                                                                                self.TRACK_FILE,self.x0)
                    else:
                        self.constraint, self.model, self.acados_solver = acados_settings(self.PREDICTION_HORIZON,
                                                                                               self.NUM_DISCRETIZATION_STEPS,
                                                                                                self.TRACK_FILE,self.x0)
                    relaunch_delay = time.time() - relaunch_time
                    print(f"Relaunch delay: {relaunch_delay}")
                self.acados_solver.set(0, "lbx", current_state)
                self.acados_solver.set(0, "ubx", current_state)
                self.acados_solver.set(0, "x", current_state)

                vref = 0.1

                sref = self.s0 + vref * self.PREDICTION_HORIZON
                Q = [1, 100, 1, 1, 1, 1, 100,100]
                Qe = [1, 100, 1, 1, 100, 100]

                for j in range(self.NUM_DISCRETIZATION_STEPS):
                    self.acados_solver.set(j, "x", current_state)
                    self.acados_solver.set(j, "p", np.array(np.array(obb_J[j]).reshape((self.n_params))))
                    self.acados_solver.set(j, "yref", np.array([self.s0 + (sref - self.s0) * j / self.NUM_DISCRETIZATION_STEPS,
                                                                 0, 0, vref, 0, 0, 0, 0]))
                    self.acados_solver.cost_set(j, 'W', np.diag(Q))
                yref_N = np.array([sref, 0, 0, vref, 0, 0])
                self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "yref", yref_N)
                self.acados_solver.cost_set(self.NUM_DISCRETIZATION_STEPS, 'W', np.diag(Qe))
                status = self.acados_solver.solve()
                x0 = self.acados_solver.get(1, "x")
                u0 = self.acados_solver.get(0, "u")
                self.simX[i, :] = self.acados_solver.get(0, "x")
                current_state = self.simX[i-1, :] if i > 0 else self.x0
                self.simU[i, :] = u0

            self.acados_solver.store_iterate("acados_ocp_iterate.json", overwrite=True, verbose=False)
            self.x0 = x0
            t_update = time.time()
            

            t_update = time.time()
            

            
            self.obstacles = self.update_obstacle_positions(i, self.obstacles0)
            [self.obstacles[0][6], self.obstacles[1][6], self.obstacles[2][6]] = [self.obstacles0[k][6] for k in range(self.N_OBSTACLES)]
            [self.obstacles[0][7], self.obstacles[1][7], self.obstacles[2][7]] = [self.obstacles0[k][7] for k in range(self.N_OBSTACLES)]
            [self.obstacles[0][8], self.obstacles[1][8], self.obstacles[2][8]] = [self.obstacles0[k][8] for k in range(self.N_OBSTACLES)]
            self.sim_obb[:, i, :] = self.obstacles + np.random.multivariate_normal(np.zeros(9), cov, self.N_OBSTACLES)
            
            self.acados_solver.set(0, "lbx", self.x0)
            self.acados_solver.set(0, "ubx", self.x0)
            self.s0 = self.x0[0]

            self.t_update += time.time() - t_update

            if self.x0[0] > self.Sgoal + 0.1:
                print("GOAL REACHED")
                self.final_t = i
                    
                break
            
            elapsed = time.time() - t
            self.tcomp[i] = elapsed
            self.tcomp_sum += elapsed
            if elapsed > self.tcomp_max:
                self.tcomp_max = elapsed
        
        for j in range(i, self.Nsim):
            self.simX[j, :] = self.simX[i, :]
            self.simU[j, :] = self.simU[i, :]
            self.sim_obb[:, j, :] = self.sim_obb[:, i, :]
    
    def plot_results(self):
        print(f"Average computation time: {self.tcomp_sum / self.final_t *1e3} ms")
        print(f"Maximum computation time: {self.tcomp_max*1e3} ms")
        # print(f"Average time for prediction {self.tpred_sum / self.Nsim}")
        # print(f"Average time for obstacle detect and prune {self.t_obb/self.Nsim}")
        # print(f"Average time for update {self.t_update/self.Nsim}")
        print(f"Reference speed: {self.REFERENCE_VELOCITY} m/s")
        print(f"Average speed: {np.average(self.simX[:self.final_t, 3])} m/s")
        print(self.final_t)
        t = np.linspace(0.0, self.final_t * self.PREDICTION_HORIZON / self.NUM_DISCRETIZATION_STEPS, self.final_t)

        plotTrackProjfinal(self.simX, self.sim_obb, # simulated trajectories
                            self.predSimX, self.predSim_obb, # predicted trajectories
                            self.TRACK_FILE,self.folder_name, self.SAVE_FIG_NAME,
                              scenario= self.SCENARIO )#
        
        plotDist(self.simX, self.sim_obb, self.constraint, t, self.folder_name, "dist")#

        plotTTC(self.simX, self.sim_obb, self.constraint, t, self.folder_name, "results" )

        plotRes(self.simX, self.simU, t, self.folder_name, "results")

        plotTrackProj(self.simX,self.sim_obb, # simulated trajectories
                      self.predSimX, self.predSim_obb, # predicted trajectories
                        self.TRACK_FILE, self.folder_name, self.SAVE_GIF_NAME, idx = self.closest_obstacle_index,
                        scenario=self.SCENARIO  ) #


        if os.environ.get("ACADOS_ON_CI") is None:
            plt.show()

if __name__ == "__main__":
    params_file = 'params/scenario1.json'
    
    params_file = 'params/scenario2.json'
    params_file = 'params/scenario3.json'
    params_file = 'params/scenario1_ttc.json'
    params_file = 'params/scenario2_ttc.json'
    params_file = 'params/scenario3_ttc.json'
    for params_file in [ 'params/scenario1.json', 
                        # 'params/scenario2.json', 
                        # 'params/scenario3.json', 
                        # 'params/scenario1_ttc.json',
                        #   'params/scenario2_ttc.json',
                        #     'params/scenario3_ttc.json'
                        ]:
        for seed in tqdm.tqdm(range(1), desc="Seeds"):
            np.random.seed(seed)
            
            sim = Simulation(SAVE=True, params_file=params_file, seed=seed)
            res = sim.run()
            
            sim.plot_results() 
            if res == 0 :
                print("Simulation failed")
            else:
                if sim.folder_name is not None:
                    sim.save_params()
