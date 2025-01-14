# Import necessary libraries
import time
import os
import numpy as np
from acados_settings_dev import acados_settings
from plotFcn import plotTrackProj, plotTrackProjfinal, plotDist, plotRes
from tracks.readDataFcn import getTrack
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm, eigh, inv
import tqdm
from time2spatial import transformProj2Orig, transformOrig2Proj
from utils import compute_ellipse_parameters
import pickle

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
    
    REAL_TIME_PLOTTING = True
    TRACK_FILE = "LMS_Track6.txt"
    PREDICTION_HORIZON = 5.0
    TIME_STEP = 0.1
    NUM_DISCRETIZATION_STEPS = int(PREDICTION_HORIZON / TIME_STEP)
    MAX_SIMULATION_TIME = 20.0
    REFERENCE_VELOCITY = 0.21
    REFERENCE_PROGRESS = REFERENCE_VELOCITY * PREDICTION_HORIZON
    DIST_THRESHOLD = 0.25

    x0 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # SAVE GIF and FIGURE
    SAVE_GIF_NAME = "sim"
    SAVE_FIG_NAME = "sim"

    # Multiple obstacles 
    N_OBSTACLES_MAX = 3 # Maximum number of obstacles considered
    OBSTACLE_WIDTH = 0.15
    OBSTACLE_LENGTH = 0.25
    # Initial positions of the obstacles [x, y, psi, v, length, width, sigmax, sigmay, sigmaxy]
    INITIAL_OBSTACLE_POSITION = np.array([0.0, 0.0, 0.0, 0.01, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0]) # 5e-4, 5e-3, 5e-8
    INITIAL_OBSTACLE_POSITION2 = np.array([1.25, -1.0, -np.pi/2+np.pi/9, 0.00, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0])
    INITIAL_OBSTACLE_POSITION3 = np.array([-1.25, -1.5, np.pi/2, 0.00, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0])
    INITIAL_OBSTACLES = [INITIAL_OBSTACLE_POSITION, INITIAL_OBSTACLE_POSITION2, INITIAL_OBSTACLE_POSITION3]
    N_OBSTACLES = len(INITIAL_OBSTACLES)
    assert N_OBSTACLES <= N_OBSTACLES_MAX, f"Number of obstacles should be less than or equal to {N_OBSTACLES_MAX}"
    

    # Q_SAFE = [1e5, 5e2, 1e-3, 1e-8, 1e-1, 5e-3, 5e-3, 5e2]
    # QE_SAFE = [ 5e5, 1e2, 1e-3, 1e-8, 5e-3, 2e-3]
    Q_SAFE = [5e3, 5e3, 1e-7, 1e-8, 1e-1, 5e-3, 1e-3, 5e-2]
    QE_SAFE = [ 5e3, 1e1, 1e0, 1e-8, 5e-3, 2e-3]

    # Q_OBB = [1e5, 1e-16, 1e-8, 1e-8, 1e-3, 5e-3, 5e-3, 5e2]
    # QE_OBB = [ 5e5, 1e-16, 1e-8, 1e-8, 5e-3, 2e-3]
    Q_OBB = [1e3, 1e-1, 1e-7, 1e-8, 1e-1, 5e-3, 1e-3, 5e-2]
    QE_OBB = [5e3, 1e-1, 1e0, 1e-8, 5e-3, 5e-3]

    folder_name =  f"results/{time.strftime('%Y%m%d-%H%M%S')}/"


    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    def __init__(self):
        self.cov_noise = np.diag([0.01**2, 0.05**2])
        print(f"Initial cov noise: {self.cov_noise}")
        self.Sref, self.constraint, self.model, self.acados_solver, self.nx, self.nu, self.Nsim, self.simX, self.predSimX, self.simU, self.sim_obb, self.predSim_obb, self.xN = self.initialize_simulation()
        self.s0 = self.model.x0[0]
        self.obstacles = self.INITIAL_OBSTACLES
        self.obstacles0 = self.INITIAL_OBSTACLES
        self.tcomp_sum = 0
        self.tcomp_max = 0
        self.tpred_sum = 0
        self.t_update = 0
        self.t_obb = 0
        self.min_dist = np.inf
        
        
        self.Sgoal = 12.0 # Goal position
        # self.n_xobb = self.INITIAL_OBSTACLE_POSITION.shape[0]
        self.n_params = self.n_xobb * self.N_OBSTACLES
    
    def save_params(self):
        """Save the parameters of the simulation."""
        
        # Save useful params in pickle file

        save_path = f"{self.folder_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + "/params.pkl", "wb") as f:
            names = ["Nsim", "MAX_SIMULATION_TIME", "TIME_STEP", "PREDICTION_HORIZON", "TRACK_FILE", "REFERENCE_VELOCITY",
                      "REFERENCE_PROGRESS", "DIST_THRESHOLD", "tcomp_sum", "tcomp_max", "Q_SAFE", "QE_SAFE", "Q_OBB", "QE_OBB",
                        "cov_noise"]
            params = [self.Nsim, self.MAX_SIMULATION_TIME, self.TIME_STEP, self.PREDICTION_HORIZON, self.TRACK_FILE,
                          self.REFERENCE_VELOCITY, self.REFERENCE_PROGRESS, self.DIST_THRESHOLD, self.tcomp_sum, self.tcomp_max,
                          self.Q_SAFE, self.QE_SAFE, self.Q_OBB, self.QE_OBB, self.cov_noise]
            dict = {name: param for name, param in zip(names, params)}

                
            # save all the parameters with names

            pickle.dump(dict, f)
        np.save(save_path + "/simX.npy", self.simX)
        np.save(save_path + "/simU.npy", self.simU)
        np.save(save_path + "/sim_obb.npy", self.sim_obb)
        np.save(save_path + "/predSimX.npy", self.predSimX)
        np.save(save_path + "/predSim_obb.npy", self.predSim_obb)
    
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
    


    def initialize_simulation(self):
        """Initialize the simulation parameters and structures."""
        track = self.TRACK_FILE
        Sref, _, _, _, _ = getTrack(track)
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
            xN[i] = constraint.pose(acados_solver.get(i, "x"))
        return Sref, constraint, model, acados_solver, nx, nu, Nsim, simX, predSimX, simU, sim_obb, predSim_obb, xN
    
    def closest_obstacle(self):
        """Find the closest obstacle to the car."""
        s_obb = np.array([transformOrig2Proj(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.TRACK_FILE)[0] for obstacle in self.obstacles])
        dists = np.array([self.dist(self.x0, obstacle) for obstacle in self.obstacles])
        
        valid_indices = np.where((s_obb%9.0 >= self.s0%9.0 - self.DIST_THRESHOLD))[0]
        
        if len(valid_indices) == 0:
            # print(f"Distance to obstacle: {dists[0]}")
            return None
        
        min_index = valid_indices[np.argmin(dists[valid_indices])]
        return min_index
    
    def is_relative_speed_ok(self, min_index, v_rel_threshold=0.1):
        """Check if the relative speed between the car and the obstacle is safe for overtaking."""

        if min_index is None:
            return self.REFERENCE_VELOCITY
        if self.x0[3] - self.obstacles[min_index][3] > v_rel_threshold :
            return self.REFERENCE_VELOCITY
        else:
            return self.obstacles[min_index][3]

    def run(self):
        """Run the simulation."""
        print(f"Initial state: {self.x0}")
        print(f"Initial pose: {self.constraint.pose(self.x0)}")
        print(f"Initial obstacle 1 pose: {self.constraint.obb_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial obstacle 2 pose: {self.constraint.obb1_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial obstacle 3 pose: {self.constraint.obb2_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial distance: {[np.linalg.norm(self.xN[0][:2] - np.array(self.obstacles)[k][:2]) for k in range(self.N_OBSTACLES)]}")


        for i in tqdm.tqdm(range(self.Nsim)):
            sref = self.s0 + self.REFERENCE_PROGRESS

            t = time.time()           
            t0_obb = time.time()
            dist_obstacle_N = np.array([[np.linalg.norm(self.xN[k][:2] - self.predSim_obb[m, i, k, :2]) for m in range(self.N_OBSTACLES)]
                                         for k in range(self.NUM_DISCRETIZATION_STEPS)])
            # if dist_obstacle_N[0] < self.min_dist:
            #     self.min_dist = dist_obstacle_N[0]
            self.t_obb += time.time() - t0_obb

            if np.any(dist_obstacle_N[0]) < 0.0:
                print("Collision detected")
                break

            t0_pred = time.time()
            obb_J = [self.update_obstacle_positions(j, self.obstacles) for j in range(self.NUM_DISCRETIZATION_STEPS+1)]

                
            # closest_obstacle = self.closest_obstacle()

            for j in range(self.NUM_DISCRETIZATION_STEPS):
                if i > 0:
                    X = self.acados_solver.get(j, "x")
                    self.xN[j] = self.constraint.pose(X)
                    self.predSimX[i, j, :] = X
                Q = self.Q_SAFE
                Qe = self.QE_SAFE
                
                # Update the obstacle positions for each obstacle at each time step
                # obb_j = self.update_obstacle_positions(j, self.obstacles)

                yref = np.array([self.s0 + (sref - self.s0) * j / self.NUM_DISCRETIZATION_STEPS, 0, 0, self.REFERENCE_VELOCITY, 0, 0, 0, 0])
                self.predSim_obb[:, i, j, :] = obb_J[j]
                self.acados_solver.set(j, "p", np.array(np.array(obb_J[j]).reshape((self.n_params))))

                # Update the reference state and cost weights
                self.acados_solver.set(j, "yref", yref)
                self.acados_solver.cost_set(j, 'W', np.diag(Q))
            
            # Update the obstacle positions for each obstacle for the last time step
            # obb_N = self.update_obstacle_positions(self.NUM_DISCRETIZATION_STEPS, self.obstacles)
            self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "p", 
                                   np.array(obb_J[self.NUM_DISCRETIZATION_STEPS]).reshape((self.n_params)))

            # Update the reference state and cost weights for the last time step
            yref_N = np.array([sref, 0, 0, self.REFERENCE_VELOCITY, 0, 0])
            self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "yref", yref_N)
            self.acados_solver.cost_set(self.NUM_DISCRETIZATION_STEPS, 'W', np.diag(Qe))
            
            self.tpred_sum += time.time() - t0_pred
            
            status = self.acados_solver.solve()


            t_update = time.time()
            self.x0 = self.acados_solver.get(0, "x")
            u0 = self.acados_solver.get(0, "u")
            self.simX[i, :] = self.x0
            self.simU[i, :] = u0

            
            self.obstacles = self.update_obstacle_positions(i, self.obstacles0)
            [self.obstacles[0][6], self.obstacles[1][6], self.obstacles[2][6]] = [self.obstacles0[k][6] for k in range(self.N_OBSTACLES)]
            [self.obstacles[0][7], self.obstacles[1][7], self.obstacles[2][7]] = [self.obstacles0[k][7] for k in range(self.N_OBSTACLES)]
            [self.obstacles[0][8], self.obstacles[1][8], self.obstacles[2][8]] = [self.obstacles0[k][8] for k in range(self.N_OBSTACLES)]
            self.sim_obb[:, i, :] = self.obstacles
            self.x0 = self.acados_solver.get(1, "x")
            self.acados_solver.set(0, "lbx", self.x0)
            self.acados_solver.set(0, "ubx", self.x0)
            self.s0 = self.x0[0]

            self.t_update += time.time() - t_update

            if self.x0[0] > self.Sgoal + 0.1:
                print("GOAL REACHED")
                    
                break
            
            elapsed = time.time() - t
            self.tcomp_sum += elapsed
            if elapsed > self.tcomp_max:
                self.tcomp_max = elapsed
        
        for j in range(i, self.Nsim):
            self.simX[j, :] = self.simX[i, :]
            self.simU[j, :] = self.simU[i, :]
            self.sim_obb[:, j, :] = self.sim_obb[:, i, :]
    
    def plot_results(self):
        print(f"Average computation time: {self.tcomp_sum / self.Nsim}")
        print(f"Maximum computation time: {self.tcomp_max}")
        print(f"Average time for prediction {self.tpred_sum / self.Nsim}")
        print(f"Average time for obstacle detect and prune {self.t_obb/self.Nsim}")
        print(f"Average time for update {self.t_update/self.Nsim}")
        print(f"Average speed: {np.average(self.simX[:, 3])} m/s")

        t = np.linspace(0.0, self.Nsim * self.PREDICTION_HORIZON / self.NUM_DISCRETIZATION_STEPS, self.Nsim)

        plotTrackProjfinal(self.simX, self.sim_obb, # simulated trajectories
                            self.predSimX, self.predSim_obb, # predicted trajectories
                            self.TRACK_FILE, self.folder_name, self.SAVE_FIG_NAME )#
        
        plotDist(self.simX, self.sim_obb, self.constraint, t, self.folder_name, "dist")

        plotRes(self.simX, self.simU, t)

        plotTrackProj(self.simX,self.sim_obb, # simulated trajectories
                      self.predSimX, self.predSim_obb, # predicted trajectories
                        self.TRACK_FILE, self.folder_name,self.SAVE_GIF_NAME ) #


        if os.environ.get("ACADOS_ON_CI") is None:
            plt.show()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
    
    sim.plot_results() 
    # get date and time
    
    sim.save_params()
