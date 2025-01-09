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
 
    TRACK_FILE = "LMS_Track6.txt"
    PREDICTION_HORIZON = 3.0
    TIME_STEP = 0.1
    NUM_DISCRETIZATION_STEPS = int(PREDICTION_HORIZON / TIME_STEP)
    MAX_SIMULATION_TIME = 30.0
    REFERENCE_VELOCITY = 0.22
    REFERENCE_PROGRESS = REFERENCE_VELOCITY * PREDICTION_HORIZON
    DIST_THRESHOLD = 0.5

    x0 = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # SAVE GIF and FIGURE
    SAVE_GIF_NAME = "sim"
    SAVE_FIG_NAME = "sim"

    # Multiple obstacles 
    N_OBSTACLES_MAX = 3 # Maximum number of obstacles considered
    OBSTACLE_WIDTH = 0.15
    OBSTACLE_LENGTH = 0.25
    # Initial positions of the obstacles [x, y, psi, v, length, width, sigmax, sigmay, sigmaxy]
    INITIAL_OBSTACLE_POSITION = np.array([0.0, 0.0, 0.0, 0.0, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0])
    INITIAL_OBSTACLE_POSITION2 = np.array([1.25, -0.5, -np.pi/2, 0.00, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0])
    INITIAL_OBSTACLE_POSITION3 = np.array([-1.25, -1.5, np.pi/2, 0.00, OBSTACLE_LENGTH, OBSTACLE_WIDTH, 0, 0, 0])
    INITIAL_OBSTACLES = [INITIAL_OBSTACLE_POSITION, INITIAL_OBSTACLE_POSITION2, INITIAL_OBSTACLE_POSITION3]
    N_OBSTACLES = len(INITIAL_OBSTACLES)
    assert N_OBSTACLES <= N_OBSTACLES_MAX, f"Number of obstacles should be less than or equal to {N_OBSTACLES_MAX}"
    

    Q_SAFE = [1e5, 5e3, 1e-1, 1e-6, 1e-1, 5e-3, 5e-3, 1e2]
    QE_SAFE = [5e5, 1e5, 1e-1, 1e-6, 5e-3, 2e-3]
    Q_OBB = [1e3, 5e-8, 1e-8, 1e-8, 1e-3, 5e-3, 5e-3, 1e2]
    QE_OBB = [5e3, 1e3, 1e-1, 1e-8, 5e-3, 2e-3]

    Zl_SAFE = 0.01 * np.ones((5,))
    Zl_SAFE[4] = 100
    Zl_OBB = 0.1 * np.ones((5,))

    def __init__(self):
        self.cov_noise = np.diag([0.01, 0.02])
        self.Sref, self.constraint, self.model, self.acados_solver, self.nx, self.nu, self.Nsim, self.simX, self.predSimX, self.simU, self.sim_obb, self.predSim_obb, self.xN = self.initialize_simulation()
        self.s0 = self.model.x0[0]
        self.obstacles = self.INITIAL_OBSTACLES
        self.tcomp_sum = 0
        self.tcomp_max = 0
        self.min_dist = np.inf
        
        self.Sgoal = 8.0 # Goal position
        # self.n_xobb = self.INITIAL_OBSTACLE_POSITION.shape[0]
        self.n_params = self.n_xobb * self.N_OBSTACLES
    
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
        x = X_obb0[0] + i * self.TIME_STEP * X_obb0[3] * np.cos(X_obb0[2]) + noise[0]
        y = X_obb0[1] + i * self.TIME_STEP * X_obb0[3] * np.sin(X_obb0[2]) + noise[1]
        return [x, y, X_obb0[2], X_obb0[3], X_obb0[4], X_obb0[5], cov_noise[0,0], cov_noise[1,1],cov_noise[1,0]]
    
    def update_obstacle_positions(self, i):
        """Update the positions of all obstacles."""
        updated_positions = []
        for obstacle in self.obstacles:
            updated_positions.append(self.evolution_function(obstacle, i, self.cov_noise))
        return updated_positions
    


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
        sim_obb[:, :, 6] = self.cov_noise[0, 0]
        sim_obb[:, :, 7] = self.cov_noise[1, 1]
        sim_obb[:, :, 8] = self.cov_noise[1, 0]

        predSim_obb = np.zeros((self.N_OBSTACLES, Nsim, self.NUM_DISCRETIZATION_STEPS, self.n_xobb))
        predSim_obb[:, :, :, 4] = self.OBSTACLE_LENGTH
        predSim_obb[:, :, :, 5] = self.OBSTACLE_WIDTH
        predSim_obb[:, :, :, 6] = self.cov_noise[0, 0]
        predSim_obb[:, :, :, 7] = self.cov_noise[1, 1]
        predSim_obb[:, :, :, 8] = self.cov_noise[1, 0]

        xN = np.zeros((self.NUM_DISCRETIZATION_STEPS, nx))
        for i in range(self.NUM_DISCRETIZATION_STEPS):
            xN[i] = model.x0
        return Sref, constraint, model, acados_solver, nx, nu, Nsim, simX, predSimX, simU, sim_obb, predSim_obb, xN
    
    def closest_obstacle(self):
        """Find the closest obstacle to the car."""
        min_dist = np.inf
        closest_obstacle = None
        for i, obstacle in enumerate(self.obstacles):
            s_obb, _,_,_= transformOrig2Proj(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.TRACK_FILE)
            
            # TODO: Modify to optimize computation time
            dist = self.dist(self.x0, obstacle)
            # print(f"Distance to obstacle {i}: {dist}")
            
            if s_obb < self.s0 - 0.25: # ignore if behind the car
                continue
            elif s_obb > self.s0 + 9.0: # 9.0 m  is the length of the track
                continue
            
            if dist < min_dist:
                min_dist = dist
                closest_obstacle = obstacle
        if min_dist == np.inf:
            dist = self.dist(self.x0, obstacle)
            print(f"Distance to obstacle {i}: {dist}")
            return obstacle
        return closest_obstacle

    def run(self):
        """Run the simulation."""
        print(f"Initial state: {self.x0}")
        print(f"Initial pose: {self.constraint.pose(self.x0)}")
        print(f"Initial obstacle 1 pose: {self.constraint.obb_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial obstacle 2 pose: {self.constraint.obb1_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial obstacle 3 pose: {self.constraint.obb2_pose(np.array(self.obstacles).reshape((self.n_params)))}")
        print(f"Initial distance: {self.constraint.dist(self.xN[0], np.array(self.obstacles).reshape((self.n_params)))}")

        alpha, beta, theta = compute_ellipse_parameters(1e-2, 0.1,  0.00, 0.95)
        print(f"Ellipse parameters: {alpha}, {beta}, {theta}")
        # ellipse = Ellipse((0, 0), 2*alpha, 2*beta, angle = theta *180 / np.pi, edgecolor='r', facecolor='none')
# 
        return

        for i in tqdm.tqdm(range(self.Nsim)):
            t = time.time()
            sref = self.s0 + self.REFERENCE_PROGRESS

            # TODO : Check for closest obstacle and update the obstacle position
            self.closest_obstacle_pose = self.closest_obstacle()

            dist_obstacle_N = np.array([self.dist(self.xN[k], self.evolution_function(self.closest_obstacle_pose, k))
                                         for k in range(self.NUM_DISCRETIZATION_STEPS)])
            if dist_obstacle_N[0] < self.min_dist:
                self.min_dist = dist_obstacle_N[0]
            if dist_obstacle_N[0] < 0.0:
                print("Collision detected")
                break

            for j in range(self.NUM_DISCRETIZATION_STEPS):
                if dist_obstacle_N[j] < self.DIST_THRESHOLD:
                    Q = self.Q_OBB
                    Qe = self.QE_OBB
                    Zl = self.Zl_OBB
                else:
                    Q = self.Q_SAFE
                    Qe = self.QE_SAFE
                    Zl = self.Zl_SAFE
                
                # Update the obstacle positions for each obstacle at each time step
                obb_j = self.update_obstacle_positions(j)
                yref = np.array([self.s0 + (sref - self.s0) * j / self.NUM_DISCRETIZATION_STEPS, 0, 0, self.REFERENCE_VELOCITY, 0, 0, 0, 0])
                self.predSim_obb[:, i, j, :] = obb_j
                self.acados_solver.set(j, "p", np.array(np.array(obb_j).reshape((self.n_params))))

                # Update the reference state and cost weights
                self.acados_solver.set(j, "yref", yref)
                self.acados_solver.cost_set(j, 'W', np.diag(Q))

            # Update the obstacle positions for each obstacle for the last time step
            obb_N = self.update_obstacle_positions(self.NUM_DISCRETIZATION_STEPS)
            self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "p", np.array(obb_N).reshape((self.n_params)))

            # Update the reference state and cost weights for the last time step
            yref_N = np.array([sref, 0, 0, self.REFERENCE_VELOCITY, 0, 0])
            self.acados_solver.set(self.NUM_DISCRETIZATION_STEPS, "yref", yref_N)
            self.acados_solver.cost_set(self.NUM_DISCRETIZATION_STEPS, 'W', np.diag(Qe))

            
            status = self.acados_solver.solve()
            elapsed = time.time() - t

            for j in range(self.NUM_DISCRETIZATION_STEPS):
                X = self.acados_solver.get(j, "x")
                self.xN[j] = X
                self.predSimX[i, j, :] = X

            self.tcomp_sum += elapsed
            if elapsed > self.tcomp_max:
                self.tcomp_max = elapsed

            self.x0 = self.acados_solver.get(0, "x")
            u0 = self.acados_solver.get(0, "u")
            for j in range(self.nx):
                self.simX[i, j] = self.x0[j]
            for j in range(self.nu):
                self.simU[i, j] = u0[j]

            

            self.obstacles = self.update_obstacle_positions(i)
            self.sim_obb[:, i, :] = np.array(self.obstacles)
            self.x0 = self.acados_solver.get(1, "x")
            self.acados_solver.set(0, "lbx", self.x0)
            self.acados_solver.set(0, "ubx", self.x0)
            self.s0 = self.x0[0]


            if self.x0[0] > self.Sgoal + 0.1:
                print("GOAL REACHED")

                    
                break

        for j in range(i, self.Nsim):
            self.simX[j, :] = self.simX[i, :]
            self.sim_obb[:, j, :] = self.sim_obb[:, i, :]
    
    def plot_results(self):

        print(f"Average computation time: {self.tcomp_sum / self.Nsim}")
        print(f"Maximum computation time: {self.tcomp_max}")
        print(f"Average speed: {np.average(self.simX[:, 3])} m/s")
        print(f"Minimum distance to obstacle: {self.min_dist}")

        t = np.linspace(0.0, self.Nsim * self.PREDICTION_HORIZON / self.NUM_DISCRETIZATION_STEPS, self.Nsim)

        plotTrackProjfinal(self.simX, self.sim_obb, # simulated trajectories
                            self.predSimX, self.predSim_obb, # predicted trajectories
                            self.TRACK_FILE, )#self.SAVE_FIG_NAME
        
        plotDist(self.simX, self.sim_obb, self.constraint, t)

        plotRes(self.simX, self.simU, t)

        plotTrackProj(self.simX,self.sim_obb, # simulated trajectories
                      self.predSimX, self.predSim_obb, # predicted trajectories
                        self.TRACK_FILE,) #self.SAVE_GIF_NAME


        if os.environ.get("ACADOS_ON_CI") is None:
            plt.show()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
    sim.plot_results()
