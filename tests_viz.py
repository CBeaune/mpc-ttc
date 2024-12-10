import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Parameters for simulation
dt = 0.5  # Time step
time_steps = 4  # Number of steps
time = np.linspace(0, dt * time_steps, time_steps)

# Bicycle model parameters
L = 2.0  # Wheelbase
v = 2.0  # Constant velocity
steering_angle = 0.0 # Constant steering angle (radians)

v_obs = 1.0 #Obstacle velocity
steering_angle_obs = 0.0 #Obstacle steering angle (radians)

# Initial state: [x, y, theta, v]
state = np.array([0.0, 0.0, 0.0, v])

# Obstacle state: [x, y, theta, v]
state_obs = np.array([5,6, -np.pi/2, v_obs])




# Function to compute the bicycle model state transition
def bicycle_model(state, dt):
    x, y, theta, v = state
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + (v / L) * np.tan(steering_angle) * dt
    return np.array([x_new, y_new, theta_new, v])

# Kalman Filter prediction step
def kalman_predict(state, P, Q, dt):
    global F
    # Update F matrix for the linearized model
    _, _, theta, v = state
    F[0, 2] = -v * np.sin(theta) * dt  # df/dtheta
    F[0, 3] = np.cos(theta) * dt       # df/dv
    F[1, 2] = v * np.cos(theta) * dt   # df/dtheta
    F[1, 3] = np.sin(theta) * dt       # df/dv
    F[2, 3] = (1 / L) * np.tan(steering_angle) * dt  # df/dv

    # Predict state
    predicted_state = bicycle_model(state, dt)
    
    # Predict covariance
    P = F @ P @ F.T + Q
    return predicted_state, P


# ================== Kalman Filter for vehicle ==================
# Kalman Filter Parameters
state_dim = 4
P = np.eye(state_dim) * 0.1  # Initial covariance
Q = np.diag([0.0001, 0.0001, 0.00005, 0.0001])*1e-45  # Process noise covariance
F = np.eye(state_dim)  # Placeholder for state transition matrix
# Simulation loop
positions = []  # Track positions for visualization
covariances = []  # Track covariance over time

for t in time:
    # Kalman predict step
    state, P = kalman_predict(state, P, Q, dt)
    
    # Store position and covariance
    positions.append(state[:2])  # Only x, y for visualization
    covariances.append(P[:2, :2])  # Store position covariance

# Convert to arrays for plotting
positions = np.array(positions)
covariances = np.array(covariances)

# Extract covariance for plotting
position_variance = np.sqrt(covariances[:, 0, 0] + covariances[:, 1, 1])  # Combined x, y uncertainty


# ================== Kalman Filter for obstacle ==================
# Kalman Filter Parameters
state_dim = 4
P = np.eye(state_dim) * 0.1  # Initial covariance
Q = np.diag([0.0001, 0.0001, 0.00005, 0.0001])*1e-45  # Process noise covariance
F = np.eye(state_dim)  # Placeholder for state transition matrix
# Simulation loop
positions_obs = []  # Track positions for visualization
covariances_obs = []  # Track covariance over time


for t in time:
    # Kalman predict step
    state_obs, P = kalman_predict(state_obs, P, Q, dt)
    
    # Store position and covariance
    positions_obs.append(state_obs[:2])  # Only x, y for visualization
    covariances_obs.append(P[:2, :2])  # Store position covariance

# Convert to arrays for plotting
positions_obs = np.array(positions_obs)
covariances_obs = np.array(covariances_obs)


# Plotting the results
fig = plt.figure(figsize=(12, 6))

# 3D plot of position probability
with sns.axes_style("whitegrid"):
    ax = fig.add_subplot()
    ax.set_xlim(-4, 8)
    ax.set_ylim(-4, 8)
    ax.axis('equal')

# Function to compute 2D Gaussian
def gaussian_2d(x, y, mean, cov):
    inv_cov = np.linalg.inv(cov)
    diff = np.stack([x - mean[0], y - mean[1]], axis=-1)
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=-1)
    normalizer = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return normalizer * np.exp(exponent)

# Choose colormap and normalize time
cmap = plt.cm.plasma  # Choose a colormap
from matplotlib import colors as mcolors
norm = mcolors.Normalize(vmin=min(time), vmax=max(time)+0.5)  # Normalize time values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # Create a scalar mappable



# Plot Gaussian distributions at each time step
for i, t in enumerate(time):
    mean = positions[i]
    cov = covariances[i]
    x_range = np.linspace(mean[0]-5, mean[0]+5, 50)
    y_range = np.linspace(mean[1]-5, mean[1]+5, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute Gaussian for the current time step
    Z = gaussian_2d(X, Y, mean, cov)
    
    # Overlay the contour for the current Gaussian, with color corresponding to time
    ax.contour(X, Y, Z, levels=10, colors=[cmap(norm(t))], alpha =0.7)  # Use colormap to set color based on time
    ax.scatter(mean[0], mean[1], color='r', label="Vehicle Position" if i == 0 else None)

    # Plot obstacle
    mean = positions_obs[i]
    cov = covariances_obs[i]
    x_range = np.linspace(mean[0]-5, mean[0]+5, 50)
    y_range = np.linspace(mean[1]-5, mean[1]+5, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute Gaussian for the current time step
    Z = gaussian_2d(X, Y, mean, cov)
    
    # Overlay the contour for the current Gaussian, with color corresponding to time
    ax.contour(X, Y, Z, levels=10, colors=[cmap(norm(t))], alpha =0.7)  # Use colormap to set color based on time
    ax.scatter(mean[0], mean[1], color='b', label="Obstacle Position" if i == 0 else None)
    

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Time (s)')
# ax.colorbar()
# surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Probability Distribution of Vehicles Position Over Time')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
# ax.set_zlabel('Y Position')
ax.legend()

# # 2D plot of uncertainty over time
# ax2 = fig.add_subplot(122)
# ax2.plot(time, position_variance, label="Position Variance", color="blue")
# ax2.set_title("Uncertainty Over Time")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Position Variance")
# ax2.legend()

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define points A and B
point_A = np.array([0, 0])  # Coordinates of B
point_B = np.array([1.5, 3])  # Coordinates of A

# Define velocity vectors for A and B
velocity_A = np.array([0.25, 0.25])  # Velocity of A
velocity_B = np.array([0.3, 0.3])  # Velocity of B

# Calculate relative velocity vector (A relative to B)
relative_velocity = velocity_B - velocity_A

# Calculate unit vector from B to A
distance_vector = point_B - point_A
distance_magnitude = np.linalg.norm(distance_vector)
unit_vector_R = distance_vector / distance_magnitude

print(np.dot(relative_velocity, unit_vector_R))

# Plot settings
with sns.axes_style("whitegrid"):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

# Plot points A and B
plt.scatter(*point_B, color='red', label='Point B ')
plt.scatter(*point_A, color='blue', label='Point A (Observer)')

# Plot line between A and B (distance d)
plt.plot([point_B[0], point_A[0]], [point_B[1], point_A[1]], 'k--', label='Distance $d$')

# Plot velocity vectors
plt.quiver(*point_A, *velocity_A, angles='xy', scale_units='xy', scale=0.5, color='blue', label='$\\vec{v}_A$')
plt.quiver(*point_B, *velocity_B, angles='xy', scale_units='xy', scale=0.5, color='red', label='$\\vec{v}_B$')

# Plot relative velocity vector
plt.quiver(*point_A, *relative_velocity, angles='xy', scale_units='xy', scale=0.5, color='green', label='$\\vec{v}_{AB}$ (relative)')

# Annotate unit vector R
plt.quiver(*point_A, *unit_vector_R, angles='xy', scale_units='xy', scale=distance_magnitude, color='gray', alpha=0.5, label='$\\hat{R}$')

# # Annotate angular size
# theta_start = np.arctan2(distance_vector[1], distance_vector[0])
# theta_end = theta_start + 0.1
# arc = np.linspace(theta_start, theta_end, 100)
# arc_x = 0.5 * np.cos(arc) + point_A[0]
# arc_y = 0.5 * np.sin(arc) + point_A[1]
# plt.plot(arc_x, arc_y, color='purple', label='$\\theta$ (angular size)')

# Add labels and legend
plt.text(point_B[0] - 0.2, point_B[1] , 'B', color='red', fontsize=12)
plt.text(point_A[0] + 0.2, point_A[1], 'A', color='blue', fontsize=12)
plt.legend(loc='upper left')
plt.title('No Looming' if np.dot(relative_velocity, unit_vector_R) > 0 else 'Looming')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)


plt.show()
