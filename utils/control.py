import numpy as np

# PID parameters for velocity and steering control
Kp_v, Ki_v, Kd_v = 0.4, 0.5, 0.05  # Velocity PID gains
Kp_theta, Ki_theta, Kd_theta = 2.0, 0.1, 0.1  # Steering angle PID gains

e_v_prev = 0
e_theta_prev = 0
integral_v = 0
integral_theta = 0

def pid_control(current_pos, current_heading, v_current, waypoint):
    # Extract current robot position
    x_robot, y_robot = current_pos
    x_waypoint, y_waypoint = waypoint[:2]

    # Compute desired heading towards the next waypoint
    theta_target = waypoint[2]
    e_theta = theta_target - current_heading

    # Compute heading control signal (steering)
    global integral_theta, e_theta_prev
    integral_theta += e_theta
    derivative_theta = e_theta - e_theta_prev
    u_theta = Kp_theta * e_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta
    # clamp control signals
    u_theta = np.clip(u_theta, -0.1, 0.1)
    e_theta_prev = e_theta
    
    # Compute velocity error and control signal (throttle)
    global integral_v, e_v_prev
    v_target = 0.2  # Constant velocity
    e_v = v_target - v_current
    integral_v += e_v
    derivative_v = e_v - e_v_prev
    u_v = Kp_v * e_v + Ki_v * integral_v + Kd_v * derivative_v
    # clamp control signals
    u_v = np.clip(u_v, -0.2, 0.2)
    e_v_prev = e_v

    # Return control signals (adjust throttle and steering)
    return u_v, u_theta

def simulate(current_pose, current_heading, u_v, u_theta):
    # Simulate robot motion with simple kinematics
    x, y = current_pose
    theta = current_heading

    # Update robot state with control signals
    x += u_v * np.cos(theta)
    y += u_v * np.sin(theta)
    theta += u_theta

    return (x, y), theta

def closest_waypoint(current_pos, waypoints):
    # Find the closest waypoint to the current robot position
    x, y = current_pos
    distances = np.linalg.norm(waypoints[:, :2] - np.array([x, y]), axis=1)
    closest_idx = np.argmin(distances)
    if np.hypot(x - waypoints[closest_idx, 0], y - waypoints[closest_idx, 1]) < 0.5:
        closest_idx += 1
    if closest_idx >= len(waypoints):
        return waypoints[-1]
    waypoints = np.delete(waypoints, np.s_[:closest_idx], axis=0)
    return waypoints


