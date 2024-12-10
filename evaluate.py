import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from utils.visualization import plot_robot, plot_cov_ellipse



# === Parameters =====
w = 0.178
h = 0.132
dt = 0.01

# === Load the trajectories =====
import pickle
with open("data/robots_crossroads.pkl", "rb") as f:
    dict_robot = pickle.load(f)
X_robot0 = dict_robot[0]
X_robot1 = dict_robot[1]
X_robot2 = dict_robot[2]

# === Load the ttc values =====
with open("data/ttc_crossroads.pkl", "rb") as f:
    ttc_norm, ttc_cov_norm, ttc_samples_norm = pickle.load(f)

# load the min distances
with open("data/min_distances_crossroads.pkl", "rb") as f:
    min_distances = pickle.load(f)


# === Evaluate performance of TTC values on ground truth =====
# Compute the ground truth TTC values (if Xrobot0 .shape[0]<30.0s --simulation time --  and no collision at the end)
ttc_gt = X_robot0.shape[0]
if ttc_gt*dt == 30.0 and np.all(min_distances > 0.08):
    ttc_gt = np.inf

# For T_lookahead in [dt, 10s] compute number of true positives, false positives, true negatives, false negatives
T_lookaheads = np.arange(dt, 10.0, dt/10)
TP = np.zeros_like(T_lookaheads)
FP = np.zeros_like(T_lookaheads)
TN = np.zeros_like(T_lookaheads)
FN = np.zeros_like(T_lookaheads)

precision = np.zeros_like(T_lookaheads)
recall = np.zeros_like(T_lookaheads)
f1 = np.zeros_like(T_lookaheads)


for i, T_lookahead in enumerate(T_lookaheads):
    for t in range(ttc_norm.shape[1]):
        if t+T_lookahead >= ttc_norm.shape[1]:
            break
        # positive gt ttc when ttc_gt < t+ T_lookahead
        if ttc_gt <= t + T_lookahead :
            if 0 < ttc_norm[0, t] < T_lookahead :
                TP[i] += 1
            else:
                FN[i] += 1
        else:
            if 0 < ttc_norm[0, t] < T_lookahead:
                FP[i] += 1
            else:
                TN[i] += 1

    precision[i] = TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] > 0 else 0
    recall[i] = TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] > 0 else 0
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0

# Plot the results precision, recall, f1 as a function of T_lookahead

fig, ax = plt.subplots()
ax.plot(T_lookaheads, precision, label="Precision")
ax.plot(T_lookaheads, recall, label="Recall")
ax.plot(T_lookaheads, f1, label="F1")
ax.set_xlabel("Lookahead time (s)")
ax.set_ylabel("Performance")
ax.legend()


# === Plot the trajectories =====
# the lighter the color, the later in time
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots()
for i in range(X_robot0.shape[0]):

    plot_robot(X_robot0[i,0], X_robot0[i,1], X_robot0[i,2], w, h, ax, color =sns.color_palette()[0], alpha = (X_robot0.shape[0]-i)/X_robot0.shape[0]) 
    
    if i >= X_robot1.shape[0]:
        plot_robot(X_robot1[-1,0], X_robot1[-1,1], X_robot1[-1,2], w, h, ax, color =sns.color_palette()[1], alpha = (X_robot0.shape[0]-i)/X_robot0.shape[0])   
    else:
        plot_robot(X_robot1[i,0], X_robot1[i,1], X_robot1[i,2], w, h, ax, color =sns.color_palette()[1], alpha = (X_robot0.shape[0]-i)/X_robot0.shape[0])
    
    if i >= X_robot2.shape[0]:
        plot_robot(X_robot2[-1,0], X_robot2[-1,1], X_robot2[-1,2], w, h, ax, color =sns.color_palette()[2], alpha = (X_robot0.shape[0]-i)/X_robot0.shape[0])   
    else:
        plot_robot(X_robot2[i,0], X_robot2[i,1], X_robot2[i,2], w, h, ax, color =sns.color_palette()[2], alpha = (X_robot0.shape[0]-i)/X_robot0.shape[0])
ax.axis('equal')
plt.grid(True)

# === Plot the ttc values =====


with sns.axes_style("whitegrid"):
    fig, [ax1, ax2] = plt.subplots(2,1)
for i in tqdm.tqdm(range(X_robot0.shape[0]), desc="Plotting TTC"):
    for j in range(2):
        if j == 0:
            ax1.scatter(i,ttc_norm[j,i] , color= 'r', label="ttc")
            ax1.scatter(i,ttc_cov_norm[j,i] , color='b', label="ttc_cov")
            ax1.scatter(i,ttc_samples_norm[j,i] , color='g', label="ttc_samples")
            ax1.set_ylim(-1,10)
            ax1.plot([min_distances[j,0], min_distances[j,0]], [-1, 10], 'k--')

        if j == 1:
            ax2.scatter(i,ttc_norm[j,i] , color= 'r', label="ttc")
            ax2.scatter(i,ttc_cov_norm[j,i] , color='b', label="ttc_cov")
            ax2.scatter(i,ttc_samples_norm[j,i] , color='g', label="ttc_samples")
            ax2.set_ylim(-1,10)
            ax2.plot([min_distances[j,0], min_distances[j,0]], [0, 10], 'k--')

plt.grid(True)
plt.show()