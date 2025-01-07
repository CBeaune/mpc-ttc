#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

import time, os
import numpy as np
from acados_settings_dev import *
from plotFcn import *
from tracks.readDataFcn import getTrack
import matplotlib.pyplot as plt
import tqdm

from time2spatial import transformProj2Orig,transformOrig2Proj

"""
Example of the frc_racecars in simulation without obstacle avoidance:
This example is for the optimal racing of the frc race cars. The model is a simple bicycle model and the lateral acceleration is constraint in order to validate the model assumptions.
The simulation starts at s=-2m until one round is completed(s=8.71m). The beginning is cut in the final plots to simulate a 'warm start'. 
"""



def dist(X, X_obb):
    assert len(X)==6, "X has to be 6 dimensional s,n,alpha,v,D,delta,psi,psidot, X is {}".format(len(X))
    assert len(X_obb)==6 , "X_obb has to be 6 dimensional x,y,psi,v,length,width"
    x,y,psi,v = transformProj2Orig(X[0], X[1], X[2], X[3])
    X_c = np.array([x,y,psi,v], dtype=object)
    return np.sqrt((X_c[0] - X_obb[0])**2 + (X_c[1] - X_obb[1])**2)

def evolution_function(X_obb0, i, cov_noise = np.zeros((2,2))):
    # X_obb0 = [x,y,psi,v,length,width] at t=0 in original coordinates
    noise = np.random.multivariate_normal([0,0], cov_noise)
    x = X_obb0[0] + i*dt*X_obb0[3]*np.cos(X_obb0[2]) + noise[0]
    y = X_obb0[1] + i*dt*X_obb0[3]*np.sin(X_obb0[2]) + noise[1]
    psi = X_obb0[2]
    v = X_obb0[3]
    return [x,y,psi,v,X_obb0[4],X_obb0[5]]


track = "LMS_Track6.txt"
[Sref, _, _, _, _] = getTrack(track)

Tf = 5.0 # prediction horizon
dt = 0.1
N = int(Tf/0.1)  # number of discretization steps
T = 20.00  # maximum simulation time[s]
vref = 0.25 # reference velocity
sref_N = vref*Tf# reference for final reference progress


# initialize static obstacle
obb_width = 0.15
obb_length = 0.25
obbC = np.array([0.5, -0.05, 0.0, 0.00]) # x,y,psi,v
obb_p = np.array([obbC[0], obbC[1], obbC[2], obbC[3], obb_length, obb_width])

# load model
constraint, model, acados_solver = acados_settings(Tf, N, track, obb_p)

# dimensions
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu
Nsim = int(T * N / Tf)

# initialize data structs
simX = np.zeros((Nsim, nx))
predSimX = np.zeros((Nsim, N, nx)) # prediction of the state for each stage for N stages
simU = np.zeros((Nsim, nu))
sim_obb = np.zeros((Nsim, 6))
predSim_obb = np.zeros((Nsim, N, 6))
xN = np.zeros((N, nx))
s0 = model.x0[0]
for i in range(N):
    xN[i] = model.x0
x0 = model.x0
print("initial state: {}".format(x0))
print("initial pose: {}".format(transformProj2Orig(x0[0], x0[1], x0[2], x0[3], track)))
print("initial obstacle pose: {}".format(evolution_function(obb_p,0)))
print('initial distance : {}'.format(dist(x0, evolution_function(obb_p,0))))
tcomp_sum = 0
tcomp_max = 0
thres =  5.0 / np.abs(vref + obb_p[3])

min_dist = np.inf
cov_noise = np.diag([0, 0])



# simulate
for i in tqdm.tqdm(range(Nsim)):

    danger_zone = False
    # update reference
    sref = s0 + sref_N

    dist_obstacle_N = np.array([dist(xN[k], evolution_function(obb_p,k)) for k in range(N)])
    thres = 0.75
    # print("min pred distance to obstacle at stage N: {}".format(np.min(dist_obstacle_N)))
    if dist_obstacle_N[0] < min_dist:
        min_dist = dist_obstacle_N[0]
    plt.plot(i,dist_obstacle_N[0],'ro')
    plt.plot([0,Nsim],[constraint.dist_min, constraint.dist_min],'k--')
    if dist_obstacle_N[0] < 0.12:
        print("collision")
        break

    
        



    for j in range(N):
        if dist_obstacle_N[j] < thres:
            # print("entering danger area")
            Q = [1e3, 5e-8, 1e-8, 1e-8, 1e-3, 5e-3, 5e-3, 5e2]
            Qe = [ 5e3, 1e3, 1e-8, 1e-8, 5e-3, 2e-3]
            Zl = 0.1 * np.ones((5,))
            Zl[4] = 100
            danger_zone = True
        else:
            Q = [1e5, 5e3, 1e-3, 1e-8, 1e-1, 5e-3, 5e-3, 5e3]
            Qe = [ 5e3, 1e3, 1e-3, 1e-8, 5e-3, 2e-3]
            Zl = 0.01 * np.ones((5,))

        if j == 0:
            obb_p = evolution_function(obb_p,0)
            obb_0_pred = evolution_function(obb_p,0, cov_noise)
            obb_j = obb_0_pred
        else:
            obb_j = evolution_function(obb_0_pred,j)
        yref = np.array([s0 + (sref - s0) * j / N, 0, 0, vref, 0, 0, 0, 0])
        predSim_obb[i, j, :] = obb_j
        acados_solver.set(j, "p", np.array(obb_j))
        acados_solver.set(j, "yref",  yref)
        # Q_obst_e = [1e-1, 1e-1, 1e-8, 1e-8, 1e-3, 5e-3]
        # Qe = [ 5e0, 1e3, 1e-8, 1e-8, 5e-3, 2e-3]
        # Q = [1e-1, 1e-1, 1e-3, 1e-8, 1e-3, 5e-3, 5e-1, 5e-1]
        # Q_obst = [1e-1, 5e3, 1e-7, 1e-8, 1e-3, 5e-3, 1e-3, 5e-3]
        acados_solver.cost_set(j, 'W', np.diag(Q))
        obb_N = evolution_function(obb_0_pred, N)
        acados_solver.set(N, "p", np.array(obb_N))
    yref_N = np.array([sref, 0, 0, 0, 0, 0])
    acados_solver.set(N, "yref", yref_N) 
    acados_solver.cost_set(N, 'W', np.diag(Qe))

    # solve ocp
    t = time.time()

    status = acados_solver.solve()
    # if status != 0:
    #     print("acados returned status {} in closed loop iteration {}.".format(status, i))

    elapsed = time.time() - t

    for j in range(N):
        X = acados_solver.get(j, "x")
        # [x,y,_,_] = transformProj2Orig(X[0],X[1],X[2],X[3],track)
        # plt.plot(x,y,'ro')
        # X_obb = acados_solver.get(j, "p")
        # plt.plot(X_obb[0],X_obb[1],'bo')
        xN[j] = X
        predSimX[i, j, :] = X
    # plt.pause(0.1)

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x0 = acados_solver.get(0, "x")
    u0 = acados_solver.get(0, "u")
    for j in range(nx):
        simX[i, j] = x0[j]
    for j in range(nu):
        simU[i, j] = u0[j]
    

    # update obstacle position 
    # obb_p = evolution_function(obb_p, 1)
    sim_obb[i, :] = np.array(obb_p)

    # update initial condition
    x0 = acados_solver.get(1, "x")
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)
    s0 = x0[0]



    # check if one lap is done and break and remove entries beyond
    if x0[0] > Sref[-1] + 0.1:
        # find where vehicle first crosses start line
        N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
        Nsim = i - N0  # correct to final number of simulation steps for plotting
        simX = simX[N0:i, :]
        simU = simU[N0:i, :]
        break

# Plot Results
t = np.linspace(0.0, Nsim * Tf / N, Nsim)

plotTrackProj(np.array([simX, sim_obb]), np.array([predSimX, predSim_obb]), track, save_name = 'noise_track3')
plotTrackProjfinal(np.array([simX, sim_obb]), np.array([predSimX, predSim_obb]), track)
plotalat(simX, simU, constraint, t)
plotRes(simX, simU, t)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / Nsim))
print("Maximum computation time: {}".format(tcomp_max))
print("Average speed:{}m/s".format(np.average(simX[:, 3])))
print("Lap time: {}s".format(Tf * Nsim / N))
print("Minimum distance to obstacle: {}".format(min_dist))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
