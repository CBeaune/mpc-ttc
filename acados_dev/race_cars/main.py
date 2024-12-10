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

"""
Example of the frc_racecars in simulation without obstacle avoidance:
This example is for the optimal racing of the frc race cars. The model is a simple bicycle model and the lateral acceleration is constraint in order to validate the model assumptions.
The simulation starts at s=-2m until one round is completed(s=8.71m). The beginning is cut in the final plots to simulate a 'warm start'. 
"""

track = "LMS_Track.txt"
[Sref, _, _, _, _] = getTrack(track)

# random seed
np.random.seed(0)

Tf = 7.0  # prediction horizon
N = 50  # number of discretization steps
T = 10.0  # maximum simulation time[s]
dt = 0.1
Nsim = int(T / dt)
vref = 0.5
sref_N = vref*Tf  # reference for final reference progress
distance=0.15

# RTI
t_preparation = np.zeros((Nsim))
t_feedback = np.zeros((Nsim))


# initialize obstacles trajectory
obbX = np.tile(np.array([np.random.normal(0.5, 0.02), 0.0, -np.pi, 0.1, 0, 0]), (N , 1))#distance*2

# load model
constraint, model, acados_solver = acados_settings(Tf, N, track, obbX[0, :])

# dimensions
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu


# initialize data structs
simX = np.zeros((Nsim, nx))
simobbX = np.zeros((Nsim, nx))
simU = np.zeros((Nsim, nu))
s0 = model.x0[0]
tcomp_sum = 0
tcomp_max = 0
x0 = model.x0
u0 = np.zeros((nu, 1))

num_iter_initial = 5
for _ in range(num_iter_initial):
    acados_solver.solve_for_x0(x0_bar = x0, fail_on_nonzero_status=False)



# simulate

for i in tqdm.tqdm(range(Nsim)):
    
    for j in range(N-3):
    # initial guess 
        acados_solver.set(j, "x", x0)
        acados_solver.set(j, "u", u0)
    acados_solver.set(N-3, "x", x0)

    # update reference
    sref = s0 + sref_N
    for j in range(N):
        yref = np.array([s0 + (sref - s0) * j / N, 0, 0, vref, 0, 0, 0, 0])

        # yref=np.array([1,0,0,1,0,0,0,0])
        acados_solver.set(j, "yref", yref)

        # update obstacle prediction
        obbX[j, 0] = obbX[0, 0] + obbX[0, 3]* np.cos(obbX[0, 2]) * j*dt 
        obbX[j, 3] = obbX[0, 3]
        acados_solver.set(j, "p", obbX[j, :])
    obbX_N = obbX[j]
    obbX_N[0] = obbX[0, 0] + obbX[0, 3]* np.cos(obbX[0, 2]) * N*dt
    obbX_N[3] = obbX[0, 3]
    acados_solver.set(N, "p", obbX_N)

    yref_N = np.array([sref, 0, 0, 0, 0, 0])
    # yref_N=np.array([0,0,0,0,0,0])
    acados_solver.set(N, "yref", yref_N)
    

    # solve ocp
    t = time.time()

    max_retries = 3
    retries = 0
    while retries < max_retries:
        status = acados_solver.solve()
        if (status == 0) or (status == 2):
            break
        else:
            # print(f"Retry {retries + 1}/{max_retries} failed with status {status}. Resetting solver...")
            # Reset states or inputs, e.g., use the previous feasible solution
            # acados_solver.set(0, "x", x0)
            retries += 1
    # if (status != 0) or (status != 2):
    #     print("acados returned status {} in closed loop iteration {}.".format(status, i))
    # acados_solver.print_statistics()
    
    elapsed = time.time() - t

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x0 = acados_solver.get(1, "x")
    u0 = acados_solver.get(1, "u")
    obbX = np.tile(obbX[1,:], (N , 1))
    # obbX[:, :2] += np.random.normal(0, 0.01, (N, 2))
    for j in range(nx):
        simX[i, j] = x0[j]
        simobbX[i, j] = obbX[0, j]
    simobbX[i,:2] += np.random.normal(0, 0.01, 2)
    for j in range(nu):
        simU[i, j] = u0[j]

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
    # if i > 0:
    #     ax1.cla()
    #     ax2.cla()
    #     t = np.arange(0, Nsim) * dt
    #     print(simX[:i,:].shape)
    #     plotTrackProj(np.dstack([simX[i,:], simobbX[i,:]]), track, ax1)
    #     plotdist(simX[:i,:], simobbX[:i,:], constraint, t[:i], ax2)
    #     plt.pause(0.2)

# Plot Results
t = np.linspace(0.0, Nsim * Tf / N, Nsim)
plotRes(simX, simU, t)
with sns.axes_style("whitegrid"):
    fig,(ax1,ax2) = plt.subplots(1,2)
    plotSimu(simX,t,simobbX, constraint,track, (ax1,ax2))

    plotTrackProj(np.dstack([simX, simobbX]), track)
    # fig.suptitle('simulated racing car')
    # plotalat(simX, simU, constraint, t)



# Print some stats
print("Average computation time: {}".format(tcomp_sum / Nsim))
print("Maximum computation time: {}".format(tcomp_max))
print("Average speed:{}m/s".format(np.average(simX[:, 3])))
print("Lap time: {}s".format(Tf * Nsim / N))
# avoid plotting when running on Travis
if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
