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

from tracks.readDataFcn import getTrack
from time2spatial import transformProj2Orig,transformOrig2Proj
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import matplotlib.pyplot as plt
import numpy as np
import tqdm

def initplot(filename='LMS_Track.txt'):
        #Setup plot
    plt.ylim(bottom=-3.0,top=0.5)
    plt.xlim(left=-1.75,right=1.75)
    plt.ylabel('y[m]')
    plt.xlabel('x[m]')

    # Plot center line
    [Sref,Xref,Yref,Psiref,_]=getTrack(filename)
    plt.plot(Xref,Yref,'-',color='k',linewidth=.5)

    # Draw Trackboundaries
    track_width = 0.3
    Xboundleft=Xref-track_width/2*np.sin(Psiref)
    Yboundleft=Yref+track_width/2*np.cos(Psiref)
    Xboundright=Xref+track_width/2*np.sin(Psiref)
    Yboundright=Yref-track_width/2*np.cos(Psiref)
    plt.plot(Xboundleft,Yboundleft,'--',color='k',linewidth=2)
    plt.plot(Xboundright,Yboundright,color='k',linewidth=2)

    # Draw opposite lane
    Xboundleft=Xref-track_width*np.sin(Psiref)
    Yboundleft=Yref+track_width*np.cos(Psiref)
    Xboundright=Xref-3*track_width/2*np.sin(Psiref)
    Yboundright=Yref+3*track_width/2*np.cos(Psiref)
    plt.plot(Xboundleft,Yboundleft,color='k',linewidth=0.5)
    plt.plot(Xboundright,Yboundright,color='k',linewidth=2)

def plotTrackProj(traj, pred=[], filename='LMS_Track.txt', T_opt=None, save_name='track_animation'):
    # load track
    simX = traj[0]
    if pred != []:
        predX = pred[0]
        print(predX.shape)
    if traj.shape[0] > 1:
        simobbX = traj[1]
    if pred.shape[0] > 1:
        predobbX = pred[1]
        print(predobbX.shape)
    s = simX[:,0]
    n = simX[:,1]
    alpha = simX[:,2]
    v = simX[:,3]
    # transform data
    [x, y, alpha, _] = transformProj2Orig(s, n, alpha, v, filename)

    # Draw obstacle
    if traj.shape[0] > 1:
        xobb = simobbX[:,0]
        yobb = simobbX[:,1]
        psi_obb = simobbX[:,2]
        v = simobbX[:,3]

    # plot racetrack map
    fig, ax = plt.subplots(figsize=(10, 10))
    initplot(filename)

    heatmap = ax.scatter(x, y, c=v, cmap=cm.YlOrRd, edgecolor='none', marker='o', s=10)
    heatmap.set_clim(0, 0.25)
    color_pred = cm.plasma(np.linspace(0, 1, predX.shape[1]))
    ax.set_aspect('equal', 'box')

    # Prepare objects for updating instead of re-creating
    line, = ax.plot([], [], '-b')
    pred_dots = [ax.plot([], [], 'o', color='r', markersize=3)[0] for _ in range(predX.shape[1])]
    pred_circles = [plt.Circle((x[0], y[0]), 0.145602198, color=color_pred[j], alpha=0.2, fill=False) for j in range(predX.shape[1])]
    pred_rects = [plt.Rectangle((x[0]- simobbX[0, 4]/2,y[0]- simobbX[0, 5]/2), simobbX[0, 4], simobbX[0, 5], angle = alpha[0]*180/np.pi,  fill=False, rotation_point = 'center', color=color_pred[j])
                   for j in range(predX.shape[1])]

    pred_obb_circles = [plt.Circle((xobb[0], yobb[0]), 0.145602198, color=color_pred[j], alpha=0.2, fill=False) for j in range(predX.shape[1])]
    pred_obb_rects = [plt.Rectangle((xobb[0]- simobbX[0, 4]/2, yobb[0]- simobbX[0, 5]/2), simobbX[0, 4], simobbX[0, 5], angle = psi_obb[0]*180/np.pi , fill=False, color=color_pred[j], rotation_point = 'center')
                       for j in range(predX.shape[1])]

    for circle, rect in zip(pred_circles, pred_rects):
        ax.add_patch(circle)
        ax.add_patch(rect)

    for circle, rect in zip(pred_obb_circles, pred_obb_rects):
        ax.add_patch(circle)
        ax.add_patch(rect)

    def update(i):
        line.set_data(x[:i], y[:i])
        for j in range(predX.shape[1]):
            [x_pred, y_pred, alpha_pred, _] = transformProj2Orig(predX[i, j, 0], predX[i, j, 1], predX[i, j, 2], predX[i, j, 3], filename)
            pred_dots[j].set_data(x_pred, y_pred)
            pred_circles[j].center = (x_pred, y_pred)
            pred_rects[j].set_xy((x_pred - simobbX[0, 4]/2, y_pred - simobbX[0, 5]/2))
            pred_rects[j].angle = alpha_pred * 180 / np.pi

            if pred.shape[0] > 1:
                pred_obb_circles[j].center = (predobbX[i, j,0], predobbX[i, j,1])
                pred_obb_rects[j].set_xy((predobbX[i, j,0] - simobbX[i, 4]/2, predobbX[i, j,1] - simobbX[i, 5]/2))
                pred_obb_rects[j].angle = predobbX[i,j,2] * 180 / np.pi

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(x),5), interval=10, repeat=False)
    writer = PillowWriter(fps=5)
    plt.show()
    ani.save(f"/home/user/Documents/05_Contributions/Images/{save_name}.gif", writer=writer)
    

def plotTrackProjfinal(traj,pred=[], filename='LMS_Track.txt', T_opt=None):
    # load track
    simX = traj[0]
    if pred != []:
        predX = pred[0]
        print(predX.shape)
    if traj.shape[0] > 1:
        simobbX = traj[1]
    if pred.shape[0] > 1:
        predobbX = pred[1]
        print(predobbX.shape)
    s=simX[:,0]
    n=simX[:,1]
    alpha=simX[:,2]
    v=simX[:,3]
    # transform data
    [x, y, alpha, _] = transformProj2Orig(s, n, alpha, v,filename)
    
    # Draw obstacle
    if traj.shape[0] > 1:
        xobb=simobbX[:,0]
        yobb=simobbX[:,1]
        psi_obb=simobbX[:,2]
        v=simobbX[:,3]
    
    
    
    # plot racetrack map
    plt.figure(figsize=(10,10))
    initplot(filename)

    # Draw driven trajectory
    heatmap = plt.scatter(x,y, c=v, cmap=cm.YlOrRd, edgecolor='none', marker='o')
    heatmap.set_clim(0, 0.25)
    # cbar = plt.colorbar(heatmap, fraction=0.035)
    # cbar.set_label("velocity in [m/s]")

    color = cm.plasma(np.linspace(0, 1, len(x)))
      

    
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    
    for i in tqdm.tqdm(range(len(x))):
        plt.plot(x[:i],y[:i], '-b')
        
        if i %10:
            circles = plt.Circle((x[i], y[i]), 0.145602198, color=color[i], alpha = 0.2, fill=False)
            # for j in range(predX.shape[1]):
            #     if j % 2 == 0:
            #         [x_pred, y_pred, _, _] = transformProj2Orig(predX[i,j,0], predX[i,j,1], predX[i,j,2], predX[i,j,3], filename)
            #         plt.plot(x_pred, y_pred, 'o', color='r')
            #         if pred.shape[0] > 1:
            #             plt.plot(predobbX[i,j,0], predobbX[i,j,1], 'o', color='g')
            plt.gca().add_patch(circles)
            
            rectangles = plt.Rectangle((x[i]-simobbX[0,4]/2, y[i]-simobbX[0,5]/2), simobbX[0,4], simobbX[0,5],
                                            angle=alpha[i]*180/np.pi, color=color[i],  rotation_point = 'center',
                                            fill=False)
            plt.gca().add_patch(rectangles)

            # Draw obstacle

            rectangles = plt.Rectangle((xobb[i]-simobbX[i,4]/2, yobb[i]-simobbX[i,5]/2), simobbX[i,4], simobbX[i,5],
                                        angle=psi_obb[i]*180/np.pi, color=color[i], rotation_point = 'center',
                                            fill=False)

            circles = plt.Circle((xobb[i], yobb[i]), 0.145602198, color=color[i], alpha = 0.2, fill=False)
            plt.gca().add_patch(circles)
            plt.gca().add_patch(rectangles)

  

    


def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t, simU[:,0], color='r')
    plt.step(t, simU[:,1], color='g')
    plt.title('closed-loop simulation')
    plt.legend(['dD','ddelta'])
    plt.ylabel('u')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:,:])
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend(['s','n','alpha','v','D','delta'])
    plt.grid(True)

def plotalat(simX,simU,constraint,t):
    Nsim=t.shape[0]
    plt.figure()
    alat=np.zeros(Nsim)
    for i in range(Nsim):
        alat[i]=constraint.alat(simX[i,:],simU[i,:])
    plt.plot(t,alat)
    plt.plot([t[0],t[-1]],[constraint.alat_min, constraint.alat_min],'k--')
    plt.plot([t[0],t[-1]],[constraint.alat_max, constraint.alat_max],'k--')
    plt.legend(['alat','alat_min/max'])
    plt.xlabel('t')
    plt.ylabel('alat[m/s^2]')
