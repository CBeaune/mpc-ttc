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

import tkinter as tk
from tkinter import filedialog
import os

from utils import compute_ellipse_parameters

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import tqdm

def initplot(filename='LMS_Track.txt'):
        #Setup plot
    with sns.axes_style("whitegrid"):
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

def plotTrackProj(simX, sim_obb, # simulated trajectories
                    predSimX, predSim_obb, # predicted trajectories
                    filename='LMS_Track.txt', save_path=None, save_name = None, fig = None, ax = None):
    
    # Load simulated data
    s=simX[:,0]
    n=simX[:,1]
    alpha=simX[:,2]
    v=simX[:,3]
    # transform data
    [x, y, alpha, _] = transformProj2Orig(s, n, alpha, v,filename)
    
    # Draw obstacle 
    # /!\ sim_obb shape is (N_obb, N_sim, 6)
    xobb=sim_obb[:, :,0] # /!\ shape is (N_obb, N_sim)
    yobb=sim_obb[:,:,1]
    psi_obb=sim_obb[:,:,2]
    v_obb=sim_obb[:,:,3]
    sigmax = sim_obb[:,:,6]
    sigmay = sim_obb[:,:,7]
    sigmaxy = sim_obb[:,:,8]

    # Load predicted data
    predX = predSimX
    predobbX = predSim_obb # /!\ predSim_obb shape is (N_OBSTACLES, Nsim, NUM_DISCRETIZATION_STEPS, 6)


    
    LENGTH = sim_obb[0,0,4]
    WIDTH = sim_obb[0,0,5]
    
    # plot racetrack map
    with sns.axes_style("whitegrid"):
        if fig == None:
            REAL_TIME_PLOTTING = False
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            REAL_TIME_PLOTTING = True
    initplot(filename)

    heatmap = ax.scatter(x, y, c=v, cmap=cm.YlOrRd, edgecolor='none', marker='o', s=10)
    heatmap.set_clim(0, 0.25)
    color_pred = sns.color_palette("flare", predX.shape[1])
    ax.set_aspect('equal', 'box')

    # Prepare objects for updating instead of re-creating
    line, = ax.plot([], [], '-b')
    pred_dots = [ax.plot([], [], 'o', color='r', markersize=3)[0] for _ in range(predX.shape[0])]

    # Covering circles
    r = 1/LENGTH * (WIDTH**2/4 + LENGTH**2/4)

    pred_circles1 = [plt.Circle((x[0] - r * np.cos(alpha[0]), y[0] - r * np.sin(alpha[0])),
                                 r, color=color_pred[j], alpha=0.3, fill=False) for j in range(predX.shape[1])]
    pred_circles2 = [plt.Circle((x[0] , y[0]),
                                 r, color=color_pred[j], alpha=0.3, fill=False) for j in range(predX.shape[1])]
    pred_circles3 = [plt.Circle((x[0] + r * np.cos(alpha[0]), y[0] + r * np.sin(alpha[0])),
                                 r, color=color_pred[j], alpha=0.3, fill=False) for j in range(predX.shape[1])]
    pred_rects = [plt.Rectangle((x[0]- LENGTH/2,y[0]- WIDTH/2), LENGTH, WIDTH, 
                                angle = alpha[0]*180/np.pi,  alpha= 0.3,  fill=False, rotation_point = 'center', color=color_pred[j])
                   for j in range(predX.shape[1])]

    
    pred_obb_circles = []
    pred_obb_rects = []
    pred_obb_ellipses = []
    for k in range(predobbX.shape[0]): # for each obstacle
        a,b,theta = compute_ellipse_parameters(sigmax[k,0], sigmay[k,0], sigmaxy[k,0], eta = 0.95)
        # print(a,b,theta)
        pred_obb_circles1 = [plt.Circle((xobb[k,0] - r * np.cos(psi_obb[k,0]), yobb[k,0] - r * np.sin(psi_obb[k,0])),
                                        r, color=color_pred[j], alpha=0.3, fill=False) for j in range(predX.shape[1])]
        pred_obb_ellipse1 = [Ellipse((xobb[k,0] - r * np.cos(psi_obb[k,0]), yobb[k,0] - r * np.sin(psi_obb[k,0])),
                                        2*a + 2*r , 2*b + 2*r, angle = theta*180/np.pi, alpha=0.3, fill=False, color=color_pred[j])
                        for j in range(predX.shape[1])]
        
        pred_obb_circles2 = [plt.Circle((xobb[k,0], yobb[k,0]),
                                        r, color=color_pred[j], alpha=0.3, fill=False) for j in range(predX.shape[1])]
        pred_obb_ellipse2 = [Ellipse((xobb[k,0], yobb[k,0]),
                                       2*a + 2*r , 2*b + 2*r, angle = theta*180/np.pi, alpha=0.3, fill=False, color=color_pred[j])
                        for j in range(predX.shape[1])]
        
        pred_obb_circles3 = [plt.Circle((xobb[k,0] + r * np.cos(psi_obb[k,0]), yobb[k,0] + r * np.sin(psi_obb[k,0])),
                                        r, color=color_pred[j], alpha=0.3, fill=False) for j in range(predX.shape[1])]
        pred_obb_ellipse3 = [Ellipse((xobb[k,0] + r * np.cos(psi_obb[k,0]), yobb[k,0] + r * np.sin(psi_obb[k,0])),
                                        2*a + 2*r , 2*b + 2*r, angle = theta*180/np.pi, alpha=0.3, fill=False, color=color_pred[j])
                        for j in range(predX.shape[1])]
        
        pred_obb_shape = [plt.Rectangle((xobb[k,0]- LENGTH/2, yobb[k,0]- WIDTH/2), LENGTH, WIDTH, angle = psi_obb[k,0]*180/np.pi ,
                                        fill=False, alpha= 0.3, color=color_pred[j], rotation_point = 'center')
                        for j in range(predX.shape[1])]

        pred_obb_circles.append([pred_obb_circles1, pred_obb_circles2, pred_obb_circles3])
        pred_obb_ellipses.append([pred_obb_ellipse1, pred_obb_ellipse2, pred_obb_ellipse3])
        pred_obb_rects.append(pred_obb_shape)

    for k in range(predobbX.shape[0]): # for each obstacle
        for circle1, circle2, circle3, rect in zip(pred_obb_circles[k][0],pred_obb_circles[k][1],pred_obb_circles[k][2], pred_obb_rects[k]):
            # ax.add_patch(circle1)
            # ax.add_patch(circle2)
            # ax.add_patch(circle3)
            ax.add_patch(rect)
        
        for ellipse1, ellipse2, ellipse3 in zip(pred_obb_ellipses[k][0],pred_obb_ellipses[k][1],pred_obb_ellipses[k][2]):
            ax.add_patch(ellipse1)
            ax.add_patch(ellipse2)
            ax.add_patch(ellipse3)
        
    for circle1, circle2, circle3, rect in zip(pred_circles1, pred_circles2, pred_circles3, pred_rects):
        with sns.axes_style("whitegrid"):
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.add_patch(circle3)
            ax.add_patch(rect)



    def update(i):
        line.set_data(x[:i], y[:i])


        for j in range(predX.shape[1]):
            [x_pred, y_pred, alpha_pred, _] = transformProj2Orig(predX[i, j, 0], predX[i, j, 1], predX[i, j, 2], predX[i, j, 3], filename)

            pred_dots[j].set_data(x_pred , y_pred)
            pred_circles1[j].center = (x_pred - r * np.cos(alpha_pred) , y_pred - r * np.sin(alpha_pred))
            pred_circles2[j].center = (x_pred, y_pred)
            pred_circles3[j].center = (x_pred + r * np.cos(alpha_pred), y_pred + r * np.sin(alpha_pred))
            pred_rects[j].set_xy((x_pred - LENGTH/2, y_pred - WIDTH/2))
            pred_rects[j].angle = alpha_pred * 180 / np.pi
            if j == 0:
                pred_circles1[j].set_alpha(0.5)
                pred_circles1[j].fill=True
                pred_circles2[j].set_alpha(0.5)
                pred_circles2[j].fill=True
                pred_circles3[j].set_alpha(0.5)
                pred_circles3[j].fill=True
                pred_rects[j].set_alpha(1.0)

            for k in range(predobbX.shape[0]): # for each obstacle
                [x_pred_obb, y_pred_obb, alpha_pred_obb] = [predobbX[k, i, j, 0], predobbX[k, i, j, 1], predobbX[k, i, j, 2]]
                # pred_obb_circles[k][0][j].center = (x_pred_obb - r * np.cos(alpha_pred_obb),
                #                                     y_pred_obb - r * np.sin(alpha_pred_obb))
                # pred_obb_circles[k][1][j].center = (x_pred_obb, y_pred_obb)
                # pred_obb_circles[k][2][j].center = (x_pred_obb + r * np.cos(alpha_pred_obb),
                #                                     y_pred_obb + r * np.sin(alpha_pred_obb))
                a,b,theta = compute_ellipse_parameters(predobbX[k, i, j, 6], predobbX[k, i, j, 7], predobbX[k, i, j, 8], eta = 0.95)

                pred_obb_ellipses[k][0][j].height = 2*b + 2*r
                pred_obb_ellipses[k][0][j].width = 2*a + 2*r
                pred_obb_ellipses[k][0][j].center = (x_pred_obb - r * np.cos(alpha_pred_obb),
                                                    y_pred_obb - r * np.sin(alpha_pred_obb))
                pred_obb_ellipses[k][0][j].angle = theta * 180 / np.pi

                pred_obb_ellipses[k][1][j].height = 2*b + 2*r
                pred_obb_ellipses[k][1][j].width = 2*a + 2*r
                pred_obb_ellipses[k][1][j].center = (x_pred_obb, y_pred_obb)
                pred_obb_ellipses[k][1][j].angle = theta * 180 / np.pi

                pred_obb_ellipses[k][2][j].height = 2*b + 2*r
                pred_obb_ellipses[k][2][j].width = 2*a + 2*r
                pred_obb_ellipses[k][2][j].center = (x_pred_obb + r * np.cos(alpha_pred_obb),
                                                    y_pred_obb + r * np.sin(alpha_pred_obb))
                pred_obb_ellipses[k][2][j].angle = theta * 180 / np.pi
                
                pred_obb_rects[k][j].set_xy((x_pred_obb - LENGTH/2, y_pred_obb - WIDTH/2))
                pred_obb_rects[k][j].angle = alpha_pred_obb * 180 / np.pi
                if j == 0:
                    # pred_obb_circles[k][0][j].set_alpha(0.5)
                    # pred_obb_circles[k][0][j].fill=True
                    # pred_obb_circles[k][1][j].set_alpha(0.5)
                    # pred_obb_circles[k][1][j].fill=True
                    # pred_obb_circles[k][2][j].set_alpha(0.5)
                    # pred_obb_circles[k][2][j].fill=True

                    pred_obb_ellipses[k][0][j].set_alpha(0.5)
                    pred_obb_ellipses[k][0][j].fill=True
                    pred_obb_ellipses[k][1][j].set_alpha(0.5)
                    pred_obb_ellipses[k][1][j].fill=True
                    pred_obb_ellipses[k][2][j].set_alpha(0.5)
                    pred_obb_ellipses[k][2][j].fill=True

                    pred_obb_rects[k][j].set_alpha(1.0)



    ani = animation.FuncAnimation(fig, update, frames=range(0, len(x),5), interval=10, repeat=False)
    writer = PillowWriter(fps=10)
    if not REAL_TIME_PLOTTING:
        plt.show()
    if save_name != None:
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        ani.save(save_path + f"{save_name}.gif", writer=writer)
    

def plotTrackProjfinal(simX, sim_obb, # simulated trajectories
                        predSimX, predSim_obb, # predicted trajectories
                        filename='LMS_Track.txt', save_path = None, save_name = None):
    # load track

    s=simX[:,0]
    n=simX[:,1]
    alpha=simX[:,2]
    v=simX[:,3]
    # transform data
    [x, y, alpha, _] = transformProj2Orig(s, n, alpha, v,filename)
    
    # Draw obstacle 
    # /!\ sim_obb shape is (N_obb, N_sim, 6)
    xobb=sim_obb[:, :,0] # /!\ shape is (N_obb, N_sim)
    yobb=sim_obb[:,:,1]
    psi_obb=sim_obb[:,:,2]
    v_obb=sim_obb[:,:,3]


    
    LENGTH = sim_obb[0,0,4]
    WIDTH = sim_obb[0,0,5]
    
    
    # plot racetrack map
    plt.figure(figsize=(10,10))

    initplot(filename)
    

    color = sns.color_palette("flare", x.shape[0])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    r = 1/LENGTH * (WIDTH**2/4 + LENGTH**2/4)
    for i in tqdm.tqdm(range(x.shape[0])):
        
        if i %10:
            
            
            circles1 = plt.Circle((x[i] - r * np.cos(alpha[i]), y[i] - r * np.sin(alpha[i])),
                                        r, color=color[i], alpha = 0.2, fill=False)
            circles2 = plt.Circle((x[i], y[i]), r, color=color[i], alpha = 0.2, fill=False)
            circles3 = plt.Circle((x[i] + r * np.cos(alpha[i]), y[i] + r * np.sin(alpha[i])),
                                    r, color=color[i], alpha = 0.2, fill=False)
            plt.gca().add_patch(circles1)
            plt.gca().add_patch(circles2)
            plt.gca().add_patch(circles3)
            
            rectangles = plt.Rectangle((x[i]-LENGTH/2, y[i]-WIDTH/2), LENGTH, WIDTH,
                                            angle=alpha[i]*180/np.pi, color=color[i],  rotation_point = 'center',
                                            fill=False)
            plt.gca().add_patch(rectangles)

            # Draw obstacles
            for j in range(xobb.shape[0]):
                
                rectangles = plt.Rectangle((xobb[j,i]-LENGTH/2, yobb[j,i]-WIDTH/2), LENGTH, WIDTH,
                                            angle=psi_obb[j,i]*180/np.pi, color=color[i], rotation_point = 'center',
                                                fill=False)

                circles1 = plt.Circle((xobb[j,i] - r * np.cos(psi_obb[j,i]), yobb[j,i] - r * np.sin(psi_obb[j,i])),
                                        r, color=color[i], alpha = 0.2, fill=False)
                circles2 = plt.Circle((xobb[j,i], yobb[j,i]), r, color=color[i], alpha = 0.2, fill=False)
                circles3 = plt.Circle((xobb[j,i] + r * np.cos(psi_obb[j,i]), yobb[j,i] + r * np.sin(psi_obb[j,i])),
                                        r, color=color[i], alpha = 0.2, fill=False)
                plt.gca().add_patch(circles1)
                plt.gca().add_patch(circles2)
                plt.gca().add_patch(circles3)
                plt.gca().add_patch(rectangles)

    # Draw driven trajectory
    # heatmap = plt.scatter(x,y, c=v, cmap=sns.color_palette("YlOrBr", as_cmap=True),s=10, edgecolor='none', marker='o')
    # cbar = plt.colorbar(heatmap, fraction=0.035)
    x_combined = x
    y_combined = y
    v_combined = v
    for k in range(xobb.shape[0]):
        x_combined = np.concatenate((x_combined, xobb[k,:]))
        y_combined = np.concatenate((y_combined, yobb[k,:]))
        v_combined = np.concatenate((v_combined, v_obb[k,:]))
    heatmap = plt.scatter(x_combined, y_combined, c=v_combined,
                           cmap=sns.color_palette("YlOrBr", as_cmap=True),s=10, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("velocity in [m/s]")
    if save_name != None:
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        plt.savefig(save_path + f"{save_name}.png")
        plt.savefig(save_path + f"{save_name}.pdf")
  

    


def plotRes(simX,simU,t):
    # plot results
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10,10), tight_layout=True)
        plt.subplot(2, 1, 1)
        plt.step(t, simU[:,0])
        plt.step(t, simU[:,1])
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
    plt.figure(figsize=(10,10), tight_layout=True)
    plt.title('Lateral acceleration')
    alat=np.zeros(Nsim)
    for i in range(Nsim):
        alat[i]=constraint.alat(simX[i,:],simU[i,:])
    plt.plot(t,alat)
    plt.plot([t[0],t[-1]],[constraint.alat_min, constraint.alat_min],'k--')
    plt.plot([t[0],t[-1]],[constraint.alat_max, constraint.alat_max],'k--')
    plt.legend(['alat','alat_min/max'])
    plt.xlabel('t')
    plt.ylabel('alat[m/s^2]')

def plotDist(simX,sim_obb,constraint,t, save_folder = None, save_name = None):
    with sns.axes_style("whitegrid"):
        Nsim=t.shape[0]
        N_obb = sim_obb.shape[0]
        
        plt.figure(figsize=(10,10), tight_layout=True)
        plt.title('Distance to obstacles covering ellipses centers')
        dist=np.zeros((Nsim, N_obb*3*3))
        for i in range(Nsim):
            dist_vector = constraint.dist(simX[i,:],np.array(sim_obb[:,i,:]).reshape((27)))
            for j in range(27):
                dist[i,j]= dist_vector[j]
        k=0
        for j in range(0,27):
            if j > 8:
                k = 1 
            if j > 17:
                k = 2
            plt.plot(t,dist[:,j], color=sns.color_palette()[k], label='Obstacle '+str(k) if j%9==0 else "")
        
        plt.plot([t[0],t[-1]],[constraint.dist_min, constraint.dist_min],'k--', label='min distance allowed')   
        plt.legend()
        plt.xlabel('t')
        plt.ylim([0, 60.0])
        plt.ylabel('dist [m]')
        if save_name != None:
            if os.path.exists(save_folder) == False:
                os.makedirs(save_folder)
            plt.savefig(save_folder + f"{save_name}.png")

def plotTTC(simX,sim_obb,constraint,t, save_folder = None, save_name = None):
    with sns.axes_style("whitegrid"):
        Nsim=t.shape[0]
        N_obb = sim_obb.shape[0]
        
        plt.figure(figsize=(10,10), tight_layout=True)
        plt.title('TTC for obstacles covering ellipses centers')
        ttc=np.ones((Nsim, N_obb*3*3))*1e3
        for i in range(Nsim):
            ttc_vector = constraint.ttc(simX[i,:],np.array(sim_obb[:,i,:]).reshape((27)))
            for j in range(27):
                ttc[i,j]= ttc_vector[j]
        k=0
        for j in range(0,27):
            if j > 8:
                k = 1 
            if j > 17:
                k = 2
            plt.plot(t,ttc[:,j], color=sns.color_palette()[k], label='Obstacle '+str(k) if j%9==0 else "")
        
        plt.plot([t[0],t[-1]],[constraint.ttc_min, constraint.ttc_min],'k--', label='min distance allowed')   
        plt.legend()
        plt.xlabel('t')
        plt.ylim([0,10.0])
        plt.ylabel('predicted ttc [s]')

def plot_results(path):
    # load results
    simX = np.load(path + '/simX.npy')
    simU = np.load(path + '/simU.npy')
    sim_obb = np.load(path + '/sim_obb.npy')
    predSimX = np.load(path + '/predSimX.npy')
    predSim_obb = np.load(path + '/predSim_obb.npy')

    import pickle as pkl
    # import as dictionary
    params = pkl.load(open(path + '/params.pkl', 'rb'))
    print(params)
    tcomp_sum = params['tcomp_sum']
    tcomp_max = params['tcomp_max']
    PREDICTION_HORIZON = params['PREDICTION_HORIZON']
    Nsim = params['Nsim']
    TIME_STEP = params['TIME_STEP']
    TRACK_FILE = params['TRACK_FILE']

    NUM_DISCRETIZATION_STEPS = int(PREDICTION_HORIZON / TIME_STEP)

    from acados_settings_dev import acados_settings
    constraint, model, acados_solver = acados_settings(PREDICTION_HORIZON, NUM_DISCRETIZATION_STEPS,
                                                        TRACK_FILE, simX[0, :])


    print(f"Average computation time: { tcomp_sum /  Nsim}")
    print(f"Maximum computation time: { tcomp_max}")
    print(f"Average speed: {np.average( simX[:, 3])} m/s")

    
    t = np.linspace(0.0,  Nsim *  PREDICTION_HORIZON /  NUM_DISCRETIZATION_STEPS,  Nsim)

    plotTrackProjfinal( simX,  sim_obb, # simulated trajectories
                            predSimX,  predSim_obb, # predicted trajectories
                            TRACK_FILE, )# SAVE_FIG_NAME
    
    plotDist( simX,  sim_obb,  constraint, t)

    plotRes( simX,  simU, t)

    plotTrackProj( simX, sim_obb, # simulated trajectories
                    predSimX,  predSim_obb, # predicted trajectories
                        TRACK_FILE,) # SAVE_GIF_NAME


    plt.show()

if __name__ == "__main__":

    os.chdir('/home/user/Documents/07_Dev/test_simu_trajectoires/acados_dev/mpc-ttc/results')
    root = tk.Tk()
    root.withdraw()
    
    path = filedialog.askdirectory(title="Select the results directory")
    if not path:
        raise ValueError("No directory selected")
    plot_results(path)