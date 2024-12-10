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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

plt.rcParams['text.usetex'] = False

def plotTrackProj(trajectories,filename='LMS_Track.txt',ax=None):
    with sns.axes_style("whitegrid"):
        if trajectories.ndim == 2:
            trajectories = np.expand_dims(trajectories, axis=0)
        assert trajectories.ndim == 3, "Trajectories must be 3D array"
        distance=0.15
        
        # plot racetrack map
        no_ax = False
        #Setup plot
        if ax == None:
            no_ax = True
            fig, ax = plt.subplots()
        ax.set_ylim(bottom=-2,top=0.5)
        ax.set_xlim(left=-1.1,right=1.6)
        ax.set_ylabel('y[m]')
        ax.set_xlabel('x[m]')

        # Plot center line
        [Sref,Xref,Yref,Psiref,_]=getTrack(filename)
        ax.plot(Xref[0:17],Yref[0:17],'-',color='k',linewidth=0.5)
        ax.plot(Xref[200:],Yref[200:],'-',color='k',linewidth=0.5)

        # Draw Trackboundaries
        Xboundleft=Xref-distance*np.sin(Psiref)
        Yboundleft=Yref+distance*np.cos(Psiref)
        Xboundright=Xref+distance*np.sin(Psiref)
        Yboundright=Yref-distance*np.cos(Psiref)
        Xbound_otherlaneleft=Xref-2*distance*np.sin(Psiref)
        Ybound_otherlaneleft=Yref+2*distance*np.cos(Psiref)
        Xbound_otherright=Xref-3*distance*np.sin(Psiref)
        Ybound_otherright=Yref+3*distance*np.cos(Psiref)
        ax.plot(Xboundleft[0:17],Yboundleft[0:17],'k--',linewidth=2)
        ax.plot(Xboundleft[200:],Yboundleft[200:],'k--',linewidth=2)
        ax.plot(Xboundright[0:17],Yboundright[0:17],color='k',linewidth=2)
        ax.plot(Xboundright[200:],Yboundright[200:],color='k',linewidth=2)
        ax.plot(Xbound_otherlaneleft[0:17],Ybound_otherlaneleft[0:17],'k-',linewidth=0.5)
        ax.plot(Xbound_otherlaneleft[200:],Ybound_otherlaneleft[200:],'k-',linewidth=0.5)
        ax.plot(Xbound_otherright[0:17],Ybound_otherright[0:17],'k-',linewidth=2)
        ax.plot(Xbound_otherright[200:],Ybound_otherright[200:],'k-',linewidth=2)
        

        # Draw driven trajectory
        # draw geometries 

        for i in range(0, trajectories.shape[-1]):
            simX=trajectories[:,:,i]
            s=simX[:,0]
            n=simX[:,1]
            alpha=simX[:,2]
            v=simX[:,3]
            # transform data
            [x, y, psi, _] = transformProj2Orig(s, n, alpha, v,filename)
            ax.plot(x,y, '-b')
            heatmap = ax.scatter(x,y, c=v, cmap=cm.rainbow, edgecolor='none', marker='o')
            for i in range(0, len(x), 5):
                rectangle = Rectangle((x[i]-0.132/2, y[i]-0.178/2),0.132, 0.178,  angle=psi[i]*180/np.pi, rotation_point='center',
                                    edgecolor='black', facecolor='none')
                ax.add_patch(rectangle)
        if no_ax:
            plt.colorbar(heatmap, ax=ax, label='v[m/s]')
        ax.set_aspect('equal', 'box')
    

        # # Put markers for s values
        # xi=np.zeros(9)
        # yi=np.zeros(9)
        # xi1=np.zeros(9)
        # yi1=np.zeros(9)
        # xi2=np.zeros(9)
        # yi2=np.zeros(9)
        # for i in range(int(Sref[-1]) + 1):
        #     try:
        #         k = list(Sref).index(i + min(abs(Sref - i)))
        #     except:
        #         k = list(Sref).index(i - min(abs(Sref - i)))
        #     [_,nrefi,_,_]=transformOrig2Proj(Xref[k],Yref[k],Psiref[k],0)
        #     [xi[i],yi[i],_,_]=transformProj2Orig(Sref[k],nrefi+0.24,0,0)
        #     # plt.text(xi[i], yi[i], f'{i}m', fontsize=12,horizontalalignment='center',verticalalignment='center')
        #     plt.text(xi[i], yi[i], '{}m'.format(i), fontsize=12,horizontalalignment='center',verticalalignment='center')
        #     [xi1[i],yi1[i],_,_]=transformProj2Orig(Sref[k],nrefi+0.12,0,0)
        #     [xi2[i],yi2[i],_,_]=transformProj2Orig(Sref[k],nrefi+0.15,0,0)
        #     plt.plot([xi1[i],xi2[i]],[yi1[i],yi2[i]],color='black')

def plotRes(simX,simU,t):
    with sns.axes_style("whitegrid"):
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
    with sns.axes_style("whitegrid"):
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

def plotdist(simX,obbX, constraint,t, ax=None):
    with sns.axes_style("whitegrid"):
        Nsim=t.shape[0]
        if ax == None:
            fig, ax = plt.subplots()

        
        dist=np.zeros(Nsim)
        for i in range(Nsim):
            [x,y,psi,v]=transformProj2Orig(simX[i,0], simX[i,1], simX[i,2], simX[i,3])
            [x_obst, y_obst, psi_obst,v_obst] = transformProj2Orig(obbX[i,0], obbX[i,1], obbX[i,2], obbX[i,3])
            dist[i]=np.sqrt((x-x_obst)**2+(y-y_obst)**2)

        d_0 = dist[-1]
        vxr = (v * np.cos(psi) - v_obst * np.cos(psi_obst))
        vyr = (v * np.sin(psi) - v_obst * np.sin(psi_obst))
        d_dot_0 = ((x - x_obst) * vxr + (y - y_obst) * vyr)/d_0
        d_dot_dot_0 = ((vxr**2 + vyr**2) - d_dot_0**2)/d_0
        k = np.arange(t[-1], t[-1]+2, 0.1)
        d_t0 = d_0 + d_dot_0 * (k - t[-1]) + d_dot_dot_0 * (k - t[-1])**2 
        
        ax.plot(k,d_t0, 'r')
        if d_dot_0[-1]<0:
            print('warning collision risk')

        ax.plot(t,dist, 'b')
        ax.plot([t[0],t[-1]],[constraint.dist_min, constraint.dist_min],'k--')
        ax.plot([t[0],t[-1]],[2*np.sqrt((0.178/2)**2 + (0.132/2)**2/4),2*np.sqrt((0.178/2)**2 + (0.132/2)**2/4)],'r--')
        ax.legend(['dist','dist_min'])
        ax.set_xlabel('t')
        ax.set_ylabel('dist[m/s^2]')

def plot_ttc(simX,obbX, constraint,t, ax=None):
    Nsim=t.shape[0]
    ttc=np.zeros(Nsim)
    if ax == None:
        fig, ax = plt.subplots()
    for i in range(Nsim):
        [x,y,psi,v]=transformProj2Orig(simX[i,0], simX[i,1], simX[i,2], simX[i,3])
        [x_obst, y_obst, psi_obst,v_obst] = transformProj2Orig(obbX[i,0], obbX[i,1], obbX[i,2], obbX[i,3])
        dist=np.sqrt((x-x_obst)**2+(y-y_obst)**2)
        vxr = (v * np.cos(psi) - v_obst * np.cos(psi_obst))
        vyr = (v * np.sin(psi) - v_obst * np.sin(psi_obst))
        d_dot = ((x - x_obst) * vxr + (y - y_obst) * vyr)/dist
        if d_dot > 0:
            ttc[i] = 20
        else:
            ttc[i] = (dist - 2*np.sqrt((0.178/2)**2 + (0.132/2)**2/4))/np.abs(d_dot)
    plt.plot(t,ttc)
    plt.plot([t[0],t[-1]],[constraint.ttc_min, constraint.ttc_min],'k--')
    plt.legend(['ttc','ttc_min'])
    plt.xlabel('t')
    plt.ylabel('ttc[s]')



def plotSimu(simX,t,simobbX, constraint,track, axes=None):

    ax1 = axes[0]
    ax2 = axes[1]
    for i, ti in enumerate(t):
        if i <= 1:
            continue
        ax1.cla()
        ax2.cla()
        plotTrackProj(np.dstack([simX[i,:], simobbX[i,:]]), track, ax1)
        plotdist(simX[:i,:], simobbX[:i,:], constraint, t[:i], ax2)
        plt.pause(0.2)
    plot_ttc(simX, simobbX, constraint, t)



