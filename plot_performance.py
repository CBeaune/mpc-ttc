import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import os 
import pickle as pkl

from utils.visualization import plot_robot, plot_cov_ellipse
from utils.control import pid_control, simulate, closest_waypoint
from utils.cubic_spline_planner import calc_spline_course
from utils.lqr_speed_steer_control import do_simulation, calc_speed_profile
from utils.rectangles import get_corners, closest_points

from ttc_computation import ttc, ttc_cov, ttc_samples, ttc_cov_optimist, ttc_2nd_order, ttc_2nd_order_cov, ttc_2nd_order_cov_optimist
from simu_trajectories import plot_results

dt = 0.1

def load_data(dir = "performances6/"):
    percent = 0.0
    performance_ttc = pkl.load(open(dir + f"performance_ttc_{percent}.pkl", "rb")) if percent is not None else pkl.load(open(dir + "performance_ttc.pkl", "rb"))
    performance_ttc_cov = pkl.load(open(dir + f"performance_ttc_cov_{percent}.pkl", "rb")) if percent is not None else pkl.load(open(dir + "performance_ttc_cov.pkl", "rb"))
    performance_ttc_samples = pkl.load(open( dir +f"performance_ttc_samples_{percent}.pkl", "rb")) if percent  is not None else pkl.load(open(dir + "performance_ttc_samples.pkl", "rb"))
    performance_ttc_cov_1sigma = pkl.load(open( dir +f"performance_ttc_cov_1sigma_{percent}.pkl", "rb")) if percent is not None  else pkl.load(open(dir + "performance_ttc_cov_1sigma.pkl", "rb"))
    performance_ttc_cov_2sigma = pkl.load(open( dir +f"performance_ttc_cov_2sigma_{percent}.pkl", "rb")) if percent is not None else pkl.load(open(dir + "performance_ttc_cov_2sigma.pkl", "rb"))
    performance_ttc_cov_optimist = pkl.load(open( dir +f"performance_ttc_cov_optimist_{percent}.pkl", "rb")) if percent is not None else pkl.load(open(dir + "performance_ttc_cov_optimist.pkl", "rb"))
    performance_ttc_cov_optimist_1sigma = pkl.load(open( dir +f"performance_ttc_cov_optimist_1sigma_{percent}.pkl", "rb")) if percent is not None else pkl.load(open(dir + "performance_ttc_cov_optimist_1sigma.pkl", "rb"))
    performance_ttc_cov_optimist_2sigma = pkl.load(open( dir +f"performance_ttc_cov_optimist_2sigma_{percent}.pkl", "rb")) if percent is not None else pkl.load(open(dir + "performance_ttc_cov_optimist_2sigma.pkl", "rb"))
    performance_ttc_2nd = pkl.load(open( dir +f"performance_ttc_2nd_{percent}.pkl", "rb"))
    performance_ttc_cov_2nd = pkl.load(open( dir +f"performance_ttc_cov_2nd_{percent}.pkl", "rb"))
    performance_ttc_optimist_2nd = pkl.load(open( dir +f"performance_ttc_optimist_2nd_{percent}.pkl", "rb"))


    
    return [performance_ttc, performance_ttc_cov, performance_ttc_samples,
            performance_ttc_cov_1sigma, performance_ttc_cov_2sigma, performance_ttc_cov_optimist,
            performance_ttc_cov_optimist_1sigma,performance_ttc_cov_optimist_2sigma,
            performance_ttc_2nd, performance_ttc_cov_2nd, performance_ttc_optimist_2nd]

def load_data_percent():
    percent = [0.0, 0.25, 0.5, 0.75]
    dir = "performances6/"
    performances = []
    for p in percent:
        # load data for each percentage
        performance_ttc = pkl.load(open(dir + f"performance_ttc_{p}.pkl", "rb"))
        performance_ttc_cov = pkl.load(open(dir + f"performance_ttc_cov_{p}.pkl", "rb"))
        performance_ttc_samples = pkl.load(open( dir +f"performance_ttc_samples_{p}.pkl", "rb"))
        performance_ttc_cov_1sigma = pkl.load(open( dir +f"performance_ttc_cov_1sigma_{p}.pkl", "rb"))
        performance_ttc_cov_2sigma = pkl.load(open( dir +f"performance_ttc_cov_2sigma_{p}.pkl", "rb"))
        performance_ttc_cov_optimist = pkl.load(open( dir +f"performance_ttc_cov_optimist_{p}.pkl", "rb"))
        performance_ttc_cov_optimist_1sigma = pkl.load(open( dir +f"performance_ttc_cov_optimist_1sigma_{p}.pkl", "rb"))
        performance_ttc_cov_optimist_2sigma = pkl.load(open( dir +f"performance_ttc_cov_optimist_2sigma_{p}.pkl", "rb"))
        performance_ttc_2nd = pkl.load(open( dir +f"performance_ttc_2nd_{p}.pkl", "rb"))
        performance_ttc_cov_2nd = pkl.load(open( dir +f"performance_ttc_cov_2nd_{p}.pkl", "rb"))
        performance_ttc_optimist_2nd = pkl.load(open( dir +f"performance_ttc_optimist_2nd_{p}.pkl", "rb"))

        performances.append([performance_ttc, performance_ttc_cov, performance_ttc_samples,
                             performance_ttc_cov_1sigma, performance_ttc_cov_2sigma,
                            performance_ttc_cov_optimist, performance_ttc_cov_optimist_1sigma,
                              performance_ttc_cov_optimist_2sigma,
                              performance_ttc_2nd, performance_ttc_cov_2nd, performance_ttc_optimist_2nd])
    return np.array(performances)


# ====== Functions for TTC computation ======
def plot_eta(performances):
    titles = [ "TTC Cov 3 sigma",  "TTC Cov 1 sigma", "TTC Cov 2 sigma", ]
    titles = ["TTC Cov Optimist 3 sigma", "TTC Cov Optimist 1 sigma", "TTC Cov Optimist 2 sigma"]
    # titles = ["0.0 %", "25.0 %", "50.0 %", "75.0 %"]
    # "TTC", "TTC Samples", "TTC Cov Optimist", "TTC Cov Optimist 1 sigma", "TTC Cov Optimist 2 sigma"
    plt.figure(figsize=(12,7))
    plt.rcParams.update({'font.size': 14})
    # update font size for axes
    plt.rcParams.update({'axes.labelsize': 14})
    for i,performance in enumerate(performances):
        label = titles[i]
        with sns.axes_style("whitegrid"):
            T = len(performance["TP"])*dt + dt
            threshold = np.arange(dt, T, dt)
            Precision1 = performance["TP"]/(performance["TP"] + performance["FP"]) 
            Recall1 = performance["TP"]/(performance["TP"] + performance["FN"])
            Accuracy1 = (performance["TP"] + performance["TN"])/(performance["TP"] + performance["FP"] + performance["FN"] + performance["TN"])
            F11 = 2*performance["TP"]/(2*performance["TP"] + performance["FP"] + performance["FN"])   
            
            # plt.plot(threshold, Precision1, label="Precision")
            # plt.plot(threshold, Accuracy1, label="Accuracy")
            # plt.plot(threshold, Recall1, label="Recall")
            plt.plot(threshold, F11, label=label)       
            plt.legend()
            plt.ylabel("score")
            plt.xlabel("TTC threshold (s)")

    plt.title("F1 score for different confidence levels" )

    plt.show()

def plot_compared(performances):
    titles = [ "TTC","TTC Cov 3 sigma", "TTC Samples", "TTC Cov Optimist",]
    # ,  "TTC Cov 1 sigma", "TTC Cov 2 sigma",  "TTC Cov Optimist 1 sigma", "TTC Cov Optimist 2 sigma"
    plt.figure(figsize=(12,7))
    plt.rcParams.update({'font.size': 14})
    # update font size for axes
    plt.rcParams.update({'axes.labelsize': 14})
    for i,performance in enumerate(performances):
        label = titles[i]
        with sns.axes_style("whitegrid"):
            T = len(performance["TP"])*dt + dt
            threshold = np.arange(dt, T, dt)
            Precision1 = performance["TP"]/(performance["TP"] + performance["FP"]) 
            Recall1 = performance["TP"]/(performance["TP"] + performance["FN"])
            Accuracy1 = (performance["TP"] + performance["TN"])/(performance["TP"] + performance["FP"] + performance["FN"] + performance["TN"])
            F11 = 2*performance["TP"]/(2*performance["TP"] + performance["FP"] + performance["FN"])   
            
            # plt.plot(threshold, Precision1, label="Precision")
            # plt.plot(threshold, Accuracy1, label="Accuracy")
            # plt.plot(threshold, Recall1, label="Recall")
            plt.plot(threshold, F11, label=label)       
            plt.legend()
            plt.ylabel("score")
            plt.xlabel("TTC threshold (s)")

    plt.title("F1 score comparison" )

    plt.show()

def plot_percent(performances, labels = ["0.0 %", "25.0 %", "50.0 %", "75.0 %"], title = "F1 score comparison for different percentage of erased non-collision scenarios"):
    # "TTC", "TTC Samples", "TTC Cov Optimist", "TTC Cov Optimist 1 sigma", "TTC Cov Optimist 2 sigma"
    plt.figure(figsize=(12,7))
    plt.rcParams.update({'font.size': 14})
    # update font size for axes
    plt.rcParams.update({'axes.labelsize': 14})
    for i,performance in enumerate(performances):
        label = labels[i]
        with sns.axes_style("whitegrid"):
            T = len(performance["TP"])*dt + dt
            threshold = np.arange(dt, T, dt)
            Precision1 = performance["TP"]/(performance["TP"] + performance["FP"]) 
            Recall1 = performance["TP"]/(performance["TP"] + performance["FN"])
            Accuracy1 = (performance["TP"] + performance["TN"])/(performance["TP"] + performance["FP"] + performance["FN"] + performance["TN"])
            F11 = 2*performance["TP"]/(2*performance["TP"] + performance["FP"] + performance["FN"])   
            
            # plt.plot(threshold, Precision1, label="Precision")
            # plt.plot(threshold, Accuracy1, label="Accuracy")
            # plt.plot(threshold, Recall1, label="Recall")
            plt.plot(threshold, F11, label=label)       
            plt.legend()
            plt.ylabel("score")
            plt.xlabel("TTC threshold (s)")

    plt.title(title)


def plot_scenario_and_ttc():
    # load data
    dict_robot = pkl.load(open("data3/twolanes/close/scenario_1.pkl", "rb"))
    X_robot0 = dict_robot[0]
    X_robot1 = dict_robot[1]
    X_robot2 = dict_robot[2]
    t_coll = dict_robot[0].shape[0]
    h = 0.178
    w = 0.132

    # === Plot_trajectories =====
    fig, ax= plt.subplots()
    fig, ax1 = plt.subplots()
    
    ttc_ = []
    ttc_cov_ = []
    ttc_cov_optimist_ = []

    ttc_2nd = []
    ttc_cov_2nd = []
    ttc_cov_optimist_2nd = []
    
    for i in range(0, X_robot0.shape[0]):
        with sns.axes_style("whitegrid"):
            if i in [0, 25, 50, 85, t_coll-1]:
                bold = 1
                ax.text(X_robot0[i,0] + 0.1,X_robot0[i,1]- 0.1, f"t = {i*dt:.1f}s", fontsize=12)
                ax.text(X_robot2[i,0]+ 0.1,X_robot2[i,1]- 0.1, f"t = {i*dt:.1f}s", fontsize=12)
            else:
                bold = 0.1
            plot_robot(X_robot0[i,0],X_robot0[i,1],X_robot0[i,2], 2*h, 2*w, ax, color = sns.color_palette()[0], alpha = bold)
            if i == X_robot1.shape[0]:
                X_robot1 = np.append(X_robot1, [X_robot1[-1,:]], axis=0)
            plot_robot(X_robot1[i,0],X_robot1[i,1],X_robot1[i,2], 2*h, 2*w, ax, color = sns.color_palette()[1], alpha = bold)
            if i == X_robot2.shape[0]:
                X_robot2 = np.append(X_robot2, [X_robot2[-1,:]], axis=0)
            X_robot2[i,:2] += np.random.multivariate_normal([0,0], np.array([[0.001, 0], [0, 0.005]])) 
            plot_robot(X_robot2[i,0],X_robot2[i,1],X_robot2[i,2], 2*h, 2*w, ax, color = sns.color_palette()[2], alpha = bold)

            ttc_value = ttc(X_robot0[i,:2], w, h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                            X_robot2[i,:2], w, h, X_robot2[i,2], X_robot2[i,3], X_robot2[i,4])
            
            ttc_.append(ttc_value if ttc_value < 10 else 10)

            ttc_value = ttc_cov(X_robot0[i,:2], w, h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                                    X_robot2[i,:2], w, h, X_robot2[i,2], X_robot2[i,3], X_robot2[i,4],
                                    np.array([[0.0001, 0], [0, 0.0005]]))
            ttc_cov_.append(ttc_value if ttc_value <10 else 10)

            ttc_value = ttc_cov_optimist(X_robot0[i,:2], w, h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                                                  X_robot2[i,:2], w, h, X_robot2[i,2], X_robot2[i,3], X_robot2[i,4],
                                                    np.array([[0.0001, 0], [0, 0.0005]]))
            ttc_cov_optimist_.append( ttc_value if ttc_value < 10 else 10)

            ttc_value = ttc_2nd_order(X_robot0[i,:2], w, h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                            X_robot2[i,:2], w, h, X_robot2[i,2], X_robot2[i,3], X_robot2[i,4])
            ttc_2nd.append(ttc_value if ttc_value < 10 else 10)

            ttc_value = ttc_2nd_order_cov(X_robot0[i,:2], w, h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                                    X_robot2[i,:2], w, h, X_robot2[i,2], X_robot2[i,3], X_robot2[i,4],
                                    np.array([[0.0001, 0], [0, 0.0005]]))
            ttc_cov_2nd.append(ttc_value if ttc_value <10 else 10)

            ttc_value = ttc_2nd_order_cov_optimist(X_robot0[i,:2], w, h, X_robot0[i,2], X_robot0[i,3], X_robot0[i,4],
                                                    X_robot2[i,:2], w, h, X_robot2[i,2], X_robot2[i,3], X_robot2[i,4],
                                                        np.array([[0.0001, 0], [0, 0.0005]]))
            ttc_cov_optimist_2nd.append( ttc_value if ttc_value < 10 else 10)




            if VIZ:
                    plt.pause(dt/100)

        ax.axes.set_aspect('equal')
    ax.grid(True)  
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    t = np.arange(0, X_robot0.shape[0]*dt, dt) 
    with sns.axes_style("whitegrid"):        
        ax1.plot(t, ttc_, label = "TTC")
        ax1.plot(t, ttc_cov_,  label = "TTC_cov" )
        ax1.plot(t, ttc_cov_optimist_, label = "TTC_cov_optimist")
        ax1.plot(t, ttc_2nd, '--', color = sns.color_palette()[0], label = "TTC_2nd")
        ax1.plot(t, ttc_cov_2nd,'--', color = sns.color_palette()[1], label = "TTC_cov_2nd")
        ax1.plot(t, ttc_cov_optimist_2nd,'--', color = sns.color_palette()[2], label = "TTC_cov_optimist_2nd")
        ax1.vlines([0.0, 2.5, 5.0, 8.5], -2.0, 12, color = 'black', linestyle = '--')
        ax1.vlines(t_coll*dt, -2.0, 12, color = 'black', linestyle = '--')
        ax.set_xlabel("Time (s)")
        ax1.set_ylabel("predicted TTC (s)")
        
        ax1.grid(True)
    
    plt.legend()
    plt.show()


def test_computation_time():

   
    cov = np.array([[0.01, 0], [0, 0.05]])
    import time
    ttc_time = []
    ttc_cov_time = []
    ttc_samples_time = []
    ttc_co_time = []
    w, h = 0.178, 0.132
    for i in tqdm.tqdm(range(2000)):
        X1 = np.random.uniform(0, 0.5, (3,))
        X2 = np.random.uniform(0.5, 1, (3,))
        while np.linalg.norm(X1[:2] - X2[:2]) < 0.3:
            X2 = np.random.uniform(0.5, 1, (3,))
        V1 = np.random.uniform(0, 0.5, (2,))
        V2 = np.random.uniform(0, 0.5, (2,))
        t0 = time.time()
        # print(X1, V1, X2, V2)
        ttc(X1[:2], w, h, X1[2], V1[0], V1[1], X2[:2], w, h, X1[2], V2[0], V1[1])
        ttc_time.append((time.time() - t0)*150)
        t0 = time.time()
        ttc_cov(X1[:2], w, h, X1[2], V1[0], V1[1], X2[:2], w, h, X1[2], V2[0], V1[1], cov)
        ttc_cov_time.append((time.time() - t0)*150)
        t0 = time.time()
        ttc_samples(X1[:2], w, h, X1[2], V1[0], V1[1], X2[:2], w, h, X1[2], V2[0], V1[1], cov)
        ttc_samples_time.append((time.time() - t0)*150)
        t0 = time.time()
        ttc_cov_optimist(X1[:2], w, h, X1[2], V1[0], V1[1], X2[:2], w, h, X1[2], V2[0], V1[1], cov)
        ttc_co_time.append((time.time() - t0)*150)
    
    with sns.axes_style("whitegrid"):
        plt.figure()
        plt.rcParams.update({'font.size': 14})
        # update font size for axes
        plt.rcParams.update({'axes.labelsize': 14})
        sns.boxplot(data = [ttc_time, ttc_cov_time, ttc_samples_time, ttc_co_time], 
                     linewidth=.75,  showfliers = False,
                    )
        plt.xticks(range(4), ["TTC", "TTC Cov", "TTC Samples", "TTC Cov Optimist"])
        positions = np.arange(0, 2, 0.1)
        # Définir les étiquettes des graduations tous les 0.2
        # labels = [f"{i*dt:.1f}" if i % 2 == 0 else '' for i,_ in enumerate(positions)]

        plt.yticks(positions)  # Appliquer les positions et les étiquettes
        plt.yscale("log")
        plt.grid(which='both', linestyle='-', linewidth=0.5)
        plt.title("Computation time for 150 prediction samples")
        plt.ylabel("Time (s)")
        plt.show()

def dataset_composition(): 
    load_dir = "data3/"
    
    plt.rcParams.update({'font.size': 14})
    # update font size for axes
    plt.rcParams.update({'axes.labelsize': 14})
    with sns.axes_style("whitegrid"):
        fig1, ax1 = plt.subplots(1,1,figsize=(12,7))
        fig2, ax2 = plt.subplots(1,1,figsize=(12,7))
        for i, dir in enumerate(os.listdir(load_dir)):
            if dir == 'crossing_0_1':
                i = 1
            elif dir == 'crossing_0_2':
                i = 2
            elif dir == 'crossing_1_1':
                i = 3
            elif dir == 'crossing_1_2':
                i = 4
            else:
                i = 0

            collision_num = 0
            clear_num = 0
            close_num = 0
            size_data_collision = []
            size_data_close = []
            size_data_clear = []
            for dirs in os.listdir(os.path.join(load_dir,dir)):
                if dirs == "collision":
                    collision_num = len(os.listdir(os.path.join(load_dir,dir, dirs)))
                    for file in os.listdir(os.path.join(load_dir,dir, dirs)):
                        dict_robot = pkl.load(open(os.path.join(load_dir,dir, dirs, file), "rb"))
                        size_data_collision.append(dict_robot[0].shape[0] +20)
                elif dirs == "clear":
                    clear_num = len(os.listdir(os.path.join(load_dir,dir, dirs)))
                    for file in os.listdir(os.path.join(load_dir,dir, dirs)):
                        dict_robot = pkl.load(open(os.path.join(load_dir,dir, dirs, file), "rb"))
                        size_data_clear.append(dict_robot[0].shape[0]+20)
                else: 
                    close_num = len(os.listdir(os.path.join(load_dir,dir, dirs)))
                    for file in os.listdir(os.path.join(load_dir,dir, dirs)):
                        dict_robot = pkl.load(open(os.path.join(load_dir,dir, dirs, file), "rb"))
                        size_data_close.append(dict_robot[0].shape[0]+20)
                #plot histogram of the number of samples for each category
            
                
            
                    
            ax1.bar([i-0.3, i, i+0.3], [collision_num, clear_num, close_num],
                    color = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2]],
                     width = 0.3, label=['collision', 'clear', 'close'] if i == 0 else None)
            data = [size_data_collision, size_data_clear, size_data_close]
            colors = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2]]
            positions = [i-0.3, i, i+0.3]
            labels = ['', f'S{i}',  '']
            for k, d in enumerate(data):
                plt.rcParams.update({'font.size': 16})
                # update font size for axes
                plt.rcParams.update({'axes.labelsize': 16})
                ax2.boxplot(d,patch_artist=True, 
                                boxprops=dict(facecolor=colors[k], color=colors[k], alpha=0.6),
                                medianprops=dict(color="black"),
                                positions=[positions[k]], showfliers=False, labels = [labels[k]])
                ax2.scatter([positions[k]] * len(d), d, color=colors[k])
        import matplotlib.patches as mpatches
        legend_elements = [mpatches.Patch(color=colors[0], label='Collision'),
                   mpatches.Patch(color=colors[1], label='Clear'),
                   mpatches.Patch(color=colors[2], label='Close')]
                

        ax1.set_xticks(range(5), ["S0", "S1", "S2", "S3", "S4"])
        ax1.set_xlabel("Scenarios type")
        ax1.set_ylabel("Number of scenarios")
        ax1.set_ylim(0, 35)
        ax1.legend()
        ax1.set_title("Dataset composition")

        ax2.set_xticks(range(5), ["S0", "S1", "S2", "S3", "S4"])
        ax2.set_ylim(0, 210)
        ax2.set_xlim(-1, 5)
        ax2.set_xlabel("Scenarios type")
        ax2.set_ylabel("Duration (s)")
        ax2.set_title("Duration of scenarios")
        ax2.legend(handles=legend_elements, loc='upper left')

    plt.show()
                

VIZ = False
def main():
    
    # test_computation_time()
    dataset_composition()
    # performances = load_data()
    # performances_eta = [performances[1], performances[3], performances[4]]
    # performances_compared = [performances[0],performances[1], performances[2], performances[5]]
    # performances_optimist = [performances[5], performances[6], performances[7]]
    # # print("data loaded")

    # plot_eta(performances_eta)
    # plot_compared(performances_compared)
    # plot_eta(performances_optimist)

    # plot_scenario_and_ttc()

    # performances = load_data()
    # performances_compared = [performances[8],performances[9], performances[10]]
    # plot_percent(performances_compared, labels = ["TTC 2nd", "TTC Cov 2nd", "TTC Cov Optimist 2nd",], 
    #              title = "F1 score comparison for 2nd order models")

    # performances_percent = load_data_percent()
    # performances_compared_percent = [performances_percent[0,1],performances_percent[1,1], performances_percent[2,1], performances_percent[3,1]]
    # plot_percent(performances_compared_percent)

    # performances_compared_percent = [performances_percent[0,5],performances_percent[1,5], performances_percent[2,5], performances_percent[3,5]]
    # plot_percent(performances_compared_percent)

    # ========= Plot results per scenarios ===========
    load_dir = "performances7/"
    for dir in os.listdir(load_dir):
        data = load_data(load_dir + dir + "/" + dir)
        peformances = [data[0], data[1], data[2], data[5]]
        plot_percent(peformances, labels = ["TTC", "TTC Cov", "TTC Samples", "TTC Cov Optimist"], title = f"Performance for {dir}")
    plt.show()


if __name__ == "__main__":
    main()