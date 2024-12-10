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


# ====== Functions for TTC computation ======
# At each dt we compute the predicted TTC between the two vehicles
# if the predicted TTC is less than T_thres AND collision is going to happen within the next T_thres
# -> true positive sample
# if the predicted TTC is less than T_thres AND collision is NOT going to happen within the next T_thres
# -> false positive sample
# if the predicted TTC is greater than T_thres AND collision is going to happen within the next T_thres
# -> false negative sample
# if the predicted TTC is greater than T_thres AND collision is NOT going to happen within the next T_thres
# -> true negative sample

# ====== Parameters ======
N = 200
dt = 0.1
T = N*dt
w = 0.132
h = 0.178
threshold = np.arange(dt, 10, dt)

def ttc_tests(dict_robot, collision, method: str, dt=0.1, p=0.99):
    assert method in ["ttc", "ttc_cov", "ttc_samples", "ttc_cov_optimist", "ttc_2nd", "ttc_2nd_cov", "ttc_2nd_cov_optimist"]
    
    cov = np.eye(2) * 0.0001 
    pred_TTC = []
    pred_cov_TTC = []
    real_TTC = []
    ttc_real_value = np.inf
    if collision:
        ttc_real_value = dict_robot[0].shape[0]*dt
        # plot_results(dict_robot)
        # plt.show()
    ego_robot = dict_robot[0]
    other_bot = dict_robot['obstacle']
    
    if other_bot is not None:
        # print(other_bot)
        
        obb_robot = dict_robot[other_bot]
    else:
        obb_robot = dict_robot[1]

    for i  in range(ego_robot.shape[0]):
    
        center1= ego_robot[i][:2]
        width1 = w
        height1 = h
        angle1 = ego_robot[i][2]
        v1 = ego_robot[i][3]
        w1 = ego_robot[i][4]

        if i >= obb_robot.shape[0]:
            center2 = obb_robot[-1][:2] + np.random.multivariate_normal([0, 0], cov)
            width2 = w
            height2 = h
            angle2 = obb_robot[-1][2]
            v2 = obb_robot[-1][3]
            w2 = obb_robot[-1][4]
        else:
            center2 = obb_robot[i][:2] + np.random.multivariate_normal([0, 0], cov)
            width2 = w
            height2 = h
            angle2 = obb_robot[i][2]
            v2 = obb_robot[i][3]
            w2 = obb_robot[i][4]

        if method == "ttc":
            pred_ttc_value = ttc(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2)
        elif method == "ttc_cov":
            pred_ttc_value = ttc_cov(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, p=p)
        elif method == "ttc_samples":
            pred_ttc_value = ttc_samples(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov)
        elif method == "ttc_cov_optimist":
            pred_ttc_value = ttc_cov_optimist(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, p=p)
        elif method == "ttc_2nd":
            pred_ttc_value = ttc_2nd_order(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2)
        elif method == "ttc_2nd_cov":
            pred_ttc_value = ttc_2nd_order_cov(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, p=p)
        elif method == "ttc_2nd_cov_optimist":
            pred_ttc_value = ttc_2nd_order_cov_optimist(center1, width1, height1, angle1, v1, w1, center2, width2, height2, angle2, v2, w2, cov, p=p)
        
        
        pred_TTC.append(pred_ttc_value)
        real_TTC.append(ttc_real_value - i * dt if collision else np.inf) 
    # plt.plot(pred_TTC, label="Predicted TTC")
    # plt.plot(real_TTC, label="Real TTC")
    # plt.legend()
    # plt.show()
    return np.array(pred_TTC), np.array(real_TTC)

def evaluate(predicted, gt, performance):
    # figure, ax = plt.subplots()
    k=0
    for thres in threshold:
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(predicted.shape[0]):
            if i + thres <= predicted.shape[0]:
                if (predicted[i] <= thres) and (predicted[i] > 0):#
                    if gt[i] < thres:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if gt[i] <= thres:
                        FN += 1
                    else:
                        TN += 1
            else:
                break
        
        # ax.plot(thres, TP /(TP + FP/2 + FN/2) if (TP + FP/2 + FN/2)!=0 else 0, 'ro'  )
    # plt.show()
        performance["TP"][k]+=TP
        performance["FP"][k]+=FP
        performance["FN"][k]+=FN
        performance["TN"][k]+=TN
        k+=1

def eval_per_scenario():
    percent = 0.0
    load_dir = "data3/"
    save_dir = "performances8/"
    
    for dir in os.listdir(load_dir):

        performance_ttc = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_cov = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_samples = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}

        performance_ttc_cov_2sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_cov_1sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)} 

        performance_ttc_cov_optimist = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_cov_optimist_2sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_cov_optimist_1sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}

        performance_ttc_2nd = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_cov_2nd = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        performance_ttc_optimist_2nd = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
        
        for dirs in tqdm.tqdm(os.listdir(os.path.join(load_dir,dir))):
            collision = False
            if dirs.startswith("collision"):
                collision = True
            if not collision:
                if np.random.rand() < percent:
                    continue
            for file in os.listdir(os.path.join(load_dir,dir,dirs)):
                if file.endswith(".pkl"):
                    dict_robot = pkl.load(open(load_dir+dir+"/"+dirs+"/"+file, "rb"))
                    
                    ttcs, gt = ttc_tests(dict_robot, collision, "ttc")
                    evaluate(ttcs, gt, performance_ttc)

                    ttcs_cov, gt = ttc_tests(dict_robot, collision, "ttc_cov")
                    evaluate(ttcs_cov, gt, performance_ttc_cov)

                    ttcs_cov, gt = ttc_tests(dict_robot, collision, "ttc_cov", p=0.95)
                    evaluate(ttcs_cov, gt, performance_ttc_cov_2sigma)

                    ttcs_cov, gt = ttc_tests(dict_robot, collision, "ttc_cov", p=0.64)
                    evaluate(ttcs_cov, gt, performance_ttc_cov_1sigma)

                    ttcs_samples, gt = ttc_tests(dict_robot, collision, "ttc_samples")
                    evaluate(ttcs_samples, gt, performance_ttc_samples)

                    ttcs_cov_optimist, gt = ttc_tests(dict_robot, collision, "ttc_cov_optimist")
                    evaluate(ttcs_cov_optimist, gt, performance_ttc_cov_optimist)

                    ttcs_cov_optimist, gt = ttc_tests(dict_robot, collision, "ttc_cov_optimist", p=0.95)
                    evaluate(ttcs_cov_optimist, gt, performance_ttc_cov_optimist_2sigma)

                    ttcs_cov_optimist, gt = ttc_tests(dict_robot, collision, "ttc_cov_optimist", p=0.64)
                    evaluate(ttcs_cov_optimist, gt, performance_ttc_cov_optimist_1sigma)

                    ttcs_2nd, gt = ttc_tests(dict_robot, collision, "ttc_2nd")
                    evaluate(ttcs_2nd, gt, performance_ttc_2nd)

                    ttcs_cov_2nd, gt = ttc_tests(dict_robot, collision, "ttc_2nd_cov")
                    evaluate(ttcs_cov_2nd, gt, performance_ttc_cov_2nd)

                    ttcs_cov_optimist_2nd, gt = ttc_tests(dict_robot, collision, "ttc_2nd_cov_optimist")
                    evaluate(ttcs_cov_optimist_2nd, gt, performance_ttc_optimist_2nd)

        if not os.path.exists(os.path.join(save_dir,dir)):
            os.makedirs(os.path.join(save_dir,dir))
        pkl.dump(performance_ttc, open(os.path.join(save_dir,dir) + f"performance_ttc_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov, open(os.path.join(save_dir,dir)+ f"performance_ttc_cov_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_samples, open(os.path.join(save_dir,dir) + f"performance_ttc_samples_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov_2sigma, open( os.path.join(save_dir,dir) + f"performance_ttc_cov_2sigma_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov_1sigma, open( os.path.join(save_dir,dir) + f"performance_ttc_cov_1sigma_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov_optimist, open( os.path.join(save_dir,dir) + f"performance_ttc_cov_optimist_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov_optimist_2sigma, open( os.path.join(save_dir,dir)+ f"performance_ttc_cov_optimist_2sigma_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov_optimist_1sigma, open( os.path.join(save_dir,dir) + f"performance_ttc_cov_optimist_1sigma_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_2nd, open( os.path.join(save_dir,dir) + f"performance_ttc_2nd_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_cov_2nd, open( os.path.join(save_dir,dir) + f"performance_ttc_cov_2nd_{percent}.pkl", "wb"))
        pkl.dump(performance_ttc_optimist_2nd, open( os.path.join(save_dir,dir) + f"performance_ttc_optimist_2nd_{percent}.pkl", "wb"))

def test():
    percent = 0.0
    load_dir = "data4/"
    save_dir = "performances8/"

    performance_ttc = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_cov = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_samples = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}

    performance_ttc_cov_2sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_cov_1sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)} 

    performance_ttc_cov_optimist = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_cov_optimist_2sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_cov_optimist_1sigma = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}

    performance_ttc_2nd = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_cov_2nd = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}
    performance_ttc_optimist_2nd = {"TP": np.zeros_like(threshold), "FP": np.zeros_like(threshold), "FN": np.zeros_like(threshold), "TN": np.zeros_like(threshold)}

    for dir in os.listdir(load_dir):
        

        for dirs in tqdm.tqdm(os.listdir(os.path.join(load_dir,dir))):
            collision = False
            if dirs.startswith("collision"):
                collision = True
            if not collision:
                if np.random.rand() < percent:
                    continue
            for file in os.listdir(os.path.join(load_dir,dir,dirs)):
                if file.endswith(".pkl"):
                    dict_robot = pkl.load(open(load_dir+dir+"/"+dirs+"/"+file, "rb"))
                    
                    ttcs, gt = ttc_tests(dict_robot, collision, "ttc")
                    evaluate(ttcs, gt, performance_ttc)

                    ttcs_cov, gt = ttc_tests(dict_robot, collision, "ttc_cov")
                    evaluate(ttcs_cov, gt, performance_ttc_cov)

                    ttcs_cov, gt = ttc_tests(dict_robot, collision, "ttc_cov", p=0.95)
                    evaluate(ttcs_cov, gt, performance_ttc_cov_2sigma)

                    ttcs_cov, gt = ttc_tests(dict_robot, collision, "ttc_cov", p=0.64)
                    evaluate(ttcs_cov, gt, performance_ttc_cov_1sigma)

                    ttcs_samples, gt = ttc_tests(dict_robot, collision, "ttc_samples")
                    evaluate(ttcs_samples, gt, performance_ttc_samples)

                    ttcs_cov_optimist, gt = ttc_tests(dict_robot, collision, "ttc_cov_optimist")
                    evaluate(ttcs_cov_optimist, gt, performance_ttc_cov_optimist)

                    ttcs_cov_optimist, gt = ttc_tests(dict_robot, collision, "ttc_cov_optimist", p=0.95)
                    evaluate(ttcs_cov_optimist, gt, performance_ttc_cov_optimist_2sigma)

                    ttcs_cov_optimist, gt = ttc_tests(dict_robot, collision, "ttc_cov_optimist", p=0.64)
                    evaluate(ttcs_cov_optimist, gt, performance_ttc_cov_optimist_1sigma)

                    ttcs_2nd, gt = ttc_tests(dict_robot, collision, "ttc_2nd")
                    evaluate(ttcs_2nd, gt, performance_ttc_2nd)

                    ttcs_cov_2nd, gt = ttc_tests(dict_robot, collision, "ttc_2nd_cov")
                    evaluate(ttcs_cov_2nd, gt, performance_ttc_cov_2nd)

                    ttcs_cov_optimist_2nd, gt = ttc_tests(dict_robot, collision, "ttc_2nd_cov_optimist")
                    evaluate(ttcs_cov_optimist_2nd, gt, performance_ttc_optimist_2nd)


    pkl.dump(performance_ttc, open( save_dir + f"performance_ttc_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov, open( save_dir + f"performance_ttc_cov_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_samples, open(save_dir + f"performance_ttc_samples_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov_2sigma, open( save_dir + f"performance_ttc_cov_2sigma_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov_1sigma, open( save_dir + f"performance_ttc_cov_1sigma_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov_optimist, open( save_dir + f"performance_ttc_cov_optimist_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov_optimist_2sigma, open( save_dir + f"performance_ttc_cov_optimist_2sigma_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov_optimist_1sigma, open( save_dir + f"performance_ttc_cov_optimist_1sigma_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_2nd, open( save_dir + f"performance_ttc_2nd_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_cov_2nd, open( save_dir + f"performance_ttc_cov_2nd_{percent}.pkl", "wb"))
    pkl.dump(performance_ttc_optimist_2nd, open( save_dir + f"performance_ttc_optimist_2nd_{percent}.pkl", "wb"))

def main():
    eval_per_scenario()
    # test()

if __name__ == "__main__":
    main()