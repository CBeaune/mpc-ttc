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

from casadi import *
from tracks.readDataFcn import getTrack

LENGTH = 0.25
WIDTH = 0.15


def bicycle_model(track="LMS_Track6.txt", x0 = np.array([-2, 0, 0, 0.0, 0, 0])):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    # load track parameters
    [s0, xref, yref, psiref, kapparef] = getTrack(track)
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    xref = np.append(xref, xref[1:length])
    yref = np.append(yref, yref[1:length])
    psiref = np.append(psiref, psiref[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 301 : length - 2]], s0)
    # print(-s0[length - 2]+ s0[length - 151])
    kapparef = np.append(kapparef[length - 300 : length - 1], kapparef)
    xref = np.append(xref[length - 300 : length - 1], xref)
    yref = np.append(yref[length - 300 : length - 1], yref)
    psiref = np.append(psiref[length - 300 : length - 1], psiref)

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    xref_s = interpolant(f"x_ref", "bspline", [s0], xref)
    yref_s = interpolant(f"y_ref", "bspline", [s0], yref)
    psiref_s = interpolant(f"psi_ref", "bspline", [s0], psiref)

    ## Race car parameters
    m = 0.043
    C1 = 0.5
    C2 = 15.5
    Cm1 = 0.28
    Cm2 = 0.05
    Cr0 = 0.011
    Cr2 = 0.006

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    v = MX.sym("v")
    D = MX.sym("D")
    delta = MX.sym("delta")
    x = vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = MX.sym("derD")
    derDelta = MX.sym("derDelta")
    u = vertcat(derD, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    # algebraic variables
    z = vertcat([])

    # obstacles parameters
    obb_x = MX.sym("obb_x")
    obb_y = MX.sym("obb_y")
    obb_psi = MX.sym("obb_psi")
    obb_v = MX.sym("obb_v")
    obb_width = MX.sym("obb_width")
    obb_length = MX.sym("obb_length")
    obb_a = MX.sym("obb_a") # demi axe of the uncertainty ellipse
    obb_b = MX.sym("obb_b")
    obb_theta = MX.sym("obbtheta")

    obb1_x = MX.sym("obb1_x")
    obb1_y = MX.sym("obb1_y")
    obb1_psi = MX.sym("obb1_psi")
    obb1_v = MX.sym("obb1_v")
    obb1_width = MX.sym("obb1_width")
    obb1_length = MX.sym("obb1_length")
    obb1_a = MX.sym("obb1_a")
    obb1_b = MX.sym("obb1_b")
    obb1_theta = MX.sym("obb1_theta")

    obb2_x = MX.sym("obb2_x")
    obb2_y = MX.sym("obb2_y")
    obb2_psi = MX.sym("obb2_psi")
    obb2_v = MX.sym("obb2_v")
    obb2_width = MX.sym("obb2_width")
    obb2_length = MX.sym("obb2_length")
    obb2_a = MX.sym("obb2_a")
    obb2_b = MX.sym("obb2_b")
    obb2_theta = MX.sym("obb2_ay")

    p = vertcat(obb_x, obb_y, obb_psi, obb_v, obb_width, obb_length, obb_a, obb_b, obb_theta,
                obb1_x, obb1_y, obb1_psi, obb1_v, obb1_width, obb1_length, obb1_a, obb1_b, obb1_theta,
                obb2_x, obb2_y, obb2_psi, obb2_v, obb2_width, obb2_length, obb2_a, obb2_b, obb2_theta)

    # dynamics
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(5 * v)
    sdota = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    f_expl = vertcat(
        sdota,
        v * sin(alpha + C1 * delta),
        v * C2 * delta - kapparef_s(s) * sdota,
        Fxd / m * cos(C1 * delta),
        derD,
        derDelta,
    )

    # constraint on forces
    a_lat = C2 * v * v * delta + Fxd * sin(C1 * delta) / m
    a_long = Fxd / m

    # Model bounds
    track_width = 0.3
    r = 1/LENGTH * (WIDTH**2 + LENGTH**2)/4
    model.n_min = 0.0 # right border of the track [m]
    model.n_max = track_width + r  # middle of the opposite lane [m]

    # state bounds
    model.throttle_min = -1.0
    model.throttle_max = 1.0

    model.delta_min = -0.40  # minimum steering angle [rad]
    model.delta_max = 0.40  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -2.0  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 2.0  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -5  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 5 # 10.0  # maximum throttle change rate

    # nonlinear constraint
    constraint.alat_min = -4.5  # maximum lateral force [m/s^2]
    constraint.alat_max = 4.5  # maximum lateral force [m/s^1]

    constraint.along_min = -2  # maximum lateral force [m/s^2]
    constraint.along_max = 2  # maximum lateral force [m/s^2]

    model.v_max = 0.25 # maximum admissible velocity [m/s]
    model.v_lim = 0.20 # velocity limit [m/s]

    # Define initial conditions
    model.x0 = x0

    
    # define obstacles constraints
    # Function to get catesian pose from Frenet coordinates
    x_c = xref_s(s) - n * sin(psiref_s(s))
    y_c = yref_s(s) + n * cos(psiref_s(s))
    psi_c = psiref_s(s) + alpha
    constraint.pose = Function("pose", [x], [x_c, y_c, psi_c])
    constraint.obb_pose = Function("obb_pose", [p], [obb_x, obb_y, obb_psi])
    constraint.obb1_pose = Function("obb1_pose", [p], [obb1_x, obb1_y, obb1_psi])
    constraint.obb2_pose = Function("obb2_pose", [p], [obb2_x, obb2_y, obb2_psi])
    

    # construct matrice 3*3 containing distances between covering circles between the car and the obstacle
      # radius of the covering circle /!\ need to be scalar to avoid broadcasting issues
    centers_ego = vertcat(horzcat(x_c - r * cos(psi_c), y_c - r * sin(psi_c)),
                            horzcat(x_c, y_c),
                            horzcat(x_c + r * cos(psi_c), y_c + r * sin(psi_c)))
    centers_obb = np.array([
        [[obb_x - r * cos(obb_psi), obb_y - r * sin(obb_psi)]    , [obb_x, obb_y]  , [obb_x + r * cos(obb_psi), obb_y + r * sin(obb_psi)]],
        [[obb1_x - r * cos(obb1_psi), obb1_y - r * sin(obb1_psi)], [obb1_x, obb1_y], [obb1_x + r * cos(obb1_psi), obb1_y + r * sin(obb1_psi)]],
        [[obb2_x - r * cos(obb2_psi), obb2_y - r * sin(obb2_psi)], [obb2_x, obb2_y], [obb2_x + r * cos(obb2_psi), obb2_y + r * sin(obb2_psi)]]
                            ])
    
    # Ellipses parameters
    constraint.dist_min = 0.0  
    α_0 = obb_a + 2*r + r #minimum distance between covering circles centers including ellipse axes
    β_0 = obb_b + 2*r + r
    α1 = obb1_a + 2*r + r
    β1 = obb1_b + 2*r + r
    α2 = obb2_a + 2*r + r
    β2 = obb2_b + 2*r + r

    α = vertcat(α_0, α1, α2)
    β = vertcat(β_0, β1, β2)
    theta = vertcat(obb_theta, obb1_theta, obb2_theta)

    dist_matrix = 100*MX.ones(centers_obb.shape[0]*centers_obb.shape[1]*centers_ego.shape[0])
    for k in range(centers_obb.shape[0]): # loop over obstacles
        for j in range(centers_obb.shape[1]): # loop over covering circles
            for l in range(centers_ego.shape[0]): # loop over covering circles of the ego car
                x_l = centers_ego[l, 0]
                y_l = centers_ego[l, 1]
                dist = (((x_l - centers_obb[k,j,0]) * cos(theta[k]) + (y_l - centers_obb[k,j,1]) * sin(theta[k]) ) /α[k])**2 +\
                       (((x_l - centers_obb[k,j,0]) * sin(theta[k]) - (y_l - centers_obb[k,j,1]) * cos(theta[k]) ) /β[k])**2 - 1
                dist_matrix[k*centers_obb.shape[1]*centers_ego.shape[0] + j*centers_ego.shape[0] + l] = dist
        
    # print("dist_matrix: {}".format(dist_matrix))
    # print("dist_matrix: {}".format(dist_matrix.shape))
    dist = sqrt((x_c - obb_x) ** 2 + (y_c - obb_y) ** 2)
    constraint.dist = Function("dist", [x, p], [dist_matrix])
    
    

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, a_lat, n, D, delta, v, dist_matrix)

    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cm1 = Cm1
    params.Cm2 = Cm2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    return model, constraint