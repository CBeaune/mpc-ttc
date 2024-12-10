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
from time2spatial import transformProj2Orig,transformOrig2Proj


def bicycle_model(track="LMS_Track.txt"):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()
    distance = 0.15

    model_name = "Spatialbicycle_model"

    # load track parameters
    [s0, xref, yref, psiref, kapparef] = getTrack(track)
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    # s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    # kapparef = np.append(kapparef, kapparef[1:length])
    # s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    # kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

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

    # parameters
    # parameters
    s_obst = MX.sym('x_obst')
    n_obst = MX.sym('y_obst')
    alpha_obst = MX.sym('psi_obst')
    v_obst = MX.sym('dx_obst')
    D_obst = MX.sym('dy_obst')
    delta_obst = MX.sym('delta_obst')
    p = vertcat(s_obst, n_obst, alpha_obst, v_obst, D_obst, delta_obst)

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

    # constraints on obstacle avoidance
    # get cartesian coordinates of the car
    # [x_c, y_c, psi_c, _] = transformProj2Orig(s, n, alpha, v,track)
    x_c = xref_s(s) - n * sin(psiref_s(s))
    y_c = yref_s(s) + n * cos(psiref_s(s))
    psi_c = psiref_s(s) + alpha
    constraint.pose_c = Function("pose_c", [x], [vertcat(x_c, y_c, psi_c)])
    
    # get cartesian coordinates of the obstacle
    # [x_obst, y_obst, psi_obst, _] = transformProj2Orig(s_obst, n_obst, alpha_obst, v_obst,track)
    x_obst = xref_s(s_obst) - n_obst * sin(psiref_s(s_obst))
    y_obst = yref_s(s_obst) + n_obst * cos(psiref_s(s_obst))
    psi_obst = psiref_s(s_obst) + alpha_obst
    constraint.pose_obst = Function("pose_obst", [x,p], [vertcat(x_obst, y_obst, psi_obst)])

    # obstacle constraints
    dist = sqrt((x_c - x_obst) ** 2 + (y_c - y_obst) ** 2)
    constraint.dist = Function("dist", [x, p], [dist])
    constraint.dist_min = 2*np.sqrt((0.178/2)**2 + (0.132/2)**2/4) # minimum distance to obstacle

    # Time to collision constraint
    d_t0 = dist
    vxr = (v * cos(psi_c) - v_obst * cos(psi_obst))
    vyr = (v * sin(psi_c) - v_obst * sin(psi_obst))
    d_dot_t0 = ((x_c - x_obst) * vxr + (y_c - y_obst) * vyr)/d_t0
    ttc = if_else(d_dot_t0 < 0, (d_t0 - 0.178)/fabs(d_dot_t0), 20)
    constraint.ttc = Function("ttc", [x, p], [ttc])
    constraint.ttc_min = 0.5  # minimum time to collision

    # Model bounds
    model.n_min = 0.0  # width of the track [m]
    model.n_max = distance*2  # width of the track [m]

    # state bound
    model.throttle_min = -1.0
    model.throttle_max = 1.0

    model.delta_min = -1  # minimum steering angle [rad]
    model.delta_max = 1  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -3  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 3  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -10  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 10  # 10.0  # maximum throttle change rate

    # nonlinear constraint
    constraint.alat_min = -4  # maximum lateral force [m/s^2]
    constraint.alat_max = 4  # maximum lateral force [m/s^1]

    constraint.along_min = -4  # maximum lateral force [m/s^2]
    constraint.along_max = 4  # maximum lateral force [m/s^2]

    # Define initial conditions
    model.x0 = np.array([-1.85, 0, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, a_lat, n, D, delta, dist ) #,ttc ,

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
