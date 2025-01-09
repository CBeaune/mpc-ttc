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

import casadi as ca

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from bicycle_model import bicycle_model
import scipy.linalg
import numpy as np


def acados_settings(Tf, N, track_file, x0):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint = bicycle_model(track_file, x0)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    n_params = model.p.size()[0]
    ocp.dims.np = n_params
    

    # define constraint
    model_ac.con_h_expr = constraint.expr

    # dimensions
    nx = model.x.rows() 
    nx_obb = 9 # number of states per obstacle 
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx 

    n_obb = 3 # number of obstacles
    n_dist_circle = 3 # number of distance to circle constraints per obstacle
    n_dist_tot = 3 * n_obb * n_dist_circle
    ocp.parameter_values = np.zeros((n_obb*nx_obb,))

    nsbx = 1
    nh = constraint.expr.shape[0]
    hard_constraints = n_dist_tot + 1 
    nsh = nh - hard_constraints
    ns = nsh + nsbx

    # discretization
    ocp.solver_options.N_horizon = N
    # set cost
    Q = np.diag([ 1e-1, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3 ])

    R = np.eye(nu)
    R[0, 0] = 1e-3
    R[1, 1] = 5e-3

    Qe = np.diag([ 5e0, 1e1, 1e-8, 1e-8, 5e-3, 2e-3 ])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # vref = 0.25
    # Tf = 5.0
    # W_norm = np.diag([1/(vref*Tf), 2/0.25, 1/np.pi/4, 1/vref, 1, 1])
    # normalized_x = ca.mtimes(W_norm, model.x)
    ocp.model.cost_y_expr = ca.vertcat(model.x,
                                        model.u)
    ocp.model.cost_y_expr_e = model.x
    unscale = N / Tf

    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[6, 0] = 1.0
    Vu[7, 1] = 1.0
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zl = 1 * np.ones((ns,))
    ocp.cost.Zu = 1 * np.ones((ns,))

    # set intial references
    ocp.cost.yref = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0])

    # setting constraints
    ocp.constraints.lbx = np.array([model.n_min, 0.0,])
    ocp.constraints.ubx = np.array([model.n_max, model.v_max,])
    ocp.constraints.idxbx = np.array([1, 3])
    ocp.constraints.lbu = np.array([model.dthrottle_min, model.ddelta_min])
    ocp.constraints.ubu = np.array([model.dthrottle_max, model.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lsbx = np.zeros([nsbx])
    ocp.constraints.usbx = np.zeros([nsbx])
    ocp.constraints.idxsbx = np.array(range(nsbx))

    ocp.constraints.lh = np.zeros(nh)
    ocp.constraints.lh[:5] = [
            constraint.along_min,
            constraint.alat_min,
            model.n_min,
            model.throttle_min,
            model.delta_min,
            ]
    
    
    # ocp.constraints.lh[6:] = constraint.dist_min
   
    ocp.constraints.uh = np.zeros(nh)
    ocp.constraints.uh[:5] = [
            constraint.along_max,
            constraint.alat_max,
            model.n_max,
            model.throttle_max,
            model.delta_max,
        ]
    for i in range(5, 5 + n_dist_tot):
        ocp.constraints.lh[i] = constraint.dist_min
        ocp.constraints.uh[i] = 1e15
    
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array([0, 1, 3, 4 ])

    # set intial condition
    # ocp.constraints.x0 = model.x0
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0 = model.x0
    ocp.constraints.ubx_0 = model.x0

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.regularize_method = 'PROJECT'
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.tol = 5e-1
    ocp.solver_options.timeout_max_time = 1e-3
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json", verbose=False)

    return constraint, model, acados_solver