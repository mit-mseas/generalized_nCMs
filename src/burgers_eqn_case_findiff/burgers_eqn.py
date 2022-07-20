from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf

from src.utilities.findiff.findiff_general import solver_arguments
from src.solvers.findiff.findiff_solver import fd_solve


########################################################
"""
Common along with testcase specific arguments
"""
class burg_args(solver_arguments):
    
    def __init__(self, T, nt, x_left = -1, x_right = 1, Nx = 10, T_start = 0., Re = 250, dleft_bv_dt = 0, dright_bv_dt = 0, dbc_l = 1, nbc_l = 0, dbc_r = 1, nbc_r = 0, max_deriv = 2, acc = 2, acc_advec = 1):
        
        solver_arguments.__init__(self, T = T, nt = nt, x_left = x_left, x_right = x_right, Nx = Nx, T_start = T_start)
        
        self.Re = Re
        self.nu = 1./self.Re
        self.t0 = np.exp(self.Re/8, dtype = np.float64)
        
        self.state_dim = Nx
        
        self.dleft_bv_dt = dleft_bv_dt
        self.dright_bv_dt = dright_bv_dt
        self.dbc_l = dbc_l
        self.nbc_l = nbc_l
        self.dbc_r = dbc_r
        self.nbc_r = nbc_r
        self.max_deriv = max_deriv
        self.acc = acc
        self.acc_advec = acc_advec
        
        

########################################################
"""
Class containign the RHS of the PDE

Everything assumed to be in numpy
"""        
class burg_rhs(fd_solve):
    
    def __init__(self, args, deriv_obj, grid_obj):
        fd_solve.__init__(self, args = args, deriv_obj = deriv_obj, grid_obj = grid_obj)
        
        self.full_vander_dx = self.deriv_obj.vander(self.grid.x_grid, m=1, acc=self.args.acc)
        self.full_vander_dxx = self.deriv_obj.vander(self.grid.x_grid, m=2, acc=self.args.acc)
        
#         self.vander_dx = self.full_vander_dx[1:-(self.args.max_deriv - 1), ]
#         self.vander_dxx = self.full_vander_dxx[1:-(self.args.max_deriv - 1), ]
        
        self.vander_dx = self.full_vander_dx[self.args.max_deriv - 1:-(self.args.max_deriv - 1), ]
        self.vander_dxx = self.full_vander_dxx[self.args.max_deriv - 1:-(self.args.max_deriv - 1), ]
        
        self.full_vander_dx_upwind = self.deriv_obj.upwind(self.grid.x_grid, acc = min(self.args.acc, 4))
        
#         self.vander_dx_upwind = self.full_vander_dx_upwind[1:-(self.args.max_deriv - 1), ]
        self.vander_dx_upwind = self.full_vander_dx_upwind[self.args.max_deriv - 1:-(self.args.max_deriv - 1), ]
    
        self.vander_dx_upwind_advec = self.deriv_obj.upwind(self.grid.x_grid, acc = self.args.acc_advec)[self.args.max_deriv - 1:-(self.args.max_deriv - 1), ]
        
        self.full_vander = self.get_vander_matrices(self.grid.x_grid, max_deriv = args.max_deriv)
        
    def get_vander_matrices(self, x, max_deriv = None):
        
        if max_deriv == None: max_deriv = self.args.max_deriv
        
        A = []
        for i in range(max_deriv+1):
            A.append(self.deriv_obj.vander(x, m=i, acc=self.args.acc))
            
        return A
    
    def rhs_int(self, t, u_t, u_t_int):
        
        u_x_int = np.einsum('ab, cb -> ca', self.vander_dx_upwind_advec, u_t)
        u_xx_int = np.einsum('ab, cb -> ca', self.vander_dxx, u_t)
        
        du_dt_int = -np.einsum('ab, ab -> ab', u_t_int, u_x_int) + self.args.nu * u_xx_int
        
        return du_dt_int
    
    def jac(self, u_stack, t, t_start = np.array([0.])):
        
        rhs_jac = [tf.linalg.diag(-u_stack[:, :, 1])]
        rhs_jac.append(tf.linalg.diag(-u_stack[:, :, 0]))
        rhs_jac.append(tf.linalg.diag(self.args.nu * np.ones(u_stack[:, :, 0].shape)))
        
        for i in range(3, self.args.max_deriv + 1):
            rhs_jac.append(tf.zeros(rhs_jac[0].shape))
        
        rhs_jac = np.stack(rhs_jac, axis = -1)
        
        return rhs_jac
        