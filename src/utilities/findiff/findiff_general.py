from IPython.core.debugger import set_trace

from math import *
import numpy as np
from findiff import FinDiff, coefficients


########################################################
"""
Some basic arguments needed for any testcase
"""
class solver_arguments():

    def __init__(self, T, nt, x_left = -1, x_right = 1, Nx = 10, T_start = 0.):
        
        self.T_start = T_start
        self.T = T
        self.nt = nt
        
        self.x_left = x_left
        self.x_right = x_right
        self.Nx = Nx
        

########################################################
"""
Class computing the Gauss-Lobatto collocation points and to-and-fro mapping between unit and real domains
"""
class grid():
    
    def __init__(self, args, x_grid = None):
        
        self.args = args
        
        if x_grid is None:
            self.x_grid = self.equal_spaced_points()
        else:
            self.x_grid = x_grid
            
        self.x_grid_real = self.x_grid
        
    def equal_spaced_points(self):
        
        x_n = np.linspace(self.args.x_left, self.args.x_right, self.args.Nx)
        
        return x_n
    
########################################################
"""
Class helping to compute Vandermod matrices corresponding to any derivative for Chebyshev polynomials
"""
class deriv():
    
    def __init__(self, args, grid_obj):
        
        self.args = args
        self.grid = grid_obj
    
    def vander(self, x_grid, m = 0, acc = 2, axis = 0):
        
        A = FinDiff(axis, x_grid, m, acc=acc).matrix(x_grid.shape).toarray()
        
        if m!= 0:
            coefs = coefficients(deriv=m, acc=acc)
            offset_c = coefs['center']['offsets']
            offset_f = coefs['forward']['offsets']
            offset_b = coefs['backward']['offsets']

            for i in range(1, int(len(offset_c)/2)):
                offset_left = [j - i for j in offset_f]
                offset_right = [j + i for j in offset_b]

                coefs_left = coefficients(deriv=m, offsets=offset_left)['coefficients'] * (1./(x_grid[1] - x_grid[0])**(m))
                coefs_right = coefficients(deriv=m, offsets=offset_right)['coefficients'] * (1./(x_grid[1] - x_grid[0])**(m))

                offset_left = [i + j for j in offset_left]
                for k in range(len(offset_f)):
                    A[i, i+offset_f[k]] = 0.

                A[i, offset_left] = coefs_left

                offset_right = [-i-1 + j for j in offset_right]
                for k in range(len(offset_b)):
                    A[-1-i, -i-1+offset_b[k]] = 0.

                A[-1-i, offset_right] = coefs_right
        
        return A
    
    def upwind(self, x_grid, acc = 1):
        
        A = np.eye(self.args.Nx)
        A[list(np.arange(1, self.args.Nx)), list(np.arange(0, self.args.Nx-1))] = -1
        A[0, 0] = -1
        A[0, 1] = 1
        
        if acc == 2 or acc == 4:
            A[1+1, 1+1] = 3./2.
            A[1+1, 0+1] = -4./2.
            A[1+1, 0] = 1./2.
            A[list(np.arange(1 + 2, self.args.Nx)), list(np.arange(1 + 2, self.args.Nx))] = 10./6.
            A[list(np.arange(1 + 2, self.args.Nx)), list(np.arange(0 + 2, self.args.Nx-1))] = -15./6.
            A[list(np.arange(1 + 2, self.args.Nx)), list(np.arange(-1 + 2, self.args.Nx - 1 - 1))] = 6./6.
            A[list(np.arange(1 + 2, self.args.Nx)), list(np.arange(-2 + 2, self.args.Nx - 1 - 2))] = -1./6.
#             A[list(np.arange(1 + 1, self.args.Nx)), list(np.arange(1 + 1, self.args.Nx))] = 3./2.
#             A[list(np.arange(1 + 1, self.args.Nx)), list(np.arange(0 + 1, self.args.Nx-1))] = -4./2.
#             A[list(np.arange(1 + 1, self.args.Nx)), list(np.arange(0, self.args.Nx - 1 - 1))] = 1./2.

            if acc == 4:
                A[list(np.arange(1 + 3, self.args.Nx)), list(np.arange(1 + 3, self.args.Nx))] = 25./12.
                A[list(np.arange(1 + 3, self.args.Nx)), list(np.arange(0 + 3, self.args.Nx-1))] = -48./12.
                A[list(np.arange(1 + 3, self.args.Nx)), list(np.arange(-1 + 3, self.args.Nx - 1 - 1))] = 36./12.
                A[list(np.arange(1 + 3, self.args.Nx)), list(np.arange(-2 + 3, self.args.Nx - 1 - 2))] = -16./12.
                A[list(np.arange(1 + 3, self.args.Nx)), list(np.arange(-3 + 3, self.args.Nx - 1 - 3))] = 3./12.
            
        A = A * (1./(x_grid[1] - x_grid[0]))
        
        return A                              