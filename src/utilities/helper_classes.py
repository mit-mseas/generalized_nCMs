from shutil import move
import pickle
import sys
import os

import numpy as np
from math import *
import tensorflow as tf
import scipy as sc

from findiff import FinDiff, coefficients

tf.keras.backend.set_floatx('float32')

#### Create save directories and copy the main script

class save_dir():
    
    def __init__(self, args, basedir, testcase_dir, save_user_inputs=True):
        self.args = args
        self.basedir = basedir
        self.testcase_dir = testcase_dir
        self.save_user_inputs = save_user_inputs
        
        os.chdir(self.basedir)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        self.checkpoint_dir = os.path.join(self.args.model_dir, "ckpt")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "cp-{epoch:04d}.ckpt")
        if not os.path.exists(self.checkpoint_dir):
          os.makedirs(self.checkpoint_dir)

        self.figsave_dir = os.path.join(self.args.model_dir, "img")
        if not os.path.exists(self.figsave_dir):
            os.makedirs(self.figsave_dir)

        
    def __call__(self, script_name):
        os.chdir(self.basedir)

        os.system("jupyter nbconvert --to python " + os.path.join(self.testcase_dir, script_name + ".ipynb"))
        move(os.path.join(self.testcase_dir, script_name + ".py"), os.path.join(self.args.model_dir, "orig_run_file.py"))

        if self.save_user_inputs:
            with open(os.path.join(self.args.model_dir, 'args.pkl'), 'wb') as output:
                pickle.dump(self.args, output, pickle.HIGHEST_PROTOCOL)
                
                
def integrate_simp_cumsum(y, x, axis = -1):
    
        base_str = 'cdefghijklmnopqrstuvwxyz'
        dx = x[1:] - x[0:-1]
        
        y_avg = 0.5 * (np.take(y, list(range(0, y.shape[axis]-1)), axis=axis) + np.take(y, list(range(1, y.shape[axis])), axis=axis))
        
        var_shape_len = len(np.shape(y_avg))
        shape_str = base_str[:var_shape_len]

        y_int = np.einsum(shape_str + ',' + shape_str[axis] + \
                               '->' + shape_str, y_avg, dx)
        
        return np.cumsum(y_int, axis=axis)
    
def vander(x_grid, m = 0, acc = 2, axis = 0):

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

def diagonalize(M):
    expanded = np.zeros(M.shape + M.shape[-1:], dtype=M.dtype)

    diagonals = np.diagonal(expanded, axis1=-2, axis2=-1)
    diagonals.setflags(write=True)

    diagonals[:] = M
    
    return expanded
