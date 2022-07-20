from IPython.core.debugger import set_trace

from src.utilities.DDE_Solver import ddeinttf
import src.utilities.findiff.findiff_general as fdgen
import src.kdv_burgers_eqn_case.kdv_burgers_eqn as burg

import numpy as np
import scipy as spy
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

tf.keras.backend.set_floatx('float32')

## Define some useful classes

#### Function to solve the burgers equation
def burger_solve(args, u0 = None, t = None, alg_name = 'dopri5', nsteps = -1):
    grid_obj = fdgen.grid(args)

    if t is None: t = tf.linspace(args.T_start, args.T, args.nt)
        
    if u0 is None: u0 = initial_cond(grid_obj.x_grid_real, args)

    deriv_obj = fdgen.deriv(args, grid_obj)

    sol_obj = burg.burg_rhs(args = args, deriv_obj = deriv_obj, grid_obj = grid_obj)
    
    u = ddeinttf(sol_obj, u0, t, alg_name = alg_name, nsteps = nsteps)
    
    return u
        

# Function to help with interpolation        
def interp_high_res_to_low_res(u_high_res, x_high_res, x_low_res, t):

    f = spy.interpolate.interp2d(x_high_res, t, tf.squeeze(u_high_res, axis=1).numpy(), kind='cubic')

    u_interp = tf.expand_dims(tf.convert_to_tensor(f(x_low_res, t), dtype=tf.float32), axis=1)

    return u_interp
    
### Define a custom plotting function
class custom_plot:

    def __init__(self, true_y, y_no_nn, x, t, figsave_dir, args, restart = 0):
        self.true_y = true_y
        self.y_no_nn = y_no_nn
        self.t = t
        self.figsave_dir = figsave_dir
        self.args = args
        self.X, self.T = np.meshgrid(x.numpy(), t.numpy())
        self.x = x
        self.restart = restart

    def plot_3d(self, *pred_y, epoch = 0):
        fig = plt.figure(figsize=(24, 4), facecolor='white')
        ax_u_lowres = fig.add_subplot(141, projection='3d')
        ax_u_diff = fig.add_subplot(142, projection='3d')
        ax_u_nn = fig.add_subplot(143, projection='3d')
        ax_u_diff_nn = fig.add_subplot(144, projection='3d')

        ax_u_lowres.cla()
        ax_u_lowres.set_title('AD Eqn Low-Res Solution')
        ax_u_lowres.set_xlabel('x')
        ax_u_lowres.set_ylabel('t')
        ax_u_lowres.plot_surface(self.X, self.T, tf.squeeze(self.y_no_nn, axis=1).numpy(), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax_u_lowres.set_xlim(self.x[0], self.x[-1])
        ax_u_lowres.set_ylim(self.t[0], self.t[-1])

        ax_u_diff.cla()
        ax_u_diff.set_title('AD Eqn Difference without NN')
        ax_u_diff.set_xlabel('x')
        ax_u_diff.set_ylabel('t')
        ax_u_diff.plot_surface(self.X, self.T, tf.squeeze(self.true_y - self.y_no_nn, axis=1).numpy(), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax_u_diff.set_xlim(self.x[0], self.x[-1])
        ax_u_diff.set_ylim(self.t[0], self.t[-1])

        if epoch >= 0 or self.restart == 1 :
            ax_u_nn.cla()
            ax_u_nn.set_title('AD Eqn with NN Low-Res Solution')
            ax_u_nn.set_xlabel('x')
            ax_u_nn.set_ylabel('t')
            ax_u_nn.plot_surface(self.X, self.T, tf.squeeze(pred_y[0][:, :, 0:self.args.state_dim], axis=1).numpy(), cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax_u_nn.set_xlim(self.x[0], self.x[-1])
            ax_u_nn.set_ylim(self.t[0], self.t[-1])

            ax_u_diff_nn.cla()
            ax_u_diff_nn.set_title('AD Eqn Difference with NN')
            ax_u_diff_nn.set_xlabel('x')
            ax_u_diff_nn.set_ylabel('t')
            ax_u_diff_nn.plot_surface(self.X, self.T, tf.squeeze(self.true_y - pred_y[0][:, :, 0:self.args.state_dim], axis=1).numpy(), cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax_u_diff_nn.set_xlim(self.x[0], self.x[-1])
            ax_u_diff_nn.set_ylim(self.t[0], self.t[-1])

        plt.show() 

        if epoch != 0: 
            fig.savefig(os.path.join(self.figsave_dir, 'img'+str(epoch)))
            
    def plot(self, *pred_y, epoch = 0):
        fig = plt.figure(figsize=(14, 10), facecolor='white')
        ax_u_lowres = fig.add_subplot(221)
        ax_u_diff = fig.add_subplot(222)
        ax_u_nn = fig.add_subplot(223)
        ax_u_diff_nn = fig.add_subplot(224)
        
        ax_u_lowres.cla()
        ax_u_lowres.set_title('Low-Res Solution', fontsize=14)
        ax_u_lowres.set_xlabel('x', fontsize=14)
        ax_u_lowres.set_ylabel('t', fontsize=14)
        plot = ax_u_lowres.contourf(self.X, self.T, tf.squeeze(self.y_no_nn, axis=1).numpy(), cmap=cm.coolwarm,
                           antialiased=False, levels=np.linspace(0, 0.5, 20), extend='min')
        ax_u_lowres.set_xlim(self.x[0], self.x[-1])
        ax_u_lowres.set_ylim(self.t[0], self.t[-1])
        plt.colorbar(plot, ax=ax_u_lowres, shrink=0.5, aspect=10)

        ax_u_diff.cla()
        ax_u_diff.set_title('|Difference without NN|', fontsize=14)
        ax_u_diff.set_xlabel('x', fontsize=14)
        ax_u_diff.set_ylabel('t', fontsize=14)
        plot = ax_u_diff.contourf(self.X, self.T, tf.abs(tf.squeeze(self.true_y - self.y_no_nn, axis=1)).numpy(), cmap=plt.get_cmap('coolwarm'),
                            antialiased=False, levels=np.linspace(0, 0.25, 20))
        ax_u_diff.set_xlim(self.x[0], self.x[-1])
        ax_u_diff.set_ylim(self.t[0], self.t[-1])
        plt.colorbar(plot, ax=ax_u_diff, shrink=0.5, aspect=10)
        
        if epoch >= 0 or self.restart == 1 :
            ax_u_nn.cla()
            ax_u_nn.set_title('Low-Res with nCM Solution', fontsize=14)
            ax_u_nn.set_xlabel('x', fontsize=14)
            ax_u_nn.set_ylabel('t', fontsize=14)
            plot = ax_u_nn.contourf(self.X, self.T, tf.squeeze(pred_y[0][:, :, 0:self.args.state_dim], axis=1).numpy(), cmap=cm.coolwarm,
                               antialiased=False, levels=np.linspace(0, 0.5, 20), extend='min')
            ax_u_nn.set_xlim(self.x[0], self.x[-1])
            ax_u_nn.set_ylim(self.t[0], self.t[-1])
            plt.colorbar(plot, ax=ax_u_nn, shrink=0.5, aspect=10)

            ax_u_diff_nn.cla()
            ax_u_diff_nn.set_title('|Difference|', fontsize=14)
            ax_u_diff_nn.set_xlabel('x', fontsize=14)
            ax_u_diff_nn.set_ylabel('t', fontsize=14)
            plot = ax_u_diff_nn.contourf(self.X, self.T, tf.abs(tf.squeeze(self.true_y - pred_y[0][:, :, 0:self.args.state_dim], axis=1)).numpy(), cmap=plt.get_cmap('coolwarm'),
                                antialiased=False, levels=np.linspace(0, 0.25, 20))
            ax_u_diff_nn.set_xlim(self.x[0], self.x[-1])
            ax_u_diff_nn.set_ylim(self.t[0], self.t[-1])
            plt.colorbar(plot, ax=ax_u_diff_nn, shrink=0.5, aspect=10)

        plt.show()
        
        if epoch != 0: 
            fig.savefig(os.path.join(self.figsave_dir, 'img'+str(epoch)))
            
            
### Initial Conditions
class initial_cond:

    def __init__(self, x, app):
        self.x = x
        self.app = app

    def __call__(self, t):
#         u1 = 2. * self.app.eta_1**2 / np.cosh(self.app.eta_1 * (self.x - self.app.x_1))**2
#         u2 = 2. * self.app.eta_2**2 / np.cosh(self.app.eta_2 * (self.x - self.app.x_2))**2
        
#         u0 = u1 + u2
        
        theta_1 = self.app.eta_1 * (self.x - self.app.x_1)
        theta_2 = self.app.eta_2 * (self.x - self.app.x_2)
        
        u = (8. * (self.app.eta_1**2 - self.app.eta_2**2) * (self.app.eta_1**2 * np.cosh(theta_2)**2 + self.app.eta_2**2 * np.sinh(theta_1)**2)) \
            / ((self.app.eta_1 - self.app.eta_2) * np.cosh(theta_1 + theta_2) + (self.app.eta_1 + self.app.eta_2) * np.cosh(theta_1 - theta_2))**2
        
        return tf.convert_to_tensor([u], dtype=tf.float32)
    
