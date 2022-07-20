from src.utilities.DDE_Solver import ddeinttf

import quadpy

import time
import timeit
import sys
import os
from tqdm import tqdm
from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate
import pickle
import types

tf.keras.backend.set_floatx('float32')
tf.config.run_functions_eagerly(False)

#### Class for user-defined variables
class ncm_arguments:
    def __init__(self, data_size = 1000, batch_time = 12, batch_time_skip = 2, batch_size = 5, epochs = 500, learning_rate = 0.05, decay_rate = 0.95, test_freq = 1, plot_freq = 2, 
                 tau_max = 1.1, tau = 0.5, train_ens_size = 1,
                 model_dir = 'DDE_runs/model_dir_test', restart = 0, val_percentage = 0.2, isplot = True, is_tstart_zero = True):
        self.data_size = data_size # Time-steps for which the ODE is solved for final loss computation and plotting
        self.batch_time = batch_time # Number of time-steps to skip for the batch final time. Total time interval will be dependent on the value of data_size as well
        self.batch_time_skip = batch_time_skip # Frequency of data points in the loss function
        self.batch_size = batch_size # Batch size 
        self.niters = np.ceil(self.data_size/(self.batch_time*self.batch_size) + 1).astype(int) # Number of iterations per epoch
        self.epochs = epochs # Number of epochs
        self.learning_rate = learning_rate # Initial learning rate
        self.decay_rate = decay_rate # parameter for exponential decay of learning rate
        self.test_freq = test_freq # Plotting frequency
        self.plot_freq = plot_freq
        self.isplot = isplot
        self.is_tstart_zero = is_tstart_zero
        
        self.tau_max = tau_max # The maximum value of time-delay present in the model
        
        self.tau = tau
        
        self.model_dir = model_dir
        self.basedir = os.getcwd()
        
        self.restart = restart
        
        self.val_percentage = val_percentage # Percentage of Length of time of training to be used for validation 

        self.base_str = 'cdefghijklmnopqrstuvwxyz' # A helper string
        
## To create batches
class create_batch:
    def __init__(self, true_y, true_y0, t, args):
        self.t = t
        self.true_y = true_y
        self.true_y0 = true_y0
        self.args = args

    def get_batch(self, flag = 0):

        s = np.random.choice(
            np.arange(self.args.data_size - self.args.batch_time,
                    dtype=np.int64), self.args.batch_size,
                    replace=False)
        
        dt = self.t[1] - self.t[0]
        n_back = np.ceil(np.abs(self.args.tau_max/dt.numpy())).astype(int) 
        temp_y = self.true_y.numpy()
        
        batch_size = self.args.batch_size
        batch_time = self.args.batch_time

        if flag != 0:
            s = np.asarray([i for i in range(batch_size)])

        batch_y0 = [[] for _ in np.arange(batch_size)]
        
        for i in np.arange(batch_size):
            t_back = np.linspace(self.t[s[i]] - ((n_back-1)*dt.numpy()), self.t[s[i]], n_back)
            n_back_neg = t_back[t_back<np.min(self.t.numpy())].shape[0]
            for k in np.arange(n_back_neg):
                batch_y0[i].append(tf.expand_dims(self.true_y0(t_back[k]), axis=0))
            batch_y0[i].append(self.true_y[s[i] - (n_back - n_back_neg)+1:s[i]+1, ])
            batch_y0[i] = tf.expand_dims(tf.concat(batch_y0[i], axis=0), axis=1)

        batch_y0 = tf.concat(batch_y0, axis=1)
            
        batch_t0 = tf.linspace(self.t[0] - (n_back*dt.numpy()), self.t[0], n_back)
        batch_t = self.t[:batch_time+1:self.args.batch_time_skip]  # (T)
        batch_y = tf.stack([temp_y[s + i] for i in range(0, batch_time+1, self.args.batch_time_skip)], axis=0)  # (T, M, D)
        batch_t_start = self.t.numpy()[s]
        
        return tf.squeeze(batch_y0, axis=-2), batch_t0, batch_t, tf.squeeze(batch_y, axis=-2), batch_t_start
       
    
#### Helper class to create interpolation functions
class create_interpolator():

    def __init__(self, batch_y0, batch_t0):
        self.batch_size = batch_y0.numpy().shape
        self.interpolator = scipy.interpolate.interp1d(
            batch_t0.numpy(),  # X
            np.reshape(batch_y0.numpy(), [self.batch_size[0], -1]),  # Y
            kind="linear", axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )

    def __call__(self, t):
        return  tf.convert_to_tensor(np.reshape(self.interpolator(t), self.batch_size[1:]), tf.float32)

class create_dirac_interpolator(create_interpolator): # dirac at end of the time interval

    def __init__(self, batch_y0, batch_t0, add_jump_val, jump_t):
        create_interpolator.__init__(self, batch_y0, batch_t0)
        self.add_jump_val = add_jump_val
        self.jump_t = jump_t

    def __call__(self, t):
        return self.add_jump_val + create_interpolator.__call__(self, t) if t==self.jump_t else create_interpolator.__call__(self, t)

class add_interpolator():

    def __init__(self, ini_interp, t_first, t_last):
        self.add_interp = [ini_interp]
        self.t = np.array([t_first, t_last])
        self.flag = False if (t_last - t_first <=0) else True

    def add(self, interp, t_first, t_last):
        self.add_interp.append(interp)
        self.t[-1] = t_first
        self.t = np.append(self.t, np.array([t_last]))

    def __call__(self, t):
        idx = np.digitize(t, self.t, right=self.flag) - 1
        return self.add_interp[idx](t)
    
    
#### Class to construct initial conditions for coupled DDE form
class process_DistDDE_IC:

    def __init__(self, z, aux_model, t_lowerlim, t_upperlim):
        self.z = z
        self.g = aux_model
        self.t_ll = t_lowerlim
        self.t_ul = t_upperlim
        self.scheme = quadpy.c1.gauss_legendre(5)
        self.intgral_z = self.integrate_z()

    def convert_to_numpy(self, t):
        return np.stack([self.g(self.z(t[i]), flag_fwd=1).numpy() for i in range(len(t))], axis=-1)

    def integrate_z(self):
        val = self.scheme.integrate(self.convert_to_numpy, [self.t_ll, self.t_ul])
        return tf.convert_to_tensor(val, tf.float32)

    def __call__(self, t):
        if tf.rank(t) == 0: 
            return tf.concat([self.z(t), self.intgral_z], axis=-1)
        else : 
            return tf.stack([tf.concat([self.z(t), self.intgral_z], axis=-1) for t in t], axis=0)


#### Adjoint system
class ncm_adj_eqn:

    def __init__(self, model, jac = None):
        self.g = model.non_mark_nn_part
        self.f = model.mark_nn_part
        self.rom = model.rom_model
        self.args = model.args
        self.jac = jac
        self.model = model

    @tf.function
    def calc_jac_non_mark_nn(self, z_stack_deriv):
        with tf.GradientTape() as tape:
            tape.watch(z_stack_deriv)
            g_z_stack_deriv = self.g(z_stack_deriv)

        dg_dz_stack_deriv = tape.batch_jacobian(g_z_stack_deriv, z_stack_deriv)
        return dg_dz_stack_deriv
    
    @tf.function
    def calc_jac_mark_nn(self, z_stack_deriv):
        with tf.GradientTape() as tape:
            tape.watch(z_stack_deriv)
            f_z_stack_deriv = self.f(z_stack_deriv)

        df_dz_stack_deriv = tape.batch_jacobian(f_z_stack_deriv, z_stack_deriv)
        return df_dz_stack_deriv
    
    @tf.function
    def calc_rom_jac(self, input, t, t_start):
        with tf.GradientTape() as tape:
            tape.watch(input)
            rom_x = self.rom.rhs(input, t, t_start)

        drom_dx = tape.batch_jacobian(rom_x, input)
    
    def __call__(self, lam_mu, t, tau, zy, t_start):
        
        zy_t = zy(t)
        z_t = zy_t[:, :self.args.state_dim]
        
        dz_dx_stack_deriv = self.model.compute_derivatives(z_t, flag_fwd = 1)
        
        dg_dz_stack_deriv = self.calc_jac_non_mark_nn(dz_dx_stack_deriv) # Hopefully should be diagonnal
        dg_dz_stack_deriv = tf.transpose(tf.linalg.diag_part(tf.transpose(dg_dz_stack_deriv, perm = [0, 3, 1, 2])), perm = [0, 2, 1])
        
        lam_mu_t = lam_mu(t)
        lam_t = lam_mu_t[:, :self.args.state_dim]
        dlam_dx_stack_deriv = self.model.compute_derivatives(lam_t)
        
        dmu_dx_stack_deriv = self.model.compute_derivatives(lam_mu_t[:, self.args.state_dim:])
        dmu_tptau_dx_stack_deriv = self.model.compute_derivatives(lam_mu(t + tau)[:, self.args.state_dim:])
        
        dmu_dg_dz_stack_deriv = tf.einsum('abd, abd -> abd', tf.cast(dmu_dx_stack_deriv - dmu_tptau_dx_stack_deriv, tf.float64), tf.cast(dg_dz_stack_deriv, tf.float64))
        
        lam_rhs = tf.zeros(z_t.shape, tf.float64)
        for i in range(dmu_dg_dz_stack_deriv.shape[-1]):
            lam_rhs = lam_rhs + (-1)**(i+1) * dmu_dg_dz_stack_deriv[:, :, i]
        
        if self.jac is not None:
            drom_dz_stack_deriv = tf.convert_to_tensor(self.jac(dz_dx_stack_deriv.numpy(), t, t_start))
            
        else:
            drom_dz_stack_deriv = self.calc_rom_jac(dz_dx_stack_deriv, t, t_start)
           
        drom_dz_stack_deriv += self.calc_jac_mark_nn(dz_dx_stack_deriv)
        
        drom_dz_stack_deriv = tf.transpose(tf.linalg.diag_part(tf.transpose(drom_dz_stack_deriv, perm = [0, 3, 1, 2])), perm = [0, 2, 1])
        
        dlam_drom_dz_stack_deriv = tf.einsum('abd, abd -> abd', tf.cast(dlam_dx_stack_deriv, tf.float64), tf.cast(drom_dz_stack_deriv, tf.float64))
        
        for i in range(dlam_drom_dz_stack_deriv.shape[-1]):
            lam_rhs = lam_rhs + (-1)**(i+1) * dlam_drom_dz_stack_deriv[:, :, i]

        mu_rhs = - lam_t
        mu_tptau_rhs = - lam_mu(t + tau)[:, :self.args.state_dim]
        
        dmu_rhs_dx_stack_deriv = self.model.compute_derivatives(mu_rhs)
        dmu_tptau_rhs_dx_stack_deriv = self.model.compute_derivatives(mu_tptau_rhs)
        
#         lam_rhs_bnd = self.boundary(lam_rhs[:, self.args.max_deriv - 1:-1], dmu_rhs_dx_stack_deriv, dmu_tptau_rhs_dx_stack_deriv, drom_dz_stack_deriv, dg_dz_stack_deriv)

#         lam_rhs = tf.cast(tf.concat([lam_rhs_bnd[:, 0:self.args.max_deriv - 1], lam_rhs[:, self.args.max_deriv - 1:-1], lam_rhs_bnd[:, -1:]], axis=-1), dtype = tf.float32)
        
        lam_rhs_bnd = self.boundary(lam_rhs[:, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], dmu_rhs_dx_stack_deriv, dmu_tptau_rhs_dx_stack_deriv, drom_dz_stack_deriv, dg_dz_stack_deriv)

        lam_rhs = tf.cast(tf.concat([lam_rhs_bnd[:, 0:self.args.max_deriv - 1], lam_rhs[:, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], lam_rhs_bnd[:, self.args.max_deriv - 1:]], axis=-1), dtype = tf.float32)

        return tf.concat([lam_rhs, mu_rhs], axis=-1)
    
    def boundary(self, lam_rhs_int, dmu_rhs_dx_stack_deriv, dmu_tptau_rhs_dx_stack_deriv, drom_dz_stack_deriv, dg_dz_stack_deriv):
        # currently only supports dirichlet BCs
       
        size_x = drom_dz_stack_deriv.shape[1]
        
        def mat_entry(eqn_id, idx, bnd):
            
            A = 0
            
            for i in range(self.args.max_deriv - eqn_id):
                A += (-1)**(i+1) * self.rom.full_vander[i][bnd, idx] * drom_dz_stack_deriv[:, bnd, i + (eqn_id + 1)]
                
            return A
        
#         def mark_rhs_entry(eqn_id, bnd):
            
#             b = 0
            
#             for i in range(self.args.max_deriv - eqn_id):
#                 b += (-1)**(i) * tf.einsum('b, cb -> c', self.rom.full_vander[i][bnd, self.args.max_deriv - 1:-1], lam_rhs_int) * drom_dz_stack_deriv[:, bnd, i + (eqn_id + 1)]
                
#             return b
        
        def mark_rhs_entry(eqn_id, bnd):
            
            b = 0
            
            for i in range(self.args.max_deriv - eqn_id):
                b += (-1)**(i) * tf.einsum('b, cb -> c', self.rom.full_vander[i][bnd, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], lam_rhs_int) * drom_dz_stack_deriv[:, bnd, i + (eqn_id + 1)]
                
            return b
        
        def non_mark_rhs_entry(dmu_dx, dg_du, eqn_id, bnd):
            
            b = 0
            
            for i in range(self.args.max_deriv - eqn_id):
                b += (-1)**(i) * dmu_dx[:, bnd, i] * dg_du[:, bnd, i + (eqn_id + 1)]
                
            return b
        
        B = []
        b = []
        b_t = []
        b_tptau = []
        k = 0
        
#         bnd = 0
#         for i in range((self.args.max_deriv)):
#             B.append([])
#             b.append(mark_rhs_entry(i, bnd))
#             b_t.append(non_mark_rhs_entry(dmu_rhs_dx_stack_deriv, dg_dz_stack_deriv, i, bnd))
#             b_tptau.append(non_mark_rhs_entry(dmu_tptau_rhs_dx_stack_deriv, dg_dz_stack_deriv, i, bnd))

#             for j in range(self.args.max_deriv):
#                 if j < self.args.max_deriv - 1:
#                     B[k].append(mat_entry(i, j, bnd))
#                 else:
#                     B[k].append(mat_entry(i, -1, bnd))

#             B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
#             k += 1
            
#         bnd = -1
#         for i in range(2):
#             B.append([])
#             b.append(mark_rhs_entry(i, bnd))
#             b_t.append(non_mark_rhs_entry(dmu_rhs_dx_stack_deriv, dg_dz_stack_deriv, i, bnd))
#             b_tptau.append(non_mark_rhs_entry(dmu_tptau_rhs_dx_stack_deriv, dg_dz_stack_deriv, i, bnd))

#             for j in range(self.args.max_deriv):
#                 if j < self.args.max_deriv - 1:
#                     B[k].append(mat_entry(i, j, bnd))
#                 else:
#                     B[k].append(mat_entry(i, -1, bnd))

#             B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
#             k += 1
                
#         B_tmp = [B[0] * self.args.nbc_l + B[1] * self.args.dbc_l]
#         b_tmp = [(b[0] + b_t[0] - b_tptau[0]) * self.args.nbc_l + (b[1] + b_t[1] - b_tptau[1]) * self.args.dbc_l]

#         for i in range(2, self.args.max_deriv):
#             B_tmp.append(B[i])
#             b_tmp.append(b[i] + b_t[i] - b_tptau[i])

#         B_tmp.append(B[self.args.max_deriv] * self.args.nbc_r + B[self.args.max_deriv+1] * self.args.dbc_r)
#         b_tmp.append((b[self.args.max_deriv] + b_t[self.args.max_deriv] - b_tptau[self.args.max_deriv]) * self.args.nbc_r + (b[self.args.max_deriv+1] + b_t[self.args.max_deriv+1] - b_tptau[self.args.max_deriv+1]) * self.args.dbc_r)

        for bnd in [0, -1]:
            for i in range((self.args.max_deriv)):
                B.append([])
                b.append(mark_rhs_entry(i, bnd))
                b_t.append(non_mark_rhs_entry(dmu_rhs_dx_stack_deriv, dg_dz_stack_deriv, i, bnd))
                b_tptau.append(non_mark_rhs_entry(dmu_tptau_rhs_dx_stack_deriv, dg_dz_stack_deriv, i, bnd))

                for j in range(2 * (self.args.max_deriv - 1)):
                    if j < self.args.max_deriv - 1:
                        B[k].append(mat_entry(i, j, bnd))
                    else:
                        B[k].append(mat_entry(i, -(self.args.max_deriv - 1)+(j % (self.args.max_deriv - 1)), bnd))
                    
                B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
                k += 1
                
        B_tmp = [B[0] * self.args.nbc_l + B[1] * self.args.dbc_l]
        b_tmp = [(b[0] + b_t[0] - b_tptau[0]) * self.args.nbc_l + (b[1] + b_t[1] - b_tptau[1]) * self.args.dbc_l]

        for i in range(2, self.args.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i] + b_t[i] - b_tptau[i])

        B_tmp.append(B[self.args.max_deriv] * self.args.nbc_r + B[self.args.max_deriv+1] * self.args.dbc_r)
        b_tmp.append((b[self.args.max_deriv] + b_t[self.args.max_deriv] - b_tptau[self.args.max_deriv]) * self.args.nbc_r + (b[self.args.max_deriv+1] + b_t[self.args.max_deriv+1] - b_tptau[self.args.max_deriv+1]) * self.args.dbc_r)

        for i in range(self.args.max_deriv+2, 2*self.args.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i] + b_t[i] - b_tptau[i])
        
        B = tf.concat(B_tmp, axis=-2)
        b = tf.stack(b_tmp, axis=-1)
        
        lam_rhs_bnd = tf.einsum('cab, cb ->ca', tf.linalg.pinv(B), b)
        
        return lam_rhs_bnd


#### Class for validation set
class create_validation_set:
    def __init__(self, y0, t, args):
        dt = t[1] - t[0]
        n_back = np.ceil(np.abs(args.tau_max/dt.numpy())).astype(int)
        t0 = tf.linspace(t[0] - args.tau_max, t[0], n_back)
        y0_t0 = []
        
        for i in range(t0.shape[0]):
            y0_t0.append(y0(t0[i]))

        self.val_true_y0 = add_interpolator(create_interpolator(tf.concat(y0_t0, axis=0), t0), t0[0], t0[-1])
        val_t_len =  args.val_percentage * (t[-1] - t[0])
        n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
        self.val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)
        self.t = t
        self.args = args
        
    def data_split(self, pred_y_whole, state_dim=-1):
        pred_y_train = pred_y_whole[0:len(self.t), :, :state_dim]
        pred_y_val = pred_y_whole[len(self.t):, :, :state_dim]
        return pred_y_train, pred_y_val
    
    def data_split_any(self, pred_y_whole):
        pred_y_train = pred_y_whole[0:len(self.t), ]
        pred_y_val = pred_y_whole[len(self.t):, :, ]
        return pred_y_train, pred_y_val


## Function to compute gradients w.r.t. trainable variables
class grad_train_var:
    
    def __init__(self, model, func, lam, z, t_lowerlim, t_upperlim, tau, args, t_start = np.array([0.]), mark_flag = 0):
        self.model = model
        self.func = func
        self.lam = lam
        self.z = z
        self.t_ll = t_lowerlim
        self.t_ul = t_upperlim
        self.tau = tau
        self.args = args
        self.t_start = t_start
        self.weight_shapes = [self.model.trainable_weights[i].shape for i in range(len(self.model.trainable_weights))]
        self.scheme = quadpy.c1.gauss_legendre(5)
        
        if isinstance(lam, create_interpolator): 
            self.flag = 1
            self.out_shape = list(self.lam(t_lowerlim).shape)
        else: 
            self.flag = 0
            self.out_shape = list(self.lam.shape)
            
        self.mark_flag = mark_flag
        self.grad = self.integrate_lam_dfdp()
        
    @tf.function
    def calc_jac(self, z):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_weights)
            h = self.func.call_non_mark_nn_part(z)
        dh_dp = tape.jacobian(h, self.model.trainable_weights, experimental_use_pfor=False)
        
        return dh_dp
    
    @tf.function
    def calc_jac_direct_nn(self, z):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_weights)
            h = self.model(z)
        dh_dp = tape.jacobian(h, self.model.trainable_weights, experimental_use_pfor=False)

        return dh_dp
        

    def lam_dfdp(self, t):

        if self.flag:
            if self.mark_flag:
                input = self.func.compute_derivatives(self.z(t), flag_fwd=1)
                dh_dp = self.calc_jac_direct_nn(input)
                
            else:
                input = self.func.process_input(self.z, t, self.tau, flag_fwd=1)
                dh_dp = self.calc_jac(input)
            lam_t = self.lam(t)
        else:
            input = self.func.compute_derivatives(self.z(t), flag_fwd=1)
            dh_dp = self.calc_jac_direct_nn(input)
            lam_t = self.lam
        
        dh_dp  = tf.concat([tf.reshape(dh_dp[i], self.out_shape + [-1]) for i in range(len(dh_dp))], axis=-1)
    
        var_shape_len = len(tf.shape(dh_dp).numpy())
      
        lam_dh_dp =  - tf.einsum('ab,ab'+self.args.base_str[:var_shape_len-2] + 
                               '->ab'+self.args.base_str[:var_shape_len-2],tf.cast(lam_t, tf.float64) , 
                                                        tf.cast(dh_dp, tf.float64)).numpy() #make sure that the space dimension does not collapse
        lam_dh_dp_shp = lam_dh_dp.shape
        lam_dh_dp = tf.transpose(tf.reshape(lam_dh_dp, [lam_dh_dp_shp[0]] + [-1] + [len(self.func.rom_model.grid.x_grid_real)] + [lam_dh_dp_shp[-1]]), perm = [0, 2, 1, 3])
        lam_dh_dp = tf.reduce_sum(lam_dh_dp, axis=-2)
        
        lam_dh_dp = self.integrate_area(lam_dh_dp)
        
        return lam_dh_dp
    
    def integrate_simp(self, y, x, axis = -1):
        dx = x[1:] - x[0:-1]
        
        y_avg = 0.5 * (tf.gather(y, list(range(0, y.shape[axis]-1)), axis=axis) + tf.gather(y, list(range(1, y.shape[axis])), axis=axis))
        
        var_shape_len = len(tf.shape(y_avg).numpy())
        shape_str = self.args.base_str[:var_shape_len]

        y_int = tf.einsum(shape_str + ',' + shape_str[axis] + \
                               '->' + shape_str, tf.cast(y_avg, tf.float64), \
                                                        tf.cast(dx, tf.float64))
        
        return tf.cast(tf.reduce_sum(y_int, axis=axis), tf.float32)
    
    def integrate_area(self, lam_dh_dp):
        
        return self.integrate_simp(lam_dh_dp, self.func.rom_model.grid.x_grid_real, axis = 1)

    def stack_lam_dfdp(self, t):
        
        return np.stack([self.lam_dfdp(t[i]) for i in range(len(t))], axis=-1)

    def integrate_lam_dfdp(self):
        
        lam_dh_dp = self.scheme.integrate(self.stack_lam_dfdp, [self.t_ll, self.t_ul])
        
        m = 0
        n = 0
        lam_dh_dp_list = []
        for i in range(len(self.weight_shapes)):
            n += tf.math.reduce_prod(self.weight_shapes[i]).numpy()
            lam_dh_dp_list.append(tf.reshape(lam_dh_dp[:, m:n], [-1] + list(self.weight_shapes[i])))
            m = n
        
        return lam_dh_dp_list


##### Training class
class train_nDistDDE:

    def __init__(self, func, rom_ens, loss_obj, optimizer, args, plot_obj, time_meter, checkpoint_dir, validation_obj, loss_history_obj):
        self.func = func
        self.rom_ens = rom_ens
        self.loss = loss_obj
        self.optimizer = optimizer
        self.args = args
        self.plot_obj = plot_obj
        self.time_meter = time_meter
        self.checkpoint_dir = checkpoint_dir
        self.val_obj = validation_obj
        self.loss_history = loss_history_obj
        
    def evaluate(self, args, eval_true_z, eval_true_z0, eval_val_true_z, epoch, t):
        
        self.loss.overwrite(args, tf.convert_to_tensor(self.func.rom_model.grid.x_grid_real))
        
        process_true_z0 = process_DistDDE_IC(eval_true_z0, self.func.call_nn_part_indiv, t_lowerlim = t[0] - self.args.tau, t_upperlim = t[0])
                
        pred_zy = ddeinttf(self.func, process_true_z0, tf.concat([t, self.val_obj.val_t], axis=0), fargs = (self.args.tau,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)

        pred_z_train, pred_z_val = self.val_obj.data_split(pred_zy, args.state_dim)
        loss_train = tf.squeeze(self.loss.eval_mode(eval_true_z, pred_z_train))  

        loss_val = tf.squeeze(self.loss.eval_mode(eval_val_true_z, pred_z_val))

        self.loss_history.add(loss_train, loss_val, epoch)
        self.loss_history.save()

        if (epoch % self.args.plot_freq == 0) and self.args.isplot == True:
            self.plot_obj.plot(pred_zy, epoch = epoch)

        return loss_train, loss_val

    def frwd_pass(self, batch_z0, batch_t0, batch_t, batch_z, batch_t_start):
        
        dloss_dpred_z_base = tf.zeros([self.args.batch_size, self.func.args.state_dim], tf.float64)
        
        batch_zy0 = process_DistDDE_IC(create_interpolator(batch_z0, batch_t0), self.func.call_nn_part_indiv, t_lowerlim = batch_t0[-1] - self.args.tau, t_upperlim = batch_t0[-1])
        
        pred_zy = ddeinttf(self.func, batch_zy0 , batch_t, fargs=(self.args.tau, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
        
        #create a interpolator function from the forward pass
        pred_zy0_zy_interp = create_interpolator(tf.concat([batch_zy0(batch_t0), pred_zy], axis=0), tf.concat([batch_t0, batch_t], axis=0))
        pred_z0_z_interp = create_interpolator(tf.concat([batch_z0, pred_zy[:, :, :self.func.args.state_dim]], axis=0), tf.concat([batch_t0, batch_t], axis=0))
        
        pred_zy = pred_zy[1:, ]
        batch_z = batch_z[1:, ]
        pred_z = pred_zy[:, :, :self.func.args.state_dim]
                
        with tf.GradientTape() as tape:
            tape.watch(pred_z)
            loss = self.loss(batch_z, pred_z)
        dloss_dpred_z_whole = tape.gradient(loss, pred_z)
        
        return pred_zy0_zy_interp, pred_zy, pred_z0_z_interp, pred_z, dloss_dpred_z_whole
        
    
    def bkwd_pass(self, pred_zy, pred_zy0_zy_interp, pred_z0_z_interp, dloss_dpred_z_whole, batch_t, batch_t_start):
        
        pred_adj_intrvl = tf.zeros([self.args.batch_time, self.args.batch_size, pred_zy.shape[-1]], tf.float64)
        batch_adj_t_intrvl = tf.linspace(batch_t[-1] + self.args.tau_max, batch_t[-1], self.args.batch_time)
        grads_avg_non_mark_nn_part = [tf.zeros(self.func.non_mark_nn_part.trainable_weights[k].shape, tf.float32) for k in range(len(self.func.non_mark_nn_part.trainable_weights))]
        grads_avg_mark_nn_part = [tf.zeros(self.func.mark_nn_part.trainable_weights[k].shape, tf.float32) for k in range(len(self.func.mark_nn_part.trainable_weights))]

        for k in range(len(batch_t)-1, 0, -1):
            dloss_dpred_z = dloss_dpred_z_whole[k-1, ]

            # Initial conditions for augmented adjoint ODE
            if k == len(batch_t)-1:
                adj_eqn_ic_interp = add_interpolator(create_dirac_interpolator(pred_adj_intrvl, batch_adj_t_intrvl, tf.concat([dloss_dpred_z, tf.zeros([self.args.batch_size, pred_zy.shape[-1] - self.func.args.state_dim])], axis=-1), batch_adj_t_intrvl[-1]), batch_adj_t_intrvl[0], batch_adj_t_intrvl[-1])
            else:
                adj_eqn_ic_interp.add(create_dirac_interpolator(pred_adj_intrvl, batch_adj_t_intrvl, tf.concat([dloss_dpred_z, tf.zeros([self.args.batch_size, pred_zy.shape[-1] - self.func.args.state_dim])], axis=-1), batch_adj_t_intrvl[-1]), batch_adj_t_intrvl[0], batch_adj_t_intrvl[-1])

            batch_adj_t_intrvl = tf.linspace(batch_t[k], batch_t[k-1], self.args.batch_time)

            # Solve for the augmented adjoint ODE 
            pred_adj_intrvl = ddeinttf(self.adj_func, adj_eqn_ic_interp, batch_adj_t_intrvl, fargs = (self.args.tau, pred_zy0_zy_interp, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
            
            # Compute gradients w.r.t. trainable weights of mark_nn_part

            grads = grad_train_var(self.func.mark_nn_part, self.func, create_interpolator(pred_adj_intrvl[:, :, :self.func.args.state_dim], batch_adj_t_intrvl), pred_z0_z_interp, batch_adj_t_intrvl[-1], batch_adj_t_intrvl[0], self.args.tau, self.args, batch_t_start, mark_flag = 1).grad # passing pred_zy0_zy_interp because self.func.process_input expects full vector zy

            grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]

            for i in range(len(self.func.mark_nn_part.trainable_weights)):
                grads_avg_mark_nn_part[i] = grads_avg_mark_nn_part[i] + grads_avg_intrvl[i]
                
            # Compute gradients w.r.t. trainable weights of non_mark_nn_part

            grads = grad_train_var(self.func.non_mark_nn_part, self.func, create_interpolator(pred_adj_intrvl[:, :, self.func.args.state_dim:], batch_adj_t_intrvl), pred_zy0_zy_interp, batch_adj_t_intrvl[-1], batch_adj_t_intrvl[0], self.args.tau, self.args, batch_t_start).grad # passing pred_zy0_zy_interp because self.func.process_input expects full vector zy

            grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]

            for i in range(len(self.func.non_mark_nn_part.trainable_weights)):
                grads_avg_non_mark_nn_part[i] = grads_avg_non_mark_nn_part[i] + grads_avg_intrvl[i]
            
        return pred_adj_intrvl, batch_adj_t_intrvl, grads_avg_mark_nn_part, grads_avg_non_mark_nn_part
    
    def reg_gradients(self, model, lambda_l1, lambda_l2, existing_grads):
        
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            loss = self.loss.reglularizer(model, lambda_l1, lambda_l2)
        dloss_dweights = tape.gradient(loss, model.trainable_weights)
        
        for i in range(len(model.trainable_weights)): 
            existing_grads[i] = existing_grads[i] + dloss_dweights[i]
        
        return existing_grads
    
    def prune_weights(self, model, thres = 1e-5):
        
        for i in range(len(model.trainable_weights)):
            
            new_val = tf.where( tf.abs(model.trainable_weights[i]) < thres, 0., model.trainable_weights[i])
            model.trainable_weights[i].assign(new_val)
    
    
    def train(self, true_z_ens, true_z0_ens, t, eval_true_z, eval_true_z0, eval_val_true_z, eval_rom, norm_eval, norm_train_ens):
        end = time.time()
        
        epoch = 0
        loss_train, loss_val = self.evaluate(self.args.args_eval_lf, eval_true_z, eval_true_z0, eval_val_true_z, epoch, t)
                
        print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | Time Elapsed {:.4f}'.format(epoch, loss_train.numpy(), loss_val.numpy(), self.time_meter.avg/60), 'mins')
                
        for epoch in range(1, self.args.epochs + 1):

            train_order = np.random.permutation(self.args.train_ens_size)
            
            for i in train_order:
                args = self.args.args_train_lf[i]
                batch_obj = create_batch(true_z_ens[i], true_z0_ens[i], t, self.args)
                
                self.func.overwrite_rom(self.args.args_train_lf[i], self.rom_ens[i], norm_train_ens[i])
                
                self.adj_func = ncm_adj_eqn(self.func, self.func.rom_model.jac)
                
                self.loss.overwrite(args, tf.convert_to_tensor(self.func.rom_model.grid.x_grid_real))
                
                for itr in tqdm(range(1, self.args.niters + 1), desc = 'Iterations', file=sys.stdout):

                    if itr == 1:
                        batch_z0, batch_t0, batch_t, batch_z, batch_t_start = batch_obj.get_batch(1)
                    else:
                        batch_z0, batch_t0, batch_t, batch_z, batch_t_start = batch_obj.get_batch()

                    if self.args.is_tstart_zero: batch_t_start = tf.zeros(list(batch_t_start.shape))

                    pred_zy0_zy_interp, pred_zy, pred_z0_z_interp, pred_z, dloss_dpred_z_whole = self.frwd_pass(batch_z0, batch_t0, batch_t, batch_z, batch_t_start)
                    
                    pred_adj_intrvl, batch_adj_t_intrvl, grads_avg_mark_nn_part, grads_avg_non_mark_nn_part = self.bkwd_pass(pred_zy, pred_zy0_zy_interp, pred_z0_z_interp, dloss_dpred_z_whole, batch_t, batch_t_start)  
#                     print('before grads_avg_non_mark_nn_part', grads_avg_non_mark_nn_part)
                    grads = grad_train_var(self.func.non_mark_nn_part, self.func, pred_adj_intrvl[-1, :, self.func.args.state_dim:], pred_z0_z_interp, -self.args.tau, 0., self.args.tau, self.args, batch_t_start).grad #

                    grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]
                    
                    for i in range(len(self.func.non_mark_nn_part.trainable_weights)): 
                        grads_avg_non_mark_nn_part[i] = grads_avg_non_mark_nn_part[i] + grads_avg_intrvl[i]
                    
                    grads_avg_mark_nn_part = [tf.clip_by_norm(g, 1.) for g in grads_avg_mark_nn_part]
                    grads_avg_non_mark_nn_part = [tf.clip_by_norm(g, 1.) for g in grads_avg_non_mark_nn_part]

                    grads_avg_mark_nn_part = self.reg_gradients(self.func.mark_nn_part, self.args.lambda_l1_mark, self.args.lambda_l2_mark, grads_avg_mark_nn_part)
                    
                    grads_avg_non_mark_nn_part = self.reg_gradients(self.func.non_mark_nn_part, self.args.lambda_l1_non_mark, self.args.lambda_l2_non_mark, grads_avg_non_mark_nn_part)
                    
#                     print('grads_avg_mark_nn_part', grads_avg_mark_nn_part)
#                     print('grads_avg_non_mark_nn_part', grads_avg_non_mark_nn_part)
        
                    grads_zip = zip(grads_avg_mark_nn_part + grads_avg_non_mark_nn_part, self.func.mark_nn_part.trainable_weights + self.func.non_mark_nn_part.trainable_weights)
                    
                    self.optimizer.apply_gradients(grads_zip)
                    
                    if (self.args.prune_thres != 0.) and (self.args.lambda_l1_mark != 0.):
                        self.prune_weights(self.func.mark_nn_part, thres = self.args.prune_thres) # thres = 5e-3)
                    if (self.args.prune_thres != 0.) and (self.args.lambda_l1_non_mark != 0.):
                        self.prune_weights(self.func.non_mark_nn_part, thres = self.args.prune_thres) # thres = 5e-3)

                self.time_meter.update(time.time() - end)

            # Plotting
            if epoch % self.args.test_freq == 0:
                
                self.func.overwrite_rom(self.args.args_eval_lf, eval_rom, norm_eval)
                
                loss_train, loss_val = self.evaluate(self.args.args_eval_lf, eval_true_z, eval_true_z0, eval_val_true_z, epoch, t)

                print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | LR {:.4f} | Time Elapsed {:.4f}'.format(epoch, loss_train.numpy(), loss_val.numpy(), self.optimizer.learning_rate(self.optimizer.iterations.numpy()).numpy(), self.time_meter.avg/60), 'mins')

                self.func.save_weights(self.checkpoint_dir.format(epoch=epoch))
                
                print(self.func.mark_nn_part.trainable_weights[0])
                print(self.func.non_mark_nn_part.trainable_weights[0])
                
            end = time.time()
            
            
            
## Helper class to compute average time, etc.
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
### Helper class to store, write and read loss values
class history:
    def __init__(self, args):
        self.train_loss = []
        self.val_loss = []
        self.epoch = []
        self.args = args

    def add(self, train_loss, val_loss, epoch):
        self.train_loss.append(train_loss.numpy())
        self.val_loss.append(val_loss.numpy())
        self.epoch.append(epoch)

    def save(self):
        with open(self.args.model_dir + '/loss_history.p', 'wb') as f:
            pickle.dump([self.epoch, self.train_loss, self.val_loss], f)

    def read(self):
        with open(self.args.model_dir + '/loss_history.p', 'rb') as f:
            [self.epoch, self.train_loss, self.val_loss] = pickle.load(f)