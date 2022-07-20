from src.utilities.DDE_Solver import ddeinttf
from src.solvers.neuralDistDDE_train_HOTs import ncm_arguments, create_batch, create_interpolator, create_dirac_interpolator, add_interpolator, create_validation_set, RunningAverageMeter, history, grad_train_var

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


#### Adjoint system
class ncm_adj_eqn:

    def __init__(self, model, jac = None):
        self.f = model.mark_nn_part
        self.rom = model.rom_model
        self.args = model.args
        self.jac = jac
        self.model = model
    
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
        
        z_t = zy(t)
        
        dz_dx_stack_deriv = self.model.compute_derivatives(z_t, flag_fwd = 1)
        
        lam_t = lam_mu(t)
        dlam_dx_stack_deriv = self.model.compute_derivatives(lam_t)

        
        lam_rhs = tf.zeros(z_t.shape, tf.float64)
        
        if self.jac is not None:
            drom_dz_stack_deriv = tf.convert_to_tensor(self.jac(dz_dx_stack_deriv.numpy(), t, t_start))
            
        else:
            drom_dz_stack_deriv = self.calc_rom_jac(dz_dx_stack_deriv, t, t_start)
#         print('jac', drom_dz_stack_deriv)   
        drom_dz_stack_deriv += self.calc_jac_mark_nn(dz_dx_stack_deriv)
#         print('/n')
#         print('jac+jac_nn', drom_dz_stack_deriv)
#         print('/n')
        drom_dz_stack_deriv = tf.transpose(tf.linalg.diag_part(tf.transpose(drom_dz_stack_deriv, perm = [0, 3, 1, 2])), perm = [0, 2, 1])
        
        dlam_drom_dz_stack_deriv = tf.einsum('abd, abd -> abd', tf.cast(dlam_dx_stack_deriv, tf.float64), tf.cast(drom_dz_stack_deriv, tf.float64))
        
        for i in range(dlam_drom_dz_stack_deriv.shape[-1]):
            lam_rhs = lam_rhs + (-1)**(i+1) * dlam_drom_dz_stack_deriv[:, :, i]
        
#         if self.args.skip_bnd != 1: 
        lam_rhs_bnd = self.boundary(lam_rhs[:, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], drom_dz_stack_deriv)

        lam_rhs = tf.cast(tf.concat([lam_rhs_bnd[:, 0:self.args.max_deriv - 1], lam_rhs[:, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], lam_rhs_bnd[:, self.args.max_deriv - 1:]], axis=-1), dtype = tf.float32)

        return tf.concat([lam_rhs], axis=-1)
    
    def boundary(self, lam_rhs_int, drom_dz_stack_deriv):
        # currently only supports dirichlet BCs
       
        size_x = drom_dz_stack_deriv.shape[1]
        
        def mat_entry(eqn_id, idx, bnd):
            
            A = 0
            
            for i in range(self.args.max_deriv - eqn_id):
                A += (-1)**(i+1) * self.rom.full_vander[i][bnd, idx] * drom_dz_stack_deriv[:, bnd, i + (eqn_id + 1)]
                
            return A
        
        def mark_rhs_entry(eqn_id, bnd):
            
            b = 0
            
            for i in range(self.args.max_deriv - eqn_id):
                b += (-1)**(i) * tf.einsum('b, cb -> c', self.rom.full_vander[i][bnd, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], lam_rhs_int) * drom_dz_stack_deriv[:, bnd, i + (eqn_id + 1)]
                
            return b
        
        B = []
        b = []
        k = 0
        
        for bnd in [0, -1]:
            for i in range((self.args.max_deriv)):
                B.append([])
                b.append(mark_rhs_entry(i, bnd))

                for j in range(2 * (self.args.max_deriv - 1)):
                    if j < self.args.max_deriv - 1:
                        B[k].append(mat_entry(i, j, bnd))
                    else:
                        B[k].append(mat_entry(i, -(self.args.max_deriv - 1)+(j % (self.args.max_deriv - 1)), bnd))
                    
                B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
                k += 1
                
        B_tmp = [B[0] * self.args.nbc_l + B[1] * self.args.dbc_l]
        b_tmp = [(b[0]) * self.args.nbc_l + (b[1]) * self.args.dbc_l]

        for i in range(2, self.args.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i])

        B_tmp.append(B[self.args.max_deriv] * self.args.nbc_r + B[self.args.max_deriv+1] * self.args.dbc_r)
        b_tmp.append((b[self.args.max_deriv]) * self.args.nbc_r + (b[self.args.max_deriv+1]) * self.args.dbc_r)

        for i in range(self.args.max_deriv+2, 2*self.args.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i])
        
        B = tf.concat(B_tmp, axis=-2)
        b = tf.stack(b_tmp, axis=-1)
        
        lam_rhs_bnd = tf.einsum('cab, cb ->ca', tf.linalg.pinv(B), b)
        
        return lam_rhs_bnd


##### Training class
class train_nODE:

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
                
        pred_zy = ddeinttf(self.func, eval_true_z0, tf.concat([t, self.val_obj.val_t], axis=0), fargs = (self.args.tau,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)

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
        
        pred_z = ddeinttf(self.func, create_interpolator(batch_z0, batch_t0) , batch_t, fargs=(self.args.tau, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
        
        #create a interpolator function from the forward pass
        pred_z0_z_interp = create_interpolator(tf.concat([batch_z0, pred_z], axis=0), tf.concat([batch_t0, batch_t], axis=0))
        
        pred_z = pred_z[1:, ]
        batch_z = batch_z[1:, ]
                
        with tf.GradientTape() as tape:
            tape.watch(pred_z)
            loss = self.loss(batch_z, pred_z)
        dloss_dpred_z_whole = tape.gradient(loss, pred_z)
        
        return pred_z0_z_interp, pred_z, dloss_dpred_z_whole
        
    
    def bkwd_pass(self, pred_zy, pred_z0_z_interp, dloss_dpred_z_whole, batch_t, batch_t_start):
        
        pred_adj_intrvl = tf.zeros([self.args.batch_time, self.args.batch_size, pred_zy.shape[-1]], tf.float64)
        batch_adj_t_intrvl = tf.linspace(batch_t[-1] + self.args.tau_max, batch_t[-1], self.args.batch_time)
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
            pred_adj_intrvl = ddeinttf(self.adj_func, adj_eqn_ic_interp, batch_adj_t_intrvl, fargs = (self.args.tau, pred_z0_z_interp, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
            
            # Compute gradients w.r.t. trainable weights of mark_nn_part

            grads = grad_train_var(self.func.mark_nn_part, self.func, create_interpolator(pred_adj_intrvl[:, :, :self.func.args.state_dim], batch_adj_t_intrvl), pred_z0_z_interp, batch_adj_t_intrvl[-1], batch_adj_t_intrvl[0], self.args.tau, self.args, batch_t_start, mark_flag = 1).grad # passing pred_zy0_zy_interp because self.func.process_input expects full vector zy

            grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]

            for i in range(len(self.func.mark_nn_part.trainable_weights)):
                grads_avg_mark_nn_part[i] = grads_avg_mark_nn_part[i] + grads_avg_intrvl[i]
                
            
        return pred_adj_intrvl, batch_adj_t_intrvl, grads_avg_mark_nn_part
    
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

                    pred_z0_z_interp, pred_z, dloss_dpred_z_whole = self.frwd_pass(batch_z0, batch_t0, batch_t, batch_z, batch_t_start)
#                     print(pred_z)
#                     print('/n')
                    pred_adj_intrvl, batch_adj_t_intrvl, grads_avg_mark_nn_part = self.bkwd_pass(pred_z, pred_z0_z_interp, dloss_dpred_z_whole, batch_t, batch_t_start)  
                    
                    grads_avg_mark_nn_part = [tf.clip_by_norm(g, 1.) for g in grads_avg_mark_nn_part]

                    grads_avg_mark_nn_part = self.reg_gradients(self.func.mark_nn_part, self.args.lambda_l1_mark, self.args.lambda_l2_mark, grads_avg_mark_nn_part)

                    grads_zip = zip(grads_avg_mark_nn_part, self.func.mark_nn_part.trainable_weights)
                    
                    self.optimizer.apply_gradients(grads_zip)
                    
                    if (self.args.prune_thres != 0.) and (self.args.lambda_l1_mark != 0.):
                        self.prune_weights(self.func.mark_nn_part, thres = self.args.prune_thres) # thres = 5e-3)

                self.time_meter.update(time.time() - end)

            # Plotting
            if epoch % self.args.test_freq == 0:
                
                self.func.overwrite_rom(self.args.args_eval_lf, eval_rom, norm_eval)
                
                loss_train, loss_val = self.evaluate(self.args.args_eval_lf, eval_true_z, eval_true_z0, eval_val_true_z, epoch, t)

                print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | LR {:.4f} | Time Elapsed {:.4f}'.format(epoch, loss_train.numpy(), loss_val.numpy(), self.optimizer.learning_rate(self.optimizer.iterations.numpy()).numpy(), self.time_meter.avg/60), 'mins')

                self.func.save_weights(self.checkpoint_dir.format(epoch=epoch))
                
                print(self.func.mark_nn_part.trainable_weights[0])
                
            end = time.time()
            