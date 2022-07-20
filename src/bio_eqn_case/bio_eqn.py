from IPython.core.debugger import set_trace

import numpy as np
from math import *
import tensorflow as tf
import scipy as sc
from scipy.linalg import block_diag

from src.utilities.helper_classes import integrate_simp_cumsum, vander, diagonalize

tf.keras.backend.set_floatx('float32')

#### User-defined parameters
class bio_eqn_args:
    def __init__(self, T, nt, nz, z_max, a, g_max, k_P, k_D, k_W, K_N, K_P, m_p, m_z, T_opt, mu_max, alpha, beta, epsilon, lamb, gamma, C_P, C_Z, C_D, gamma_c, T_bio_min, T_bio_max, bio_model, Temp, Salt, I_0 = 158.075, K_z0 = 8.64, K_zb = 0.0864, gamma_K = 0.1, extra_terms = 0, acc = 2, max_deriv = 2, skip_bnd = 0, water_density = 1028.13, dbc_l = 0, nbc_l = 1, dbc_r = 0, nbc_r = 1, dleft_bv_dt = 0, dright_bv_dt = 0, Z_mort_sat = 1.):
        self.T = T
        self.nt = nt
        self.nz = nz
        self.dt = self.T / self.nt

        self.z_max = z_max
        self.a = a
        self.g_max = g_max
        self.k_P = k_P
        self.k_D = k_D
        self.k_W = k_W
        self.K_N = K_N
        self.K_P = K_P
        self.m_p = m_p
        self.m_z = m_z
#         self.s_P = s_P
#         self.s_D = s_D
        self.T_opt = T_opt
        self.mu_max = mu_max
        self.alpha = alpha
        
        self.beta = beta
        self.epsilon = epsilon
        self.lamb = lamb
        self.gamma = gamma
        
        self.C_P = C_P
        self.C_Z = C_Z
        self.C_D = C_D
        self.gamma_c = gamma_c
        
        self.T_bio_min = T_bio_min
        self.T_bio_max = T_bio_max 
        self.Temp = Temp
        self.Salt = Salt
        self.I_0 = I_0
        
        self.K_z0 = K_z0
        self.K_zb = K_zb
        self.gamma_K = gamma_K
        
        self.Z_mort_sat = Z_mort_sat
        
        self.z = tf.linspace(0., z_max, nz)
        self.dz = self.z[1] - self.z[0]
        self.T_bio = tf.cast(tf.linspace(T_bio_min, T_bio_max, nz), tf.float32)
        
        self.I_0_t = lambda t: I_0 - 50.*np.sin((t + 365./6.) * 2. * pi / 365.)

        self.bio_model = bio_model
        self.extra_terms = extra_terms
        
        if bio_model == 'NPZ-OA': self.state_dim = 5 * nz
        elif bio_model == 'NPZD-OA': self.state_dim = 6 * nz
            
        self.acc = acc
        self.max_deriv = max_deriv
        self.skip_bnd = skip_bnd
        self.water_density = water_density
        
        self.dbc_l = dbc_l
        self.dbc_r = dbc_r
        self.nbc_l = nbc_l
        self.nbc_r = nbc_r
        self.dleft_bv_dt = dleft_bv_dt
        self.dright_bv_dt = dright_bv_dt

### RHS of Bio Eqns
class bio_eqn:

    def __init__(self, app, diff_coeff, grid = None, deriv_obj = None):
        self.app = app
        self.diff_coeff = diff_coeff
        
        self.grid = grid
        self.deriv_obj = deriv_obj
        
        if deriv_obj != None:
            self.full_vander = self.get_vander_matrices(self.grid.x_grid, max_deriv = app.max_deriv)
        else: self.full_vander = None
        
    def get_vander_matrices(self, x, max_deriv = None):
        
        if max_deriv == None: max_deriv = self.app.max_deriv
        
        A = []
        for i in range(max_deriv+1):
            vand = [self.deriv_obj.vander(x, m=i, acc=self.app.acc)]
            
            for i in range(int(self.app.state_dim / self.app.nz) - 1):
                vand.append(vand[0])
            
            vand = block_diag(*vand)
            
            A.append(vand)
            
        return A
        
    def PAR(self, t, I_0_t):
        
        self.I = lambda t: np.stack([I_0_t(t[i]) * np.exp(self.app.k_W * self.app.z.numpy()) for i in range(t.shape[0])], axis=0)

#         I_P = np.concatenate([np.expand_dims(np.zeros(P[:, 0].shape), axis=-1), integrate_simp_cumsum(P, self.app.z, axis=-1)], axis=-1)
#         I_D = np.concatenate([np.expand_dims(np.zeros(D[:, 0].shape), axis=-1), integrate_simp_cumsum(D, self.app.z, axis=-1)], axis=-1)

#         I_P = np.exp(self.app.k_P * I_P)
#         I_D = np.exp(self.app.k_D * I_D)
        
        I_z = self.I(t) #* I_P * I_D
        
        return I_z
        
    def rhs_bio(self, x_t, t, t_start):
        t = t + t_start
        
        self.f_N = x_t[:, 0*self.app.nz:1*self.app.nz] / (self.app.K_N + x_t[:, 0*self.app.nz:1*self.app.nz])
            
        self.f_I = (1 - np.exp(-self.app.alpha * self.I_t / self.app.mu_max)) * np.exp(-self.app.beta * self.I_t / self.app.mu_max)

        self.U_P = self.app.mu_max * self.f_N * self.f_I * self.f_T * x_t[:, 1*self.app.nz:2*self.app.nz]

        self.G_Z = self.app.g_max * self.f_T * x_t[:, 2*self.app.nz:3*self.app.nz] * x_t[:, 1*self.app.nz:2*self.app.nz]**2 / (x_t[:, 1*self.app.nz:2*self.app.nz]**2 + self.app.K_P**2)

#             dP_dz = np.einsum('ab, cb -> ca', self.vander_dz, x_t[:, 1*self.app.nz:2*self.app.nz])
#             dD_dz = np.einsum('ab, cb -> ca', self.vander_dz, x_t[:, 3*self.app.nz:4*self.app.nz])

        self.mort_P = self.app.m_p * x_t[:, 1*self.app.nz:2*self.app.nz]
        self.mort_Z = self.app.m_z * x_t[:, 2*self.app.nz:3*self.app.nz]
        self.quad_mort_Z = self.app.extra_terms * self.app.m_z * x_t[:, 2*self.app.nz:3*self.app.nz]**2
#         self.quad_mort_Z = self.app.extra_terms * self.app.m_z * x_t[:, 2*self.app.nz:3*self.app.nz]**2 / (self.app.Z_mort_sat + x_t[:, 2*self.app.nz:3*self.app.nz])
            
        if self.app.bio_model == 'NPZD-OA':
            
            self.remin = self.app.epsilon * self.f_T * x_t[:, 3*self.app.nz:4*self.app.nz]
            
            dxdt = [-self.U_P + self.app.lamb * self.G_Z + self.remin]
            
            dxdt.append(self.U_P - self.G_Z - self.f_T * self.mort_P)# + self.app.s_P * dP_dz) # add sedimentation

            dxdt.append(self.app.gamma * self.G_Z - self.f_T * self.mort_Z - self.quad_mort_Z)

            dxdt.append((1. - self.app.gamma - self.app.lamb) * self.G_Z + self.mort_P + self.mort_Z + self.quad_mort_Z - self.remin)# + self.app.s_D * dD_dz) # add sedimentation
            
        elif self.app.bio_model == 'NPZ-OA':
            
            dxdt = [-self.U_P + (1. - self.app.gamma) * self.G_Z + self.mort_P + self.mort_Z + self.quad_mort_Z]
            
            dxdt.append(self.U_P - self.G_Z - self.f_T * self.mort_P)# + self.app.s_P * dP_dz) # add sedimentation

            dxdt.append(self.app.gamma * self.G_Z - self.f_T * self.mort_Z - self.quad_mort_Z)
        
        dxdt = np.concatenate(dxdt, axis=-1)
        
        return dxdt
    
    def rhs_oa(self, dxdt):
        
        if self.app.bio_model == 'NPZD-OA':
            doadt = [-self.app.C_P * dxdt[:, 1*self.app.nz:2*self.app.nz] 
                    -self.app.C_Z * dxdt[:, 2*self.app.nz:3*self.app.nz]
                    -self.app.C_D * dxdt[:, 3*self.app.nz:4*self.app.nz] - self.app.gamma_c * self.app.C_P * self.U_P]
            doadt.append((-dxdt[:, 0*self.app.nz:1*self.app.nz] - 2. * self.app.gamma_c * self.app.C_P * self.U_P) / self.app.water_density)
            
        if self.app.bio_model == 'NPZ-OA':
            doadt = [-self.app.C_P * dxdt[:, 1*self.app.nz:2*self.app.nz] 
                    -self.app.C_Z * dxdt[:, 2*self.app.nz:3*self.app.nz]
                    - self.app.gamma_c * self.app.C_P * self.U_P]
            doadt.append((-dxdt[:, 0*self.app.nz:1*self.app.nz] - 2. * self.app.gamma_c * self.app.C_P * self.U_P) / self.app.water_density)
            
        doadt = np.concatenate(doadt, axis=-1)
        
        return doadt
    
    def add_diff(self, dxdt, x_t, t, t_start):
        t = t + t_start

        K_zt = np.stack([self.diff_coeff(self.app.z, t[k]) for k in range(len(t))], axis=0)
        
        K_zt = np.tile(K_zt, [1, int(self.app.state_dim / self.app.nz)])
           
        dK_zt_dz = np.einsum('ab, cb -> ca', self.full_vander[1], K_zt)
        dx_t_dz = np.einsum('ab, cb -> ca', self.full_vander[1], x_t)
        d2x_t_dz2 = np.einsum('ab, cb -> ca', self.full_vander[2], x_t)
        
        dxdt += np.einsum('ab, ab -> ab', dK_zt_dz, dx_t_dz) + np.einsum('ab, ab -> ab', K_zt, d2x_t_dz2)
        
        return dxdt
    
    def __call__(self, x, t, t_start = np.array([0.])):
        x_t = x(t).numpy()
        
        dxdt = self.rhs(x_t, t, t_start)
        
        return dxdt
    
    def rhs(self, x_t, t, t_start = np.array([0.])):
        
        self.f_T = np.exp( - self.app.a * np.abs(self.app.Temp(self.app.z, t + t_start) - self.app.T_opt))
        self.I_t = self.PAR(t+t_start, self.app.I_0_t)
        
        dxdt = self.rhs_bio(x_t, t, t_start)
        
        doadt = self.rhs_oa(dxdt)
        
        dxdt = np.concatenate([dxdt, doadt], axis=-1)
        
        dxdt = self.add_diff(dxdt, x_t, t, t_start)
        
        if self.app.skip_bnd != 1:
            dxdt = self.apply_bnd(t+t_start, x_t, dxdt)
        
        dxdt = tf.convert_to_tensor(dxdt, tf.float32)
        
        return dxdt
    
    def apply_bnd(self, t, x_t, dxdt):
        dxdt_bnd = self.boundary(t, x_t, dxdt)
        dxdt = self.separate_states(dxdt)

        dxdt = np.concatenate((dxdt_bnd[:, 0:self.app.max_deriv - 1], dxdt[:, self.app.max_deriv - 1:-(self.app.max_deriv - 1)], dxdt_bnd[:, self.app.max_deriv - 1:]), axis=-1)

        dxdt = self.merge_states(dxdt)
        return dxdt
    
    def initial_cond_solve(self, x, t, t_start = np.array([0.])):
        x_t = x(t).numpy()
        
        self.f_T = np.ones(x_t[:, 0*self.app.nz:1*self.app.nz].shape)
        I_0_t = lambda k: self.app.I_0
        self.I_t = self.PAR(t+t_start, I_0_t)
        
        dxdt = self.rhs_bio(x_t, t, t_start)
        
        dxdt = tf.convert_to_tensor(dxdt, tf.float32)
        
        return dxdt
    
    def jac(self, x_t_stack, t, t_start):
        
        t = t + t_start
        
        x_t = x_t_stack[:, :, 0]
        
        f_T = np.exp( - self.app.a * np.abs(self.app.Temp(self.app.z, t) - self.app.T_opt))
        I_t = self.PAR(t, self.app.I_0_t)
        f_N = x_t[:, 0*self.app.nz:1*self.app.nz] / (self.app.K_N + x_t[:, 0*self.app.nz:1*self.app.nz])

        df_N_dN = self.app.K_N / (self.app.K_N + x_t[:, 0*self.app.nz:1*self.app.nz])**2

        f_I = (1 - np.exp(-self.app.alpha * I_t / self.app.mu_max)) * np.exp(-self.app.beta * I_t / self.app.mu_max)

        dU_P_dN = self.app.mu_max * df_N_dN * f_I * f_T * x_t[:, 1*self.app.nz:2*self.app.nz]

        dU_P_dP = self.app.mu_max * f_N * f_I * f_T

        dG_Z_dP = self.app.g_max * f_T * x_t[:, 2*self.app.nz:3*self.app.nz] * 2. * x_t[:, 1*self.app.nz:2*self.app.nz] * self.app.K_P**2 / (x_t[:, 1*self.app.nz:2*self.app.nz]**2 + self.app.K_P**2)**2

        dG_Z_dZ = self.app.g_max * f_T * x_t[:, 1*self.app.nz:2*self.app.nz]**2 / (x_t[:, 1*self.app.nz:2*self.app.nz]**2 + self.app.K_P**2)

        dmort_P_dP = self.app.m_p * np.ones(x_t[:, 1*self.app.nz:2*self.app.nz].shape)
        dmort_Z_dZ =  self.app.m_z * np.ones(x_t[:, 2*self.app.nz:3*self.app.nz].shape)
        dquad_mort_Z_dZ = self.app.extra_terms * self.app.m_z * 2. * x_t[:, 2*self.app.nz:3*self.app.nz]
#         dquad_mort_Z_dZ = self.app.extra_terms * ((self.app.Z_mort_sat + x_t[:, 2*self.app.nz:3*self.app.nz]) * self.app.m_z * 2. * x_t[:, 2*self.app.nz:3*self.app.nz] - self.app.m_z * x_t[:, 2*self.app.nz:3*self.app.nz]**2) / (self.app.Z_mort_sat + x_t[:, 2*self.app.nz:3*self.app.nz])**2
        
        if self.app.bio_model == 'NPZD-OA': 
            
            dreminD_dD = self.app.epsilon * self.f_T * np.ones(x_t[:, 3*self.app.nz:4*self.app.nz].shape)
            
            no_deriv = np.zeros(x_t[:, 4*self.app.nz:5*self.app.nz].shape)
            
            dS_N_dN = -dU_P_dN
            dS_N_dP = -dU_P_dP + self.app.lamb * dG_Z_dP
            dS_N_dZ = self.app.lamb * dG_Z_dZ
            dS_N_dD = dreminD_dD
            dS_N_dDIC = no_deriv
            dS_N_dTA = no_deriv
            
            dS_P_dN = dU_P_dN
            dS_P_dP = dU_P_dP - dG_Z_dP - self.f_T * dmort_P_dP
            dS_P_dZ = -dG_Z_dZ
            dS_P_dD = no_deriv
            dS_P_dDIC = no_deriv
            dS_P_dTA = no_deriv
            
            dS_Z_dN = no_deriv
            dS_Z_dP = self.app.gamma * dG_Z_dP
            dS_Z_dZ = self.app.gamma * dG_Z_dZ - self.f_T * dmort_Z_dZ - dquad_mort_Z_dZ
            dS_Z_dD = no_deriv
            dS_Z_dDIC = no_deriv
            dS_Z_dTA = no_deriv
            
            dS_D_dN = no_deriv
            dS_D_dP = (1. - self.app.gamma - self.app.lamb) * dG_Z_dP + dmort_P_dP 
            dS_D_dZ = (1. - self.app.gamma - self.app.lamb) * dG_Z_dZ + dmort_Z_dZ + dquad_mort_Z_dZ
            dS_D_dD = -dreminD_dD
            dS_D_dDIC = no_deriv
            dS_D_dTA = no_deriv
            
            dS_DIC_dN = -self.app.C_P * dS_P_dN - self.app.C_Z * dS_Z_dN - self.app.C_D * dS_D_dN - self.app.gamma_c * self.app.C_P * dU_P_dN
            dS_DIC_dP = -self.app.C_P * dS_P_dP - self.app.C_Z * dS_Z_dP - self.app.C_D * dS_D_dP - self.app.gamma_c * self.app.C_P * dU_P_dP
            dS_DIC_dZ = -self.app.C_P * dS_P_dZ - self.app.C_Z * dS_Z_dZ - self.app.C_D * dS_D_dZ
            dS_DIC_dD = -self.app.C_P * dS_P_dD - self.app.C_Z * dS_Z_dD - self.app.C_D * dS_D_dD
            dS_DIC_dDIC = no_deriv
            dS_DIC_dTA = no_deriv
            
            dS_TA_dN = (-dS_N_dN - 2. * self.app.gamma_c * self.app.C_P * dU_P_dN) / self.app.water_density
            dS_TA_dP = (-dS_N_dP - 2. * self.app.gamma_c * self.app.C_P * dU_P_dP) / self.app.water_density
            dS_TA_dZ = -dS_N_dZ / self.app.water_density
            dS_TA_dD = -dS_N_dD / self.app.water_density
            dS_TA_dDIC = -dS_N_dDIC / self.app.water_density
            dS_TA_dTA = -dS_N_dTA / self.app.water_density
            
            dS_dB = np.block([[diagonalize(dS_N_dN), diagonalize(dS_N_dP), diagonalize(dS_N_dZ), diagonalize(dS_N_dD), diagonalize(dS_N_dDIC), diagonalize(dS_N_dTA)], [diagonalize(dS_P_dN), diagonalize(dS_P_dP), diagonalize(dS_P_dZ), diagonalize(dS_P_dD), diagonalize(dS_P_dDIC), diagonalize(dS_P_dTA)], [diagonalize(dS_Z_dN), diagonalize(dS_Z_dP), diagonalize(dS_Z_dZ), diagonalize(dS_Z_dD), diagonalize(dS_Z_dDIC), diagonalize(dS_Z_dTA)], [diagonalize(dS_D_dN), diagonalize(dS_D_dP), diagonalize(dS_D_dZ), diagonalize(dS_D_dD), diagonalize(dS_D_dDIC), diagonalize(dS_D_dTA)], [diagonalize(dS_DIC_dN), diagonalize(dS_DIC_dP), diagonalize(dS_DIC_dZ), diagonalize(dS_DIC_dD), diagonalize(dS_DIC_dDIC), diagonalize(dS_DIC_dTA)], [diagonalize(dS_TA_dN), diagonalize(dS_TA_dP), diagonalize(dS_TA_dZ), diagonalize(dS_TA_dD), diagonalize(dS_TA_dDIC), diagonalize(dS_TA_dTA)]])
            
            
        elif self.app.bio_model == 'NPZ-OA': 
            
            no_deriv = np.zeros(x_t[:, 4*self.app.nz:5*self.app.nz].shape)

            dS_N_dN = -dU_P_dN
            dS_N_dP = -dU_P_dP + (1. - self.app.gamma) * dG_Z_dP + dmort_P_dP
            dS_N_dZ = (1. - self.app.gamma) * dG_Z_dZ + dmort_Z_dZ + dquad_mort_Z_dZ
            dS_N_dDIC = no_deriv
            dS_N_dTA = no_deriv
            
            dS_P_dN = dU_P_dN
            dS_P_dP = dU_P_dP - dG_Z_dP - self.f_T * dmort_P_dP
            dS_P_dZ = -dG_Z_dZ
            dS_P_dDIC = no_deriv
            dS_P_dTA = no_deriv
            
            dS_Z_dN = no_deriv
            dS_Z_dP = self.app.gamma * dG_Z_dP
            dS_Z_dZ = self.app.gamma * dG_Z_dZ - self.f_T * dmort_Z_dZ - dquad_mort_Z_dZ
            dS_Z_dDIC = no_deriv
            dS_Z_dTA = no_deriv
            
            dS_DIC_dN = -self.app.C_P * dS_P_dN - self.app.C_Z * dS_Z_dN - self.app.gamma_c * self.app.C_P * dU_P_dN
            dS_DIC_dP = -self.app.C_P * dS_P_dP - self.app.C_Z * dS_Z_dP - self.app.gamma_c * self.app.C_P * dU_P_dP
            dS_DIC_dZ = -self.app.C_P * dS_P_dZ - self.app.C_Z * dS_Z_dZ
            dS_DIC_dDIC = no_deriv
            dS_DIC_dTA = no_deriv
            
            dS_TA_dN = (-dS_N_dN - 2. * self.app.gamma_c * self.app.C_P * dU_P_dN) / self.app.water_density
            dS_TA_dP = (-dS_N_dP - 2. * self.app.gamma_c * self.app.C_P * dU_P_dP) / self.app.water_density
            dS_TA_dZ = -dS_N_dZ / self.app.water_density
            dS_TA_dDIC = -dS_N_dDIC / self.app.water_density
            dS_TA_dTA = -dS_N_dTA / self.app.water_density
            
            dS_dB = np.block([[diagonalize(dS_N_dN), diagonalize(dS_N_dP), diagonalize(dS_N_dZ), diagonalize(dS_N_dDIC), diagonalize(dS_N_dTA)], [diagonalize(dS_P_dN), diagonalize(dS_P_dP), diagonalize(dS_P_dZ), diagonalize(dS_P_dDIC), diagonalize(dS_P_dTA)], [diagonalize(dS_Z_dN), diagonalize(dS_Z_dP), diagonalize(dS_Z_dZ), diagonalize(dS_Z_dDIC), diagonalize(dS_Z_dTA)], [diagonalize(dS_DIC_dN), diagonalize(dS_DIC_dP), diagonalize(dS_DIC_dZ), diagonalize(dS_DIC_dDIC), diagonalize(dS_DIC_dTA)], [diagonalize(dS_TA_dN), diagonalize(dS_TA_dP), diagonalize(dS_TA_dZ), diagonalize(dS_TA_dDIC), diagonalize(dS_TA_dTA)]])
            
        K_zt = np.stack([self.diff_coeff(self.app.z, t[k]) for k in range(len(t))], axis=0)
        K_zt = np.tile(K_zt, [1, int(self.app.state_dim / self.app.nz)])

        dK_zt_dz = np.einsum('ab, cb -> ca', self.full_vander[1], K_zt)

        dS_ddB_dz = diagonalize(dK_zt_dz)

        dS_dd2B_d2z = diagonalize(K_zt)

        jac = np.stack([dS_dB, dS_ddB_dz, dS_dd2B_d2z], axis=-1)
                        
        return jac
    
    def boundary(self, t, u_t, du_dt):
        
        du_dt = self.separate_states(du_dt)
#         du_dt = np.reshape(du_dt, [-1, du_dt.shape[-1]])
        du_dt_int = du_dt[:, self.app.max_deriv - 1:-(self.app.max_deriv - 1)]
        
        full_vander = [self.full_vander[i][:self.app.nz, :self.app.nz] for i in range(len(self.full_vander))]

        def mat_entry(eqn_id, idx, bnd):
            
            A = full_vander[eqn_id][bnd, idx]
            
            return A
            
        def rhs_entry(eqn_id, bnd):
            
            b = - np.einsum('b, cb -> c', full_vander[eqn_id][bnd, self.app.max_deriv - 1:-(self.app.max_deriv - 1)], du_dt_int)
            return b
        
        B = []
        b = []
        k = 0
        
        for bnd in [0, -1]:
            for i in range(self.app.max_deriv):
                B.append([])
                b.append(rhs_entry(i, bnd))
                
                for j in range(2 * (self.app.max_deriv - 1)):
                    if j < self.app.max_deriv - 1:
                        B[k].append(mat_entry(i, j, bnd))
                    else:
                        B[k].append(mat_entry(i, -(self.app.max_deriv - 1)+(j % (self.app.max_deriv - 1)), bnd))
                   
                B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
                k += 1
             
        B_tmp = [B[0] * self.app.dbc_l + B[1] * self.app.nbc_l]
        b_tmp = [self.app.dleft_bv_dt + (b[0]) * self.app.dbc_l + (b[1]) * self.app.nbc_l]

        for i in range(2, self.app.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i])

        B_tmp.append(B[self.app.max_deriv] * self.app.dbc_r + B[self.app.max_deriv+1] * self.app.nbc_r)
        b_tmp.append(self.app.dright_bv_dt + (b[self.app.max_deriv]) * self.app.dbc_r + (b[self.app.max_deriv+1]) * self.app.nbc_r)

        for i in range(self.app.max_deriv+2, 2*self.app.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i])

        B = tf.concat(B_tmp, axis=-2)
        b = tf.stack(b_tmp, axis=-1)
        
        du_dt_bnd = tf.einsum('ab, cb ->ca', tf.linalg.pinv(B), b)
        
        return du_dt_bnd
    
    def separate_states(self, B):
        B_shp = B.shape
        B = np.reshape(B, [B_shp[0], int(self.app.state_dim / self.app.nz), self.app.nz])
        B = np.reshape(B, [-1, B.shape[-1]])
        return B
    
    def merge_states(self, B):
        B = np.reshape(B, [-1, int(self.app.state_dim / self.app.nz), self.app.nz])
        B = np.reshape(B, [B.shape[0], -1])
        return B
        
def convert_high_complex_to_low_complex_states(x_high_complex, app): 
    true_x_low_complex = []

    true_x_low_complex.append(x_high_complex[:, :, 0*app.nz:1*app.nz] 
                              + x_high_complex[:, :, 3*app.nz:4*app.nz])
    true_x_low_complex.append(x_high_complex[:, :, 1*app.nz:2*app.nz])
    true_x_low_complex.append(x_high_complex[:, :, 2*app.nz:3*app.nz])
    true_x_low_complex.append(x_high_complex[:, :, 4*app.nz:5*app.nz])
    true_x_low_complex.append(x_high_complex[:, :, 5*app.nz:6*app.nz])

    true_x_low_complex = tf.concat(true_x_low_complex, axis=-1)

    return true_x_low_complex