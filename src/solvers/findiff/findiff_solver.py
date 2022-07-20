from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf

########################################################
"""
Generic class to compute the RHS
"""        
class fd_solve():
    
    def __init__(self, args, deriv_obj, grid_obj):
        self.args = args
        self.deriv_obj = deriv_obj
        self.grid = grid_obj
        
    def __call__(self, u, t):
        
        u_t = u(t).numpy()
#         u_t_int = u_t[:, 1:-(self.args.max_deriv - 1)]
        u_t_int = u_t[:, self.args.max_deriv - 1:-(self.args.max_deriv - 1)]
        
        du_dt_int = self.rhs_int(t, u_t, u_t_int)
        du_dt_bnd = self.boundary(t, u_t, du_dt_int)
        
#         du_dt = np.concatenate((du_dt_bnd[:, :1], du_dt_int, du_dt_bnd[:, 1:]), axis=-1)
        
        du_dt = np.concatenate((du_dt_bnd[:, 0:self.args.max_deriv - 1], du_dt_int, du_dt_bnd[:, self.args.max_deriv - 1:]), axis=-1)

        return tf.convert_to_tensor(du_dt, dtype=tf.float32)
        
        
    def rhs_int(self, t, u_t, u_t_int, x_grid_int):
        
        pass
    
    def boundary(self, t, u_t, du_dt_int):
        
#         A_x0 = self.A_x[0, ]
#         A_xN = self.A_x[1, ]
        
#         B = np.array([[self.dbc_l + self.nbc_l*A_x0[0], self.nbc_l*A_x0[-1]], [self.nbc_r*A_xN[0], self.dbc_r + self.nbc_r*A_xN[-1]]])
#         b = np.zeros([u_t.shape[0], 2])
#         b[:, 0] = self.dleft_bv_dt - self.nbc_l*np.einsum('b, cb -> c', A_x0[1:-1], du_dt_int)
#         b[:, 1] = self.dright_bv_dt - self.nbc_r*np.einsum('b, cb -> c', A_xN[1:-1], du_dt_int)

#         du_dt_bnd = np.einsum('ab, cb ->ca', np.linalg.pinv(B), b)

        def mat_entry(eqn_id, idx, bnd):
            
            A = self.full_vander[eqn_id][bnd, idx]
            
            return A
        
#         def rhs_entry(eqn_id, bnd):
            
#             b = - np.einsum('b, cb -> c', self.full_vander[eqn_id][bnd, 1:-(self.args.max_deriv - 1)], du_dt_int)
#             return b
            
        def rhs_entry(eqn_id, bnd):
            
            b = - np.einsum('b, cb -> c', self.full_vander[eqn_id][bnd, self.args.max_deriv - 1:-(self.args.max_deriv - 1)], du_dt_int)
            return b
        
        B = []
        b = []
        k = 0
        
#         bnd = 0
#         for i in range(2):
#             B.append([])
#             b.append(rhs_entry(i, bnd))

#             for j in range(self.args.max_deriv):
#                 if j < 1:
#                     B[k].append(mat_entry(i, j, bnd))
#                 else:
#                     B[k].append(mat_entry(i, -(self.args.max_deriv - 1)+((j-1) % (self.args.max_deriv - 1)), bnd))

#             B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
#             k += 1
            
#         bnd = -1
#         for i in range(self.args.max_deriv):
#             B.append([])
#             b.append(rhs_entry(i, bnd))

#             for j in range(self.args.max_deriv):
#                 if j < 1:
#                     B[k].append(mat_entry(i, j, bnd))
#                 else:
#                     B[k].append(mat_entry(i, -(self.args.max_deriv - 1)+((j-1) % (self.args.max_deriv - 1)), bnd))

#             B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
#             k += 1
             
#         B_tmp = [B[0] * self.args.dbc_l + B[1] * self.args.nbc_l]
#         b_tmp = [self.args.dleft_bv_dt + (b[0]) * self.args.dbc_l + (b[1]) * self.args.nbc_l]

#         B_tmp.append(B[2] * self.args.dbc_r + B[2+1] * self.args.nbc_r)
#         b_tmp.append(self.args.dright_bv_dt + (b[2]) * self.args.dbc_r + (b[2+1]) * self.args.nbc_r)

#         for i in range(2+2, 2+self.args.max_deriv):
#             B_tmp.append(B[i])
#             b_tmp.append(b[i])
        
        for bnd in [0, -1]:
            for i in range(self.args.max_deriv):
                B.append([])
                b.append(rhs_entry(i, bnd))
                
                for j in range(2 * (self.args.max_deriv - 1)):
                    if j < self.args.max_deriv - 1:
                        B[k].append(mat_entry(i, j, bnd))
                    else:
                        B[k].append(mat_entry(i, -(self.args.max_deriv - 1)+(j % (self.args.max_deriv - 1)), bnd))
                   
                B[k] = tf.expand_dims(tf.stack(B[k], axis = -1), axis = -2)
                k += 1
             
        B_tmp = [B[0] * self.args.dbc_l + B[1] * self.args.nbc_l]
        b_tmp = [self.args.dleft_bv_dt + (b[0]) * self.args.dbc_l + (b[1]) * self.args.nbc_l]

        for i in range(2, self.args.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i])

        B_tmp.append(B[self.args.max_deriv] * self.args.dbc_r + B[self.args.max_deriv+1] * self.args.nbc_r)
        b_tmp.append(self.args.dright_bv_dt + (b[self.args.max_deriv]) * self.args.dbc_r + (b[self.args.max_deriv+1]) * self.args.nbc_r)

        for i in range(self.args.max_deriv+2, 2*self.args.max_deriv):
            B_tmp.append(B[i])
            b_tmp.append(b[i])

        B = tf.concat(B_tmp, axis=-2)
        b = tf.stack(b_tmp, axis=-1)
        
        du_dt_bnd = tf.einsum('ab, cb ->ca', tf.linalg.pinv(B), b)
        
        return du_dt_bnd