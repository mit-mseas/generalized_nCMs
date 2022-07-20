import src.utilities.findiff.findiff_general as fdgen

class ad_eqn_analy:

    def __init__(self, x, app):
        self.x = x
        self.app = app

    def __call__(self, t):
        u = self.x / (t + 1.) / (1. + np.sqrt(np.divide(t + 1., self.app.t0), dtype = np.float64) * np.exp(self.app.Re * (self.x**2 / ((4. * t) + 4.)), dtype = np.float64))
        return tf.convert_to_tensor([u], dtype=tf.float32)

### Define a custom loss function 
class custom_loss():
    
    def __init__(self, args, x, mse_weight = 0.):
        self.args = args
        self.base_str = 'cdefghijklmnopqrstuvwxyz' # A helper string
        self.area = self.integral(tf.ones(x.shape), x, axis = 0)
        self.x = x
        self.mse_weight = mse_weight
        
    def overwrite(self, args, x):
        self.args = args
        self.x = x
        self.area = self.integral(tf.ones(x.shape), x, axis = 0)
        
    def integral(self, y, x, axis = -1):
        dx = x[1:] - x[0:-1]
        
        y_avg = 0.5 * (tf.gather(y, list(range(0, y.shape[axis]-1)), axis=axis) + tf.gather(y, list(range(1, y.shape[axis])), axis=axis))
        
        var_shape_len = len(tf.shape(y_avg).numpy())
        shape_str = self.base_str[:var_shape_len]

        y_int = tf.einsum(shape_str + ',' + shape_str[axis] + \
                               '->' + shape_str, tf.cast(y_avg, tf.float64), \
                                                        tf.cast(dx, tf.float64))
        
        return tf.cast(tf.reduce_sum(y_int, axis=axis), tf.float32)

    def __call__(self, true_y, pred_y):
        
        loss = tf.sqrt(tf.math.squared_difference(pred_y, true_y) + tf.keras.backend.epsilon()) + self.mse_weight * tf.math.squared_difference(pred_y, true_y)
        loss = tf.reduce_sum(loss, -1) / self.area #Just a hack to get pointwise derivatives
        loss = tf.reduce_mean(loss, axis=0)
        
        return loss
    
    def eval_mode(self, true_y, pred_y):
        loss = tf.abs(pred_y - true_y) 
        loss = self.integral(loss, self.x) / self.area
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(loss, axis=0)
        
        return loss
    
    def reglularizer(self, model, lambda_l1, lambda_l2):
        weights   = model.trainable_weights
        
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in weights ]) 
        
        lossL1 = tf.add_n([ tf.reduce_sum(tf.abs(tf.reshape(v, [-1]))) for v in weights ]) 
        
        loss = lambda_l1 * lossL1 + lambda_l2 * lossL2
        
        return loss

### Solve for the high resolution model for evaluation purpose

t = tf.linspace(0., args.args_eval_hf.T, args.args_eval_hf.nt)

# Compute FOM for the validation time
dt = t[1] - t[0]
val_t_len =  args.val_percentage * (t[-1] - t[0])
n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)

if args.use_analytical_sol:

    ad_eqn_ana_inst = ad_eqn_analy(fdgen.grid(args.args_eval_hf).x_grid_real, args.args_eval_hf)

    u_high_res = []
    for i in range(t.shape[0]):
        u_high_res.append(tf.expand_dims(ad_eqn_ana_inst(t[i]), axis = 0))

    u_high_res = tf.concat(u_high_res, axis=0)
    
    val_u_high_res = []
    for i in range(val_t.shape[0]):
        val_u_high_res.append(tf.expand_dims(ad_eqn_ana_inst(val_t[i]), axis = 0))

    val_u_high_res = tf.concat(val_u_high_res, axis=0)
    
else:

    if args.read_hf_data != 1:
        u_high_res = burger_solve(args.args_eval_hf, t = t, alg_name = args.ode_alg_name, nsteps = args.nsteps)
        val_u_high_res = burger_solve(args.args_eval_hf, u0 = ncm.create_interpolator(u_high_res, t), t = val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)
    else:
        with open('data/burgers_T'+str(int(args.args_eval_hf.T*args.read_hf_data_fac))+'_nt'+str(int(args.args_eval_hf.nt*args.read_hf_data_fac))+'_Nx'+str(int(args.args_eval_hf.Nx))+'_Re'+str(int(args.args_eval_hf.Re))+'_'+args.ode_alg_name+'.pkl', 'rb') as output:
            read_dict = pickle.load(output)

        u_high_res = tf.stack([ncm.create_interpolator(read_dict['u'], read_dict['t'])(t[i]) for i in range(t.shape[0])], axis=0)
        val_u_high_res = tf.stack([ncm.create_interpolator(read_dict['u'], read_dict['t'])(val_t[i]) for i in range(val_t.shape[0])], axis=0)

print('High resolution model done!')

### Solve for low resolution model

u_low_res = burger_solve(args.args_eval_lf, t = t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

val_u_low_res = burger_solve(args.args_eval_lf, u0 = ncm.create_interpolator(u_low_res, t), t = val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

print('Low resolution model done!')

# Interpolate high resolution solution on low resolution grid
grid_obj_hf = fdgen.grid(args.args_eval_hf)
grid_obj_lf = fdgen.grid(args.args_eval_lf)
true_u_low_res = interp_high_res_to_low_res(u_high_res, grid_obj_hf.x_grid, grid_obj_lf.x_grid, t)

val_true_u_low_res = interp_high_res_to_low_res(val_u_high_res, grid_obj_hf.x_grid, grid_obj_lf.x_grid, val_t) 
true_u0 = initial_cond(grid_obj_lf.x_grid_real, args.args_eval_lf)

#### Compute normalization values
grid_obj_hf = fdgen.grid(args.args_norm)
true_u0_hf = initial_cond(grid_obj_hf.x_grid_real, args.args_norm)
deriv_obj = fdgen.deriv(args.args_norm, grid_obj_hf)
norm_eval = []
norm_eval.append(tf.constant(np.max(np.abs(np.squeeze(tf.cast(true_u0_hf(0.), tf.float64).numpy())))))

for i in range(1, args.args_norm.max_deriv+1):
    norm_eval.append(tf.constant(np.max(np.abs(np.squeeze(np.einsum('ab, db -> da', deriv_obj.vander(grid_obj_hf.x_grid_real, m=i), true_u0_hf(0.).numpy()))))))

norm_eval = tf.convert_to_tensor(norm_eval)

### Solve high resolution model for training data

true_u0_ens = []
true_u_low_res_ens = []
norm_train_ens = []
for i in range(args.train_ens_size):

    if args.use_analytical_sol:
        
        ad_eqn_ana_inst = ad_eqn_analy(fdgen.grid(args.args_train_hf[i]).x_grid_real, args.args_train_hf[i])

        u_high_res_tmp = []
        for k in range(t.shape[0]):
            u_high_res_tmp.append(tf.expand_dims(ad_eqn_ana_inst(t[k]), axis = 0))

        u_high_res_tmp = tf.concat(u_high_res_tmp, axis=0)
        
    else:
        
        if args.read_hf_data != 1:
            u_high_res_tmp = burger_solve(args.args_train_hf[i], t = t, alg_name = args.ode_alg_name, nsteps = args.nsteps)
        else:

            with open('data/burgers_T'+str(int(args.args_train_hf[i].T*args.read_hf_data_fac))+'_nt'+str(int(args.args_train_hf[i].nt*args.read_hf_data_fac))+'_Nx'+str(int(args.args_train_hf[i].Nx))+'_Re'+str(int(args.args_train_hf[i].Re))+'_'+args.ode_alg_name+'.pkl', 'rb') as output:
                read_dict = pickle.load(output)

            u_high_res_tmp = tf.stack([ncm.create_interpolator(read_dict['u'], read_dict['t'])(t[i]) for i in range(t.shape[0])], axis=0)
    
    grid_obj_hf = fdgen.grid(args.args_train_hf[i])
    grid_obj_lf = fdgen.grid(args.args_train_lf[i])    
    true_u_low_res_ens.append(interp_high_res_to_low_res(u_high_res_tmp, grid_obj_hf.x_grid, grid_obj_lf.x_grid, t))
    true_u0_ens.append(initial_cond(grid_obj_lf.x_grid_real, args.args_train_lf[i]))
    
    #### Compute normalization values
    norm_train_ens.append(norm_eval)
    
#### Change back to base directory
os.chdir(basedir)