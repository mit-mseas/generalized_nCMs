from src.solvers.neuralDistDDE_train_HOTs import create_interpolator, create_validation_set
import src.utilities.findiff.findiff_general as fdgen

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
        
        loss = tf.math.squared_difference(pred_y, true_y) + tf.keras.backend.epsilon()
        loss_shp = loss.shape
        loss = tf.transpose(tf.reshape(loss, loss_shp[0:2] + [6, self.args.nz]), perm = [0, 1, 3, 2])
        loss = tf.sqrt(tf.reduce_sum(loss, axis=-1))
        loss = tf.reduce_sum(loss, -1) / self.area #Just a hack to get pointwise derivatives
        loss = tf.reduce_mean(loss, axis=0)
        
        return loss
    
    def eval_mode(self, true_y, pred_y):
        loss = tf.math.squared_difference(pred_y, true_y) 
        loss_shp = loss.shape
        loss = tf.transpose(tf.reshape(loss, loss_shp[0:2] + [6, self.args.nz]), perm = [0, 1, 3, 2])
        loss = tf.sqrt(tf.reduce_sum(loss, axis=-1))
        loss = self.integral(loss, self.x) / self.area
        loss = tf.reduce_mean(loss, axis=0)
        
        return loss
    
    def reglularizer(self, model, lambda_l1, lambda_l2):
        weights   = model.trainable_weights
        
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in weights ]) 
        
        lossL1 = tf.add_n([ tf.reduce_sum(tf.abs(tf.reshape(v, [-1]))) for v in weights ]) 
        
        loss = lambda_l1 * lossL1 + lambda_l2 * lossL2
        
        return loss

### Solve for the high complexity model
t_M = np.array([0, 25, 175, 365, 365 + 25, 365 + 175, 365+365])
M = np.array([-80., -25., -25., -80., -25., -25., -80])

fig, ax = plt.subplots()
ax.plot(t_M, M)
ax.set_title('Mixed layer depth', fontsize=14)
ax.set_ylabel('m', fontsize=14)
ax.set_xlabel('t (days)', fontsize=14)
plt.show()

K_z_obj = diff_coeff(args.args_eval_hf, M, t_M, 'linear')

t = tf.linspace(0., args.args_eval_hf.T, args.args_eval_hf.nt) # Time array

#### produce eqb ICs ####
t_ic = tf.linspace(0., 30., 30) # Time array

class initial_cond_npzd:

    def __init__(self, app):
        self.app = app

    def __call__(self, t):

        if self.app.bio_model == 'NPZD-OA':
            x0 = [self.app.T_bio - 3*0.05*self.app.T_bio, 0.05*self.app.T_bio, 0.05*self.app.T_bio, 0.05*self.app.T_bio]
        return tf.expand_dims(tf.concat(x0, axis=0), axis=0)

x0_high_complex_ic = initial_cond_npzd(args.args_train_hf[0]) # Initial conditions

grid_obj = fdgen.grid(args.args_train_hf[0], args.args_train_hf[0].z)
deriv_obj = fdgen.deriv(args.args_train_hf[0], grid_obj)
x_high_complex_ic = ddeinttf(bio.bio_eqn(args.args_train_hf[0], K_z_obj, grid_obj, deriv_obj).initial_cond_solve, x0_high_complex_ic, t_ic, alg_name = args.ode_alg_name, nsteps = args.nsteps)
########################

##### reinitialize using the eqb conditions ########
class initial_cond_reuse:

    def __init__(self, app, x0):
        self.app = app
        self.x0 = x0

    def __call__(self, t):

        OA_0 = [self.app.C_P * self.x0[:, 0*self.app.nz:1*self.app.nz], (198.10 + 61.75 * self.app.Salt(self.app.z, np.array([0.]))) / 1000.]
        
        return tf.concat([self.x0] + OA_0, axis=-1)

x0_high_complex = initial_cond_reuse(args.args_train_hf[0], x_high_complex_ic[-1, ]) # Initial conditions

grid_obj = fdgen.grid(args.args_train_hf[0], args.args_train_hf[0].z)
deriv_obj = fdgen.deriv(args.args_train_hf[0], grid_obj)
x_high_complex = ddeinttf(bio.bio_eqn(args.args_train_hf[0], K_z_obj, grid_obj, deriv_obj), x0_high_complex, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

# Compute FOM for the validation time
dt = t[1] - t[0]
val_t_len =  args.val_percentage * (t[-1] - t[0])
n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)

val_x_high_complex = ddeinttf(bio.bio_eqn(args.args_train_hf[0], K_z_obj, grid_obj, deriv_obj), create_interpolator(x_high_complex, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

print('Higher complexity model done!')


### Transform states of high complexity model to low complexity model

# Create modes for the training and validation period combined
true_x_low_complex = x_high_complex

true_x0_low_complex = x0_high_complex

# Solve the low complexity model
grid_obj = fdgen.grid(args.args_train_hf[0], args.args_train_hf[0].z)
deriv_obj = fdgen.deriv(args.args_train_hf[0], grid_obj)
x_low_complex = ddeinttf(bio.bio_eqn(args.args_eval_lf, K_z_obj, grid_obj, deriv_obj), true_x0_low_complex, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

val_x_low_complex = ddeinttf(bio.bio_eqn(args.args_eval_lf, K_z_obj, grid_obj, deriv_obj), create_interpolator(x_low_complex, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

#### Create validation set
val_obj = create_validation_set(true_x0_low_complex, t, args)

val_true_x_low_complex = val_x_high_complex

grid_obj = fdgen.grid(args.args_train_lf[0], args.args_train_lf[0].z)
deriv_obj = fdgen.deriv(args.args_train_lf[0], grid_obj)
train_rom_rhs_obj = bio.bio_eqn(args.args_train_lf[0], K_z_obj, grid_obj, deriv_obj)

#### Change back to base directory
os.chdir(basedir)