import tensorflow as tf
import numpy as np
from sklearn import linear_model
from tqdm import trange
from src.CSGAN_auxfun import array_to_complex
import tensorflow_probability as tfp
from scipy import stats

def fib_sphere(num_points, radius=1):
    radius = tf.cast(radius, dtype=tf.float32)
    ga = (3 - tf.math.sqrt(5.)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * tf.range(num_points, dtype=tf.float32)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = tf.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = tf.math.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * tf.sin(theta)
    x = alpha * tf.cos(theta)

    x_batch = tf.tensordot(radius, x, 0)
    y_batch = tf.tensordot(radius, y, 0)
    z_batch = tf.tensordot(radius, z, 0)

    # expand the dimensions of each coordinate
    shape = x_batch.shape.as_list()
    if len(shape) < 3:
        x_batch = tf.expand_dims(x_batch, 0)
        y_batch = tf.expand_dims(y_batch, 0)
        z_batch = tf.expand_dims(z_batch, 0)

    # Stack the x, y, z coordinates into a single array
    grid = tf.stack([x_batch, y_batch, z_batch], axis=-1)
    if len(grid.shape.as_list()) > 3:
        grid = tf.squeeze(grid, axis=0)
    xyz = tf.transpose(grid, perm=[0, 2, 1])
    return xyz

def wavenumber(f, n_PW, c = 343.):
    k = 2*np.pi*f/c
    k_grid = fib_sphere(n_PW, k)
    return k_grid

def build_sensing_mat(k, grid):
    """ Build the sensing matrix for a given set of plane waves and microphones

    Args:
        k (tf.Tensor): 3 x n_plane_waves tensor, where n_plane_waves is the number of plane waves
        grid (tf.Tensor): 3 x n_microphones tensor, where n_microphones is the number of microphones

    Returns:
        tf.Tensor: n_microphones x n_plane_waves tensor, where each row corresponds to a microphone and
            each column corresponds to a plane wave
        """
    # k is a 3 x n_plane_waves tensor, where n_plane_waves is the number of plane waves
    # grid is a 3 x n_microphones tensor, where n_microphones is the number of microphones
    # Returns a n_microphones x n_plane_waves tensor, where each row corresponds to a microphone and
    # each column corresponds to a plane wave
    # Compute the dot product between k and x
    dot_product = tf.einsum('oij,ik->ojk', k, grid)

    # Compute the sensing matrix using the dot product and the exponential function
    sensing_matrix = tf.math.exp(-tf.complex(0., dot_product))

    # Transpose the sensing matrix to get the desired shape
    return tf.transpose(sensing_matrix, perm=[0, 2, 1])


@tf.function
def tf_sensing_mat(f, n_pw, grid, k_samp=None, c=343):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c)
    H = build_sensing_mat(k_samp, grid)
    return H, k_samp

def scale_linear_regression(x, y):
    """
    This function performs a linear regression on the input data and returns the
    slope and intercept of the regression line.
    """
    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept


# @tf.function
def MAC_loss(p, p_hat, returnMAC=False):
    p = tf.cast(p, p_hat.dtype)

    numer = tf.abs(tf.experimental.numpy.vdot(p, p_hat))
    denom = tf.abs(tf.linalg.norm(p, ord=2) * tf.linalg.norm(p_hat, ord=2))

    MACLoss = tf.divide(numer, denom)
    if returnMAC:
        return MACLoss
    else:
        return (1 - MACLoss)


def normalize_pressure(p, normalization='l2norm', epsilon=1e-8):
    # assert p.ndim == 1
    if normalization == 'maxabs':
        return p / tf.reduce_max(tf.abs(p)), tf.reduce_max(abs(p))
    if normalization == 'l2norm':
        return p / tf.linalg.norm(p), tf.linalg.norm(p)
    if normalization == 'standardization':
        mu = p.mean()
        var = p.var()
        pnorm = (p - mu) / np.sqrt(var + epsilon)
        return pnorm, (mu, var)


def adaptive_CSGAN(generator,
                   optimizer_z,
                   optimizer_w,
                   Pmeasured,
                   grid_measured,
                   frq,
                   z_hat,
                   max_itr=200,
                   k_vec=None,
                   complex_net=False,
                   conditional_net=False,
                   lambda_=0.1,
                   verbose=True):
    # Instatiate (frequency) label for generator input
    freq = tf.constant(np.atleast_2d(frq))
    frq = np.atleast_1d(frq)
    H, k = tf_sensing_mat(frq,
                          generator.output.shape[1],
                          grid_measured,
                          k_samp=k_vec)
    z_hat = tf.Variable(z_hat, name='z_hat')
    generator.trainable = True
    t = trange(max_itr, desc='Loss', leave=True, position=0, disable=not verbose)
    losses_collection = []
    for i in t:
        with tf.GradientTape() as tz, tf.GradientTape() as tw:
            tz.watch(z_hat)
            # H = tf.transpose(H)
            if conditional_net:
                fake_coefficients = generator([z_hat, freq])
            else:
                fake_coefficients = generator(z_hat)
            if not complex_net:
                fake_coefficients = array_to_complex(fake_coefficients)
            fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients)
            # fake_sound_fields, _ = normalize_pressure(fake_sound_fields)

            znorm = tf.cast(tf.norm(z_hat, ord=2), dtype=tf.float32)
            # znorm = tf.norm(z_hat, ord=2)
            tz.watch(fake_coefficients)
            tw.watch(fake_coefficients)
            misfit = tf.reduce_mean(tf.abs(Pmeasured - fake_sound_fields) ** 2)
            regulariser = lambda_ * tf.norm(fake_coefficients, ord=1)
            regulariser = tf.cast(regulariser, misfit.dtype)
            loss = misfit + regulariser
            losses_collection.append(misfit)
            t.set_description(
                "iter: {}, Loss: {:.3e}, misfit: {:.3e}, R(G(z)): {:.3e}".format(i, loss, misfit, regulariser),
                refresh=True)
        dJdz = tz.gradient(loss, [z_hat])
        optimizer_z.apply_gradients(zip(dJdz, [z_hat]))
        dJdw = tw.gradient(loss, generator.trainable_variables)
        optimizer_w.apply_gradients(zip(dJdw, generator.trainable_variables))

    if conditional_net:
        inputs = [z_hat, freq]
    else:
        inputs = [z_hat]
    outputs = {}
    outputs['Ginputs'] = inputs
    outputs['zhat'] = z_hat
    outputs['wavenumber_vec'] = k
    outputs['Gweights'] = generator.get_weights()
    outputs['H'] = H

    return outputs


get_latent = lambda shape: tf.complex(tf.random.normal(shape, mean=0, stddev=.5),
                                      tf.random.normal(shape, mean=0, stddev=.5))


def CSGAN(generator,
          optimizer,
          Pmeasured,
          grid_measured,
          frq,
          n_resets=5,
          max_itr=1000,
          complex_net=False,
          conditional_net=False,
          lambda_z=0.1,
          verbose=True):
    # Instatiate (frequency) label for generator input
    freq = tf.constant(np.atleast_2d(frq))
    loss_best = 1e10
    frq = np.atleast_1d(frq)

    H, k = tf_sensing_mat(frq,
                          generator.output.shape[1],
                          grid_measured)

    for ii in range(n_resets):
        if complex_net:
            z_batch = get_latent((1, 128))
        else:
            z_batch = tf.random.normal((1, 128), 0, 1)
        z_batch = tf.Variable(z_batch, name='latent_z')
        t = trange(max_itr, desc='Loss', leave=True, position=0, disable=not verbose)
        for i in t:
            with tf.GradientTape() as tape:
                tape.watch(z_batch)
                if conditional_net:
                    fake_coefficients = generator([z_batch, freq], training=False)
                else:
                    fake_coefficients = generator(z_batch, training=False)
                if not complex_net:
                    fake_coefficients = array_to_complex(fake_coefficients)
                fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients)
                misfit = tf.reduce_mean((tf.abs(Pmeasured - fake_sound_fields) ** 2))

                znorm = tf.norm(z_batch, ord=2)
                znorm = tf.cast(znorm, dtype=misfit.dtype)

                loss = misfit + lambda_z * znorm  # + .1*MACLoss

                t.set_description(
                    "iter: {} loss: {:.3E}, misfit: {:.3E}, || z ||_2 : {:.3E} ".format(i, loss, misfit, znorm),
                    refresh=True)

            gradients = tape.gradient(loss, [z_batch])
            optimizer.apply_gradients(zip(gradients, [z_batch]))
        if tf.abs(loss) < loss_best:
            zhat = tf.identity(z_batch)
            loss_best = tf.abs(loss)

    outputs = {}
    if conditional_net:
        inputs = [zhat, freq]
    else:
        inputs = [zhat]
    outputs['Ginputs'] = inputs
    outputs['zhat'] = zhat
    outputs['wavenumber_vec'] = k
    outputs['H'] = H
    return outputs


def CSGAN_Sparse(generator,
                 optz,
                 optv,
                 Pmeasured,
                 grid_measured,
                 frq,
                 n_resets=5,
                 max_itr=1000,
                 conditional_net=False,
                 complex_net=False,
                 lambda_v=0.01,
                 verbose=True):
    # Instatiate (frequency) label for generator input
    freq = tf.constant(np.atleast_2d(frq))
    loss_best = 10000
    frq = np.atleast_1d(frq)

    H, k = tf_sensing_mat(frq,
                          generator.output.shape[1],
                          grid_measured)

    for ii in range(n_resets):
        if complex_net:
            z_batch = get_latent((1, 128))
        else:
            z_batch = tf.random.normal((1, 128), 0, 0.1)
        z_batch = tf.Variable(z_batch, name='latent_z')
        v_spar = tf.Variable(initial_value=tf.random.normal(shape=(H.shape[-1], 2), dtype=tf.float32),
                             constraint=lambda t: tf.clip_by_norm(t, 1.))
        t = trange(max_itr, desc='Loss', position=0, leave=True, disable=not verbose)
        for i in t:
            with tf.GradientTape() as tape_z, tf.GradientTape() as tape_v:
                tape_z.watch(z_batch)
                # tape_v.watch(v_spar)
                # H = tf.transpose(H)
                if conditional_net:
                    fake_coefficients = generator([z_batch, freq], training=False)
                else:
                    fake_coefficients = generator(z_batch, training=False)
                if not complex_net:
                    fake_coefficients = array_to_complex(fake_coefficients)
                sparsepred = fake_coefficients + v_spar
                fake_coefficients_cmplx = array_to_complex(sparsepred)
                fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
                misfit = tf.reduce_mean(tf.abs(Pmeasured - fake_sound_fields) ** 2)
                vnorm = tf.norm(array_to_complex(v_spar), ord=1)
                vnorm = tf.cast(vnorm, dtype=misfit.dtype)
                loss = misfit + lambda_v * vnorm  # + 1*MACLoss

                t.set_description("iter: {} loss: {:.3E} || v ||_1: {:.3E} ".format(i, loss, vnorm), refresh=True)

            dJdz = tape_z.gradient(loss, [z_batch])
            optz.apply_gradients(zip(dJdz, [z_batch]))
            dJdv = tape_v.gradient(loss, [v_spar])
            optv.apply_gradients(zip(dJdv, [v_spar]))

        if tf.abs(loss) < loss_best:
            zhat = tf.identity(z_batch)
            loss_best = tf.abs(loss)
    outputs = {}
    if conditional_net:
        inputs = [zhat, freq]
    else:
        inputs = [zhat]
    outputs['Ginputs'] = inputs
    outputs['zhat'] = zhat
    outputs['wavenumber_vec'] = k
    outputs['v_sparse'] = v_spar
    outputs['H'] = H
    return outputs


def nmse(y_true, y_predicted, db=True):
    nmse_ = np.mean(abs(y_true - y_predicted) ** 2) / np.mean(abs(y_true) ** 2)
    if db:
        nmse_ = 10 * np.log10(nmse_)
    return nmse_


def mac_similarity(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def complex_clip_by_norm(t, norm_from=None):
    if norm_from is None:
        norm = tf.abs(tf.linalg.norm(t))
    else:
        norm = tf.abs(tf.linalg.norm(norm_from))
    stacked_complex = tf.clip_by_norm(tf.stack((tf.math.real(t), tf.math.imag(t)), axis=-1),
                                      norm)
    return tf.complex(stacked_complex[..., 0], stacked_complex[..., 1])


def planewaveGAN(generator,
                 optimizer_x,
                 Pmeasured,
                 grid_measured,
                 frq,
                 z_hat,
                 max_itr=200,
                 complex_net=False,
                 conditional_net=False,
                 k_vec=None,
                 lambda_=0.1,
                 optimizer_theta=None,
                 verbose=True,
                 use_MAC_loss=False,
                 pref=None,
                 grid_ref=None):
    # Instatiate (frequency) label for generator input
    freq = np.asarray(frq)[..., np.newaxis]
    freq = tf.constant(freq[..., np.newaxis])
    frq = np.atleast_1d(frq)
    H, k = tf_sensing_mat(frq,
                          generator.output.shape[1],
                          grid_measured,
                          k_samp=k_vec)
    Href, _ = tf_sensing_mat(frq,
                             generator.output.shape[1],
                             grid_ref,
                             k_samp=k_vec)

    if conditional_net:
        fake_coefficients = generator([z_hat, freq], training=False)
    else:
        fake_coefficients = generator(z_hat, training=False)

    x_hat = fake_coefficients
    if complex_net:
        x = tf.Variable(get_latent((fake_coefficients.shape)), name='x', dtype=fake_coefficients.dtype,
                        constraint=lambda t: complex_clip_by_norm(t, norm_from=x_hat))
    else:
        x = tf.Variable(tf.random.normal(fake_coefficients.shape), name='x', dtype=tf.float32,
                        constraint=lambda t: tf.clip_by_norm(t, tf.linalg.norm(x_hat, ord=1)))

    t = trange(max_itr, desc='Loss', position=0, leave=True, disable=not verbose)
    losses_collection = []
    best_misfit = 1e10
    for i in t:
        with tf.GradientTape() as tx, tf.GradientTape() as t_theta:
            if conditional_net:
                x_hat = generator([z_hat, freq], training=True)
            else:
                x_hat = generator(z_hat, training=True)
            tx.watch(x)
            # H = tf.transpose(H)
            if not complex_net:
                x_ = array_to_complex(x)
            else:
                x_ = x
            fake_sound_fields = tf.einsum('ijk, ik -> ij', H, x_)
            misfit = tf.reduce_mean(tf.abs(Pmeasured - fake_sound_fields))
            regulariser = tf.norm(x - x_hat, ord=1) ** 2
            regulariser = tf.cast(regulariser, misfit.dtype)
            loss = misfit + lambda_ * regulariser
            if use_MAC_loss:
                MACLoss = MAC_loss(Pmeasured, fake_sound_fields)
                loss += MACLoss
            else:
                MACLoss = 0.
            losses_collection.append(loss)
            t.set_description("iter: {} Loss: {:.4e} Misfit: {:.4e} || x - x_hat||_2 ^2: {:.4e}, "
                              "MAC loss : {:.4e}".format(i,
                                                         loss, misfit,
                                                         regulariser,
                                                         MACLoss),
                              refresh=True)
        optimizer_x.minimize(loss, var_list=[x], tape=tx)
        # early stopping criteria
        if i > int(max_itr * 0.8):
            if MACLoss < best_misfit:
                best_misfit = MACLoss
                best_z = z_hat
                best_coefficients = x
        if optimizer_theta is not None:
            optimizer_theta.minimize(loss, var_list=generator.trainable_variables, tape=t_theta)
    if conditional_net:
        inputs = [best_z, freq]
    else:
        inputs = [best_z]
    if not complex_net:
        coefficients = array_to_complex(best_coefficients)
        if use_MAC_loss:
            sf1 = tf.einsum('ijk, ik -> ij', H, coefficients)
            sf2 = tf.einsum('ijk, ik -> ij', H, -coefficients)
            error1 = tf.reduce_mean(tf.abs(Pmeasured - sf1) ** 2)
            error2 = tf.reduce_mean(tf.abs(Pmeasured - sf2) ** 2)
            if error2 < error1:
                coefficients = -coefficients
    else:
        coefficients = x
    final_sf = tf.einsum('ijk, ik -> ij', H, coefficients)
    scale, bias = scale_linear_regression(tf.reshape(final_sf, -1), tf.reshape(Pmeasured, -1))

    outputs = {'Ginputs': inputs, 'coefficients': coefficients, 'zhat': z_hat, 'wavenumber_vec': k,
               'Gweights': generator.get_weights(), 'H': H, 'scale': scale, 'bias': bias}
    return outputs


def optimizer_function_factory(model, loss, z_batch, freq, y_target,
                               conditional_net=False,
                               ):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters

        with tf.GradientTape() as tape:
            tape.watch(z_batch)
            # update the parameters in the model
            # calculate the loss
            if conditional_net:
                coeffs = model([z_batch, freq])
            else:
                coeffs = model(z_batch)

            loss_value = loss(coeffs, y_target, z_batch)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, z_batch)
        # grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        # tf.print("Iter:", f.iter, "loss:", loss_value)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    return f


def CSGAN_LBGFS(generator,
                Pmeasured,
                grid_measured,
                frq,
                max_itr=200,
                lambda_z=0.001,
                complex_net=False,
                conditional_net=False,
                k_vec=None):
    # Instatiate (frequency) label for generator input
    freq = tf.constant(np.atleast_2d(frq))
    loss_best = 10000
    frq = np.atleast_1d(frq)
    H, k = tf_sensing_mat(frq,
                          generator.output.shape[1],
                          grid_measured[0],
                          grid_measured[1],
                          grid_measured[2],
                          k_samp=k_vec)

    def lbfgs_loss_fn(coefficients, y_target, z_batch):
        if not complex_net:
            fake_coefficients_cmplx = array_to_complex(coefficients)
        else:
            fake_coefficients_cmplx = coefficients
        fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
        znorm = tf.abs(tf.norm(z_batch))
        misfit = tf.reduce_sum(tf.abs(y_target - fake_sound_fields) ** 2, axis=[-2, -1]) + lambda_z * znorm
        misfit = tf.expand_dims(misfit, axis=-1)
        return misfit

    if not complex_net:
        z_init = tf.Variable(tf.random.normal([1, 128]))
    else:
        z_init = tf.Variable(get_latent([1, 128]))
    # this function makes it possible to use L-BFGS optimizer in tensorflow
    func = optimizer_function_factory(generator, lbfgs_loss_fn, z_init, freq, Pmeasured,
                                      conditional_net=conditional_net)

    tolerance = 1
    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func,
                                           initial_position=z_init,
                                           max_iterations=50000,
                                           tolerance=tolerance)

    z_out = results.position
    if conditional_net:
        inputs = [z_out, freq]
        x = generator([z_out, freq])
    else:
        inputs = [z_out]
        x = generator(z_out)
    if complex_net:
        coeffs = x
    else:
        coeffs = array_to_complex(x)
    outputs = {'Ginputs': inputs, 'coefficients': coeffs, 'zhat': z_out, 'wavenumber_vec': k, 'H': H}
    del z_init
    return outputs


def total_var_norm(input, size_dim=1):
    input = tf.cast(input, tf.complex64)
    size = input.shape[size_dim]
    inp_real = tf.math.real(input)
    inp_imag = tf.math.imag(input)
    L = np.diff(np.eye(size), prepend=0).astype(np.float32)
    L_oper = tf.linalg.LinearOperatorFullMatrix(L)
    tv = tf.norm(L_oper.matmul(inp_real) + L_oper.matmul(inp_imag), ord=1)
    return tv


def infer_sf(generator, inputs, frq, grid, sparsegen=False, ptych=False, scale=False):
    # if not isinstance(frq, np.ndarray):
    Ginputs = inputs['Ginputs']
    if "Gweights" in inputs:
        weights = inputs['Gweights']
        generator.set_weights(weights)
    k_vec = inputs['wavenumber_vec']

    frq = np.atleast_1d(frq)
    pred_coefficients = generator(Ginputs, training=False)
    if sparsegen:
        vsparse = inputs['v_sparse']
        if pred_coefficients.dtype == tf.complex64:
            pred_coefficients_cmplx = pred_coefficients + vsparse
        else:
            pred_coefficients_cmplx = array_to_complex(pred_coefficients + vsparse)
    elif ptych:
        pred_coefficients_cmplx = inputs['coefficients']
    else:
        if pred_coefficients.dtype == tf.complex64:
            pred_coefficients_cmplx = pred_coefficients
        else:
            pred_coefficients_cmplx = array_to_complex(pred_coefficients)

    H, k = tf_sensing_mat(frq, generator.output.shape[1], grid, k_samp=k_vec)
    pred_sound_fields = tf.einsum('ijk, ik -> ij', H, pred_coefficients_cmplx)
    pred_sound_fields = tf.squeeze(pred_sound_fields)
    if scale:
        pred_sound_fields = inputs['scale'] * pred_sound_fields + inputs['bias']
    return pred_sound_fields, H


def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


def Ridge_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.
    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1e-2, 1e-8, 50),
                                   fit_intercept=False)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept=False)

    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk


def Lasso_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.
    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.LassoCV(cv=5, alphas=np.geomspace(1e-2, 1e-8, 50),
                                   fit_intercept=False)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Lasso(alpha=alpha_titk, fit_intercept=False)

    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk
