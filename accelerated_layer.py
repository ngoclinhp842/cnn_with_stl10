'''accelerated_layer.py
Extremely fast Conv2d, Maxpool2D neural network layers using JAX.
Oliver W. Layton
CS 343: Neural Networks
Project 3: Convolutional Neural Networks
'''
import numpy as np
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

import layer
from filter_ops import get_pooling_out_shape


@jit
def conv2d_forward_im2col_jax(inputs, kers, b):
    '''Computes the forward pass thru Conv2D layer using the im2col algorithm.

    Parameters:
    ----------
    inputs: ndarray. shape=(batch_sz, n_chans, img_y, img_x).
        Inputs to the Conv2D layer.
    kers: ndarray. shape=(n_kers, D, ker_x, ker_y).
        Convolution layer kernels.
    b: ndarray. shape=(n_kers,).
        Convolution layer bias.

    Returns:
    -------
    net_in: JAX array. shape=(batch_sz, n_kers, img_y, img_x).
        Net input to the Conv2D layer.
    input_cols: JAX array. shape=(D*ker_x*ker_y, img_y*img_x*batch_sz)
        The mini-batch of images flattened into an array of columns of values that fall inside each filter window in
        each position of each image.
    '''
    batch_sz, n_chans, img_y, img_x = inputs.shape
    n_kers, n_ker_chans, ker_x, ker_y = kers.shape

    # (75, 32, 32, 512) = (ker_sz*ker_sz*D, I_y, I_x, N)
    # Want:               (D*ker_sz*ker_sz, I_y*I_x*N)
    input_cols = lax.conv_general_dilated_patches(inputs,
                                                  filter_shape=(ker_y, ker_x),
                                                  window_strides=(1, 1),
                                                  padding='same',
                                                  dimension_numbers=('NCHW', 'OIHW', 'CHWN'))
    # (D, ker_sz, ker_sz, I_y, I_x, N)
    input_cols = input_cols.reshape(n_chans, ker_x, ker_y, img_y, img_x, batch_sz)
    # (D*ker_sz*ker_sz, I_y*I_x*N)
    input_cols = input_cols.reshape(n_chans*ker_x*ker_y, img_y*img_x*batch_sz)
    net_in = kers.reshape(len(kers), -1) @ input_cols + b.reshape(-1, 1)
    # (n_kers, I_y, I_x, N)
    net_in = net_in.reshape(n_kers, img_y, img_x, batch_sz)
    # (N, n_kers, I_y, I_x)
    net_in = net_in.transpose(3, 0, 1, 2)
    return net_in, input_cols

@jit
def conv2d_backward_jax(d_upstream, inputs, input_cols, kers):
    '''Computes the backwards pass thru the Conv2D layer using JAX and tranposed convolution.

    Parameters:
    ----------
    d_upstream: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
        Upstream gradient arriving at the Conv2D layer.
    inputs: ndarray. shape=(batch_sz, n_chans, img_y, img_x).
        Inputs to the Conv2D layer from the forward pass.
    kers: ndarray. shape=(n_kers, D, ker_x, ker_y).
        Convolution layer kernels.

    Returns:
    -------
    dprev_net_act: JAX array. shape=(batch_sz, n_kers, img_y, img_x).
        Gradient flowing out of the 2D convolution layer below to the previous layer.
    d_wts: JAX array. shape=(n_kers, D, ker_x, ker_y).
        Gradient with respect to the Conv2D kers.
    d_b: JAX array. shape=(n_kers,).
        Gradient with respect to the Conv2D bias neurons.
    '''
    batch_sz, n_chans, img_y, img_x = inputs.shape
    n_kers, n_ker_chans, ker_x, ker_y = kers.shape

    # bias gradient
    d_b = jnp.sum(d_upstream, axis=(0, 2, 3))

    # wt gradient
    # reshape upstream grad to be compatible with im2col format
    # (n_kers, img_y, img_x, N)
    d_upstream_flat = jnp.transpose(d_upstream, (1, 2, 3, 0))
    # (n_kers, img_y*img_x*N)
    d_upstream_flat = d_upstream_flat.reshape(n_kers, -1)
    # Use the im2col 2D representation of minibatch images to compute the weight gradient
    d_wts = d_upstream_flat @ input_cols.T
    # Inflate from 2D to 4D
    d_wts = d_wts.reshape(kers.shape)

    # prev layer act gradient
    # Do transposed convolution to get the input gradient.
    # We need to prepare the kernels then do a regular convolution. This is a 2 step process.
    # Transpose n_kers and D
    kers_t = jnp.swapaxes(kers, 0, 1)
    # Rot kers spatially 180 deg
    kers_t = jnp.flip(kers_t, axis=(2, 3))
    # Now do the convolution to get the input gradient. Should have same shape as input
    dprev_net_act = lax.conv_general_dilated(d_upstream,
                                             kers_t,
                                             window_strides=(1, 1),
                                             padding='same',
                                             dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

    return dprev_net_act, d_wts, d_b


@partial(jit, static_argnames=['out_dim', 'pool_size'])
def maxpool2d_forward_jax(input, out_dim, pool_size):
    '''Computes the MaxPool2D layer forward pass by strategic reshaping and summing of the input.

    Parameters:
    ----------
    inputs: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
        Inputs to the MaxPool2D layer from the forward pass.
    out_dim: int.
        The height/width of the spatial dimensions after the max pooling has been applied. Should be computed by the
        max pooling output shape eq.
    pool_size: int.
        The height/width of the pooling window.

    Returns:
    -------
    net_in: JAX array. shape=(batch_sz, n_kers, img_y, img_x).
        Net input to the MaxPool2D layer.
        NOTE: The img_y, img_x in the shape are just placeholders for the spatial dims. In most situations, the spatial
        dims will decrease in size.
    input_reshaped: JAX array. shape=(batch_sz, n_chans, out_dim, pool_size, out_dim, pool_size).
        The reshaped mini-batch. Needed for the backward pass.
    '''
    mini_batch_sz, n_chans, img_y, img_x = input.shape

    # Partition the input into pool_sz chunks, then take the max within each chunk
    input_reshaped = jnp.reshape(input, [mini_batch_sz, n_chans, out_dim, pool_size, out_dim, pool_size])
    net_in = input_reshaped.max(axis=3).max(axis=4)
    return net_in, input_reshaped

@jit
def maxpool2d_backward_jax(d_upstream, input, input_reshaped, net_in):
    '''Computes the backwards pass thru the MaxPool2D layer using JAX.

    Parameters:
    ----------
    d_upstream: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
        Upstream gradient arriving at the MaxPool2D layer.
    input: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
        Inputs to the MaxPool2D layer from the forward pass.
    input_reshaped: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
        Reshaped inputs from the forward pass before the max pooling operation was applied.
    kers: ndarray. shape=(n_kers, D, ker_x, ker_y).
        Convolution layer kernels.
    net_in: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
        Net input to the MaxPool2D layer.

    Returns:
    -------
    dprev_net_act: JAX array. shape=(batch_sz, n_kers, img_y, img_x).
        Gradient flowing out of the max pooling layer below to the previous layer layer.
    '''
    # Find where net_in matches input (where the max vals are)
    net_in = net_in[:, :, :, jnp.newaxis, :, jnp.newaxis]
    maxes = input_reshaped == net_in

    # Find matching values in upstream gradient and in destination gradient passed to prev layer
    d_upstream = d_upstream[:, :, :, jnp.newaxis, :, jnp.newaxis]
    dprev_net_act_rs = jnp.zeros_like(input_reshaped)
    d_upstream_broadcast, _ = jnp.broadcast_arrays(d_upstream, dprev_net_act_rs)
    dprev_net_act_rs = jnp.where(maxes, d_upstream_broadcast, dprev_net_act_rs)

    dprev_net_act_rs /= jnp.sum(maxes, axis=(3, 5), keepdims=True)
    dprev_net_act = dprev_net_act_rs.reshape(input.shape)
    return dprev_net_act


class Conv2DAccel(layer.Conv2D):
    '''Fast Conv2D layer'''
    def compute_net_in(self):
        '''Uses JAX to perform 2D convolution on the mini-batch of images.'''
        net_in, input_cols = conv2d_forward_im2col_jax(self.input, self.wts, self.b)
        self.net_in = np.asarray(net_in)
        self.input_cols = np.asarray(input_cols)

    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes fast convolution backward pass using the im2col algorithm and JAX.

        Parameters:
        ----------
        d_upstream: ndrarray. shape=(batch_sz, n_kers, img_y, img_x).
            The upstream gradient flowing into the current Conv2D layer.

        Returns:
        -------
        dprev_net_act: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
            Gradient flowing out of the 2D convolution layer below to the previous layer.
        d_wts: ndarray. shape=(n_kers, D, ker_x, ker_y).
            Gradient with respect to the Conv2D kers.
        d_b: ndarray. shape=(n_kers,).
            Gradient with respect to the Conv2D bias neurons.
        '''
        dprev_net_act, d_wts, d_b = conv2d_backward_jax(d_upstream, self.input, self.input_cols, self.wts)
        dprev_net_act = np.asarray(dprev_net_act)
        d_wts = np.array(d_wts)
        d_b = np.asarray(d_b)

        # regularize the weight gradient
        d_wts += self.reg*self.wts

        return dprev_net_act, d_wts, d_b


class MaxPool2DAccel(layer.MaxPool2D):
    '''Fast 2D max pooling layer.'''
    def compute_net_in(self):
        '''Computes fast 2D max pooling net-in using reshaping (partitioning input into small windows).'''
        mini_batch_sz, n_chans, img_y, img_x = self.input.shape
        out_dim = get_pooling_out_shape(img_x, self.pool_size, self.strides)

        net_in, input_reshaped = maxpool2d_forward_jax(self.input, out_dim, self.pool_size)
        self.net_in = np.asarray(net_in)
        self.input_reshaped = np.asarray(input_reshaped)


    def backward_netIn_to_prevLayer_netAct(self, d_upstream):
        '''Computes the backwards pass thru the MaxPool2D layer.

        Parameters:
        ----------
        d_upstream: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
            Upstream gradient arriving at the MaxPool2D layer.

        Returns:
        -------
        dprev_net_act: ndarray. shape=(batch_sz, n_kers, img_y, img_x).
            Gradient flowing out of the max pooling layer below to the previous layer layer.
        None
        None
        '''
        net_in = jnp.array(self.net_in)
        input = jnp.array(self.input)
        input_reshaped = jnp.array(self.input_reshaped)
        dprev_net_act = maxpool2d_backward_jax(d_upstream, input, input_reshaped, net_in)
        dprev_net_act = np.asarray(dprev_net_act)

        return dprev_net_act, None, None


if __name__ == '__main__':
    # Forward test
    # batch_sz, n_chans, img_y, img_x = 4, 3, 9, 9
    # n_kers, n_ker_chans, ker_x, ker_y = 2, 3, 5, 5

    # layer = Conv2DAccel2(0, 'testconv', n_kers=n_kers, ker_sz=ker_x)

    # np.random.seed(0)
    # layer.input = np.random.random([batch_sz, n_chans, img_y, img_x])
    # # print(layer.input)
    # layer.wts = np.random.random([n_kers, n_ker_chans, ker_x, ker_y ])
    # layer.b = np.random.random(n_kers)
    # layer.compute_net_in()
    # print(layer.input_cols[1, 0:50])
    # print(layer.input_cols[1, 100:150])
    # print(layer.input_cols[20, 200:250])
    # print(layer.net_in.ravel()[400:450])

    # Backward test
    batch_sz, n_chans, img_y, img_x = 4, 3, 12, 12
    n_kers, n_ker_chans, ker_x, ker_y = 2, 3, 5, 5

    layer = Conv2DAccel(0, 'testconv', n_kers=n_kers, ker_sz=ker_x)

    rng = np.random.default_rng(0)
    layer.input = rng.random([batch_sz, n_chans, img_y, img_x])
    # print(layer.input)
    layer.wts = rng.random([n_kers, n_ker_chans, ker_x, ker_y ])
    layer.b = rng.random(n_kers)
    d_upstream = rng.random([batch_sz, n_kers, img_y, img_x])
    layer.compute_net_in()
    dprev_net_act, d_wts, d_b = layer.backward_netIn_to_prevLayer_netAct(d_upstream)
    print(dprev_net_act.ravel()[:20])
    # print(dprev_net_act.shape)
